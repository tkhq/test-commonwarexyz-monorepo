//! Criterion measurements for complete proof construction.
//!
//! The timed operation is `UserBatch::build_batch_proof` only. Database initialization, the
//! seeded bootstrap, batch generation, and runtime setup are outside the timer. The database
//! uses the Tokio storage backend and `EightCap`, with Sequential or Rayon workers.

use super::common::{ACTION_COUNTS, KEY_COUNTS, PARALLEL_THREADS, prepare};
use commonware_parallel::{Rayon, Sequential, Strategy};
use commonware_runtime::{Runner as _, tokio};
use commonware_storage::{
    merkle::{Graftable, mmb, mmr},
    qmdb::floor::Proportional,
};
use commonware_utils::NZUsize;
use criterion::{Criterion, criterion_group};
use std::{
    hint::black_box,
    sync::mpsc,
    time::{Duration, Instant},
};

fn bench_family<F: Graftable, S: Strategy>(
    c: &mut Criterion,
    family: &str,
    strategy: S,
    threads: usize,
) {
    for key_count in KEY_COUNTS {
        for action_count in ACTION_COUNTS {
            let seed = 0x5eed_u64;
            // The first run of a case starts a runtime thread that builds the database and keeps
            // it alive through warmup and samples. Only the async proof loop is timed there.
            std::thread::scope(|scope| {
                let mut server = None;
                c.bench_function(
                    &format!("{}/family={family} keys={key_count} actions={action_count} seed={seed} threads={threads}", module_path!()),
                    |b| {
                        let (commands, responses) = server.get_or_insert_with(|| {
                            let (commands, requests) = mpsc::channel::<u64>();
                            let (results, responses) = mpsc::channel();
                            let strategy = strategy.clone();
                            scope.spawn(move || {
                                tokio::Runner::default().start(|context| async move {
                                    let fixture = prepare::<F, _, _>(
                                        context,
                                        key_count,
                                        action_count,
                                        seed,
                                        "stateless-prover-bench",
                                        strategy,
                                    )
                                    .await;
                                    results.send(Duration::ZERO).unwrap();
                                    for iters in requests {
                                        let start = Instant::now();
                                        for _ in 0..iters {
                                            black_box(
                                                fixture
                                                    .batch
                                                    .build_batch_proof(&fixture.database, &Proportional)
                                                    .await
                                                    .unwrap(),
                                            );
                                        }
                                        results.send(start.elapsed()).unwrap();
                                    }
                                    fixture.database.destroy().await.unwrap();
                                })
                            });
                            responses.recv().unwrap();
                            (commands, responses)
                        });
                        b.iter_custom(|iters| {
                            commands.send(iters).unwrap();
                            responses.recv().unwrap()
                        });
                    },
                );
                // Closing the command channel lets the runtime thread destroy the database.
                drop(server);
            });
        }
    }
}

fn bench_prover(c: &mut Criterion) {
    bench_family::<mmr::Family, _>(c, "mmr", Sequential, 1);
    bench_family::<mmb::Family, _>(c, "mmb", Sequential, 1);
    let strategy = Rayon::new(NZUsize!(PARALLEL_THREADS)).unwrap();
    bench_family::<mmr::Family, _>(c, "mmr", strategy.clone(), PARALLEL_THREADS);
    bench_family::<mmb::Family, _>(c, "mmb", strategy, PARALLEL_THREADS);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_prover
}
