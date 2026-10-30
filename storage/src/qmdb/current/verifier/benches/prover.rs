//! Criterion measurements for complete proof construction.
//!
//! The timed operation is `UserBatch::build_batch_proof`, or with [`Timed::Step`] `UserBatch::prove`,
//! which returns the proof and the merkleized batch a writer sends and applies. Both merkleize the
//! batch: `build_batch_proof` drops it. [`Timed::Merkleize`] times the writer alone,
//! [`Timed::Chain`] times `UserBatch::prove_on` over pending batches, and [`Timed::Applied`]
//! proves the same batch once those batches are applied. Database initialization, the
//! seeded bootstrap, batch generation, and runtime setup are outside the timer. The database uses
//! the Tokio storage backend and `EightCap`, with Sequential or Rayon workers.

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

/// What one timed iteration does.
#[derive(Clone, Copy)]
pub(crate) enum Timed {
    /// Build the proof bundle.
    Proof,
    /// Build the proof bundle and keep the merkleized batch.
    Step,
    /// Merkleize the batch through the writer alone.
    Merkleize,
    /// Build the proof bundle and keep the merkleized batch, on top of pending batches.
    Chain,
    /// As [`Timed::Chain`], with the batches below applied first: the same batch from the
    /// same state.
    Applied,
}

pub(crate) fn bench_family<F: Graftable, S: Strategy>(
    c: &mut Criterion,
    name: &str,
    timed: Timed,
    family: &str,
    strategy: S,
    threads: usize,
    depths: &[usize],
) {
    for key_count in KEY_COUNTS {
        for action_count in ACTION_COUNTS {
            for &depth in depths {
                let seed = 0x5eed_u64;
                // The first run of a case starts a runtime thread that builds the database and
                // keeps it alive through warmup and samples. Only the async proof loop is timed
                // there.
                std::thread::scope(|scope| {
                    let mut server = None;
                    c.bench_function(
                        &format!("{name}/family={family} keys={key_count} actions={action_count} depth={depth} seed={seed} threads={threads}"),
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
                                            depth,
                                            matches!(timed, Timed::Applied),
                                        )
                                        .await;
                                        results.send(Duration::ZERO).unwrap();
                                        for iters in requests {
                                            let start = Instant::now();
                                            for _ in 0..iters {
                                                match timed {
                                                    Timed::Proof => drop(black_box(
                                                        fixture
                                                            .batch
                                                            .build_batch_proof(&fixture.database, &Proportional)
                                                            .await
                                                            .unwrap(),
                                                    )),
                                                    Timed::Step | Timed::Applied => drop(black_box(
                                                        fixture
                                                            .batch
                                                            .prove(&fixture.database, &mut Proportional)
                                                            .await
                                                            .unwrap(),
                                                    )),
                                                    Timed::Merkleize => drop(black_box(
                                                        fixture
                                                            .batch
                                                            .merkleize(&fixture.database, &mut Proportional)
                                                            .await
                                                            .unwrap(),
                                                    )),
                                                    Timed::Chain => drop(black_box(
                                                        fixture
                                                            .batch
                                                            .prove_on(
                                                                fixture.pending.last().unwrap(),
                                                                &fixture.database,
                                                                &mut Proportional,
                                                            )
                                                            .await
                                                            .unwrap(),
                                                    )),
                                                }
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
}

/// Run `timed` for both families, sequentially and on Rayon, with each of `depths` pending
/// batches under the measured one.
pub(crate) fn bench_all(c: &mut Criterion, name: &str, timed: Timed, depths: &[usize]) {
    bench_family::<mmr::Family, _>(c, name, timed, "mmr", Sequential, 1, depths);
    bench_family::<mmb::Family, _>(c, name, timed, "mmb", Sequential, 1, depths);
    let strategy = Rayon::new(NZUsize!(PARALLEL_THREADS)).unwrap();
    bench_family::<mmr::Family, _>(
        c,
        name,
        timed,
        "mmr",
        strategy.clone(),
        PARALLEL_THREADS,
        depths,
    );
    bench_family::<mmb::Family, _>(c, name, timed, "mmb", strategy, PARALLEL_THREADS, depths);
}

fn bench_prover(c: &mut Criterion) {
    bench_all(c, module_path!(), Timed::Proof, &[0]);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_prover
}
