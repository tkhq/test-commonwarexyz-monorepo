//! Criterion measurements for stateless verification.
//!
//! The timed operation is `UserBatch::verify` from an empty trusted cache, without carrying the
//! facts forward. Database initialization, the seeded bootstrap and batch, proof construction,
//! and post-commitment construction are outside the timer. No malformed or adversarial proof is
//! used here; those cases belong to the fuzzer and correctness tests.

use super::common::{ACTION_COUNTS, Fixture, KEY_COUNTS, PARALLEL_THREADS, prepare};
use commonware_parallel::{Rayon, Sequential, Strategy};
use commonware_runtime::{Runner as _, tokio};
use commonware_storage::{
    merkle::{Graftable, mmb, mmr},
    qmdb::{current::verifier::TrustedMap, floor::Proportional},
};
use commonware_utils::NZUsize;
use criterion::{Criterion, criterion_group};
use std::hint::black_box;

fn bench_family<F: Graftable, S: Strategy>(
    c: &mut Criterion,
    family: &str,
    strategy: S,
    threads: usize,
) {
    for key_count in KEY_COUNTS {
        for action_count in ACTION_COUNTS {
            let seed = 0x5eed_u64;
            // Verification needs no database, so drop it once the fixture is built.
            let mut fixture = None;
            c.bench_function(
                &format!("{}/family={family} keys={key_count} actions={action_count} seed={seed} threads={threads}", module_path!()),
                |b| {
                    let (batch, pre, proof, post) = fixture.get_or_insert_with(|| {
                        let strategy = strategy.clone();
                        tokio::Runner::default().start(|context| async move {
                            let Fixture { database, batch, pre, proof, post, .. } = prepare::<F, _, _>(
                                context,
                                key_count,
                                action_count,
                                seed,
                                "stateless-verifier-bench",
                                strategy,
                                0,
                                false,
                            )
                            .await;
                            database.destroy().await.unwrap();
                            (batch, pre, proof, post)
                        })
                    });
                    b.iter(|| {
                        black_box(
                            batch
                                .verify_with_strategy(
                                    *pre,
                                    TrustedMap::empty_cache(*pre),
                                    proof,
                                    *post,
                                    &strategy,
                                    &mut Proportional,
                                )
                                .unwrap(),
                        )
                    });
                },
            );
        }
    }
}

fn bench_verifier(c: &mut Criterion) {
    bench_family::<mmr::Family, _>(c, "mmr", Sequential, 1);
    bench_family::<mmb::Family, _>(c, "mmb", Sequential, 1);
    let strategy = Rayon::new(NZUsize!(PARALLEL_THREADS)).unwrap();
    bench_family::<mmr::Family, _>(c, "mmr", strategy.clone(), PARALLEL_THREADS);
    bench_family::<mmb::Family, _>(c, "mmb", strategy, PARALLEL_THREADS);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_verifier
}
