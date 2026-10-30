//! Criterion measurements for proving a batch on top of pending batches.
//!
//! The `prove_on` cases time `UserBatch::prove_on` over `depth` pending batches of the same
//! size, which the database has not applied: reads consult each of their changes before the
//! database's. The `prove_applied` cases prove the same batch from the same state once those
//! batches are applied, so the two differ only in where the state is read from. Each depth
//! proves a different batch, since it follows the batches below it in the seeded workload, so
//! compare the two at one depth, not depths with each other or with [`super::step`]. Setup
//! matches [`super::prover`].

use super::prover::{Timed, bench_all};
use criterion::{Criterion, criterion_group};

/// Pending batches under the measured one.
const DEPTHS: [usize; 3] = [1, 4, 8];

fn bench_chain(c: &mut Criterion) {
    bench_all(
        c,
        &format!("{}::prove_on", module_path!()),
        Timed::Chain,
        &DEPTHS,
    );
}

fn bench_applied(c: &mut Criterion) {
    bench_all(
        c,
        &format!("{}::prove_applied", module_path!()),
        Timed::Applied,
        &DEPTHS,
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_chain, bench_applied
}
