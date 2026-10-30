//! Criterion measurements for a writer's whole step: the proof bundle and the merkleized batch.
//!
//! A writer sends the bundle with the batch's root and applies the merkleized batch, so this is
//! the work one batch costs it. Setup matches [`super::prover`].

use super::prover::{Timed, bench_all};
use criterion::{Criterion, criterion_group};

fn bench_step(c: &mut Criterion) {
    bench_all(c, module_path!(), Timed::Step, &[0]);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_step
}
