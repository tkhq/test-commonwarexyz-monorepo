//! Criterion measurements for the writer alone: `UserBatch::merkleize` of the batch
//! [`super::step`] proves. `prove` runs its opening beside the writer, so the difference between
//! the two is the time proving adds to the writer's, not the bundle's own work. Setup matches
//! [`super::prover`].

use super::prover::{Timed, bench_all};
use criterion::{Criterion, criterion_group};

fn bench_merkleize(c: &mut Criterion) {
    bench_all(c, module_path!(), Timed::Merkleize, &[0]);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_merkleize
}
