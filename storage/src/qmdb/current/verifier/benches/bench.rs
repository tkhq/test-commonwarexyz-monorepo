//! Criterion entry point for the stateless verifier prover and verifier benchmarks.

mod chain;
mod common;
mod harness;
mod merkleize;
mod prover;
mod step;
mod verifier;

use criterion::criterion_main;

criterion_main!(
    prover::benches,
    step::benches,
    merkleize::benches,
    chain::benches,
    verifier::benches
);
