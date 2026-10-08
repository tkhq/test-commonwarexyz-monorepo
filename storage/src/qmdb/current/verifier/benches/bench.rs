//! Criterion entry point for the stateless verifier prover and verifier benchmarks.

mod common;
mod harness;
mod prover;
mod verifier;

use criterion::criterion_main;

criterion_main!(prover::benches, verifier::benches);
