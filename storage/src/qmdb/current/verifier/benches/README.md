# Stateless prover and verifier benchmarks

From the workspace root:

```sh
cargo bench -p commonware-storage --bench stateless_verifier
RUSTFLAGS="--cfg full_bench" cargo bench -p commonware-storage --bench stateless_verifier
```

Criterion times `build_batch_proof`, a writer's step (the bundle and `merkleize`) and
empty-cache `verify` separately, using honest seeded workloads from `harness.rs`. The
verifier case does not call `into_cache`, which a recipient that keeps facts pays in
addition. Fixtures use MMR and MMB, the Tokio storage runtime, EightCap, SHA-256,
8-byte keys and values, and 32-byte bitmap chunks. `threads=1` uses Sequential and
`threads=8` uses Rayon with its default adaptive scheduling.

By default each family runs 10,000 initial keys with 1, 128 and 2,048 actions. With
`--cfg full_bench`, as upstream's nightly benchmark job builds, it also runs 1,000
initial keys and 32 and 512 actions. Range deletes can affect more keys than the
action count.

Each case builds its database the first time Criterion runs it, so listing cases or
filtering by name builds nothing for the cases skipped, and keeps one runtime,
checkpoint, and batch alive through warmup and measurement. Setup, PRNG work,
normalization, and model checks are outside the timers; producing or verifying and
dropping the result are inside. Proving includes internal checks. Add
`-- 'threads=8'` to the benchmark command to select only Rayon cases.
