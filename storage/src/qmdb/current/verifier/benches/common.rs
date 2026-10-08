//! Shared setup for the prover and verifier benchmarks.
//!
//! Database population, proof construction, and model validation happen before timing, the
//! first time Criterion runs a case, so listing or filtering cases builds nothing. Both
//! strategies use the same Tokio storage backend and seeded workload.

use super::harness;
use commonware_cryptography::Sha256;
use commonware_parallel::Strategy;
use commonware_storage::{
    merkle::Graftable,
    qmdb::{
        any::value::FixedEncoding,
        current::verifier::{BatchProofBundle, CurrentCommitment, TrustedMap, UserBatch},
        floor::Proportional,
    },
};
use std::collections::BTreeMap;

/// Initial key counts. Upstream's nightly benchmark job builds with `--cfg full_bench`.
#[cfg(not(full_bench))]
pub(crate) const KEY_COUNTS: [u64; 1] = [10_000];
#[cfg(full_bench)]
pub(crate) const KEY_COUNTS: [u64; 2] = [1_000, 10_000];

/// Actions per measured batch.
#[cfg(not(full_bench))]
pub(crate) const ACTION_COUNTS: [usize; 3] = [1, 128, 2048];
#[cfg(full_bench)]
pub(crate) const ACTION_COUNTS: [usize; 5] = [1, 32, 128, 512, 2048];

/// Rayon workers for the parallel cases; the others run sequentially.
pub(crate) const PARALLEL_THREADS: usize = 8;

/// Commitment for the shared fixed-value fixture.
pub(crate) type Commitment<F> = CurrentCommitment<
    F,
    harness::Key,
    FixedEncoding<harness::Value>,
    Sha256,
    { harness::CHUNK_BYTES },
>;

/// Complete proof bundle for the shared fixed-value fixture.
pub(crate) type Proof<F> = BatchProofBundle<
    F,
    harness::Key,
    FixedEncoding<harness::Value>,
    Sha256,
    { harness::CHUNK_BYTES },
>;

/// Validated database, batch, and proof shared by both timed operations.
pub(crate) struct Fixture<
    F: Graftable,
    E: commonware_storage::Context + commonware_runtime::Spawner,
    S: Strategy,
> {
    pub(crate) database: harness::ScenarioDb<F, E, S>,
    pub(crate) batch: UserBatch<harness::Key, harness::Value>,
    pub(crate) pre: Commitment<F>,
    pub(crate) proof: Proof<F>,
    pub(crate) post: Commitment<F>,
}

/// Prepare the seeded database and normalized batch outside the measured region.
pub(crate) async fn prepare<
    F: Graftable,
    E: commonware_storage::Context + commonware_runtime::Spawner,
    S: Strategy,
>(
    context: E,
    key_count: u64,
    action_count: usize,
    seed: u64,
    partition: &str,
    strategy: S,
) -> Fixture<F, E, S> {
    let mut database =
        harness::open_db_with_strategy::<F, _, _>(context, partition, strategy).await;
    let mut generator = harness::SeededWorkload::new(seed, key_count * 2);
    let initial = generator.initial_batch(key_count);
    let mut model = BTreeMap::new();
    initial.apply_model(&mut model);
    let initial = initial
        .batch
        .merkleize(&database, &mut Proportional)
        .await
        .unwrap();
    database = database.apply_batch(initial).await.unwrap();

    let logical = generator.next_batch(action_count);
    logical.apply_model(&mut model);
    let batch = logical.batch;
    let pre = Commitment::<F>::from_parts(database.bounds().end, database.root()).unwrap();
    let proof = batch
        .build_batch_proof(&database, &Proportional)
        .await
        .unwrap();
    let merkleized = batch.merkleize(&database, &mut Proportional).await.unwrap();
    let post =
        Commitment::<F>::from_parts(merkleized.bounds().tip.size, merkleized.root()).unwrap();
    assert_eq!(
        proof.claims.post_floor,
        merkleized.bounds().inactivity_floor
    );
    for index in 0..key_count * 2 {
        let key = harness::key(index);
        assert_eq!(
            merkleized.get(&key, &database).await.unwrap(),
            model.get(&key).cloned()
        );
    }
    let verified = batch
        .verify(
            pre,
            TrustedMap::empty_cache(pre),
            &proof,
            post,
            &mut Proportional,
        )
        .unwrap();
    assert_eq!(verified.commitment(), &post);
    Fixture {
        database,
        batch,
        pre,
        proof,
        post,
    }
}
