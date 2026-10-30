//! Shared setup for the prover and verifier benchmarks.
//!
//! Database population, proof construction, and model validation happen before timing, the
//! first time Criterion runs a case, so listing or filtering cases builds nothing. Both
//! strategies use the same Tokio storage backend and seeded workload.

use super::harness;
use commonware_cryptography::{Sha256, sha256::Digest};
use commonware_parallel::Strategy;
use commonware_storage::{
    merkle::Graftable,
    qmdb::{
        any::{ordered::fixed::Update, value::FixedEncoding},
        current::{
            batch::MerkleizedBatch,
            verifier::{BatchProofBundle, CurrentCommitment, TrustedMap, UserBatch},
        },
        floor::Proportional,
    },
};
use std::{collections::BTreeMap, sync::Arc};

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

/// A merkleized batch the fixture's database has not applied.
pub(crate) type Pending<F, S> = Arc<
    MerkleizedBatch<F, Digest, Update<harness::Key, harness::Value>, { harness::CHUNK_BYTES }, S>,
>;

/// Validated database, batch, and proof shared by both timed operations.
pub(crate) struct Fixture<
    F: Graftable,
    E: commonware_storage::Context + commonware_runtime::Spawner,
    S: Strategy,
> {
    pub(crate) database: harness::ScenarioDb<F, E, S>,
    /// Pending batches under the measured one, oldest first: it is proved on the last.
    pub(crate) pending: Vec<Pending<F, S>>,
    pub(crate) batch: UserBatch<harness::Key, harness::Value>,
    pub(crate) pre: Commitment<F>,
    pub(crate) proof: Proof<F>,
    pub(crate) post: Commitment<F>,
}

/// Prepare the seeded database and normalized batch outside the measured region, with `depth`
/// batches of the same size between them: pending, or with `apply`, applied, so that the same
/// batch is proved from the same state either way.
#[allow(clippy::too_many_arguments)]
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
    depth: usize,
    apply: bool,
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
    let mut pending = Vec::<Pending<F, S>>::new();
    for _ in 0..depth {
        let logical = generator.next_batch(action_count);
        logical.apply_model(&mut model);
        let (_, merkleized) = match pending.last() {
            Some(parent) => logical
                .batch
                .prove_on(parent, &database, &mut Proportional)
                .await
                .unwrap(),
            None => logical
                .batch
                .prove(&database, &mut Proportional)
                .await
                .unwrap(),
        };
        pending.push(merkleized);
    }
    if apply {
        for merkleized in pending.drain(..) {
            database = database.apply_batch(merkleized).await.unwrap();
        }
    }

    let logical = generator.next_batch(action_count);
    logical.apply_model(&mut model);
    let batch = logical.batch;
    let (pre, (proof, merkleized)) = match pending.last() {
        Some(parent) => (
            Commitment::<F>::from_parts(parent.bounds().tip.size, parent.root()).unwrap(),
            batch
                .prove_on(parent, &database, &mut Proportional)
                .await
                .unwrap(),
        ),
        None => (
            Commitment::<F>::from_parts(database.bounds().end, database.root()).unwrap(),
            batch.prove(&database, &mut Proportional).await.unwrap(),
        ),
    };
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
        pending,
        batch,
        pre,
        proof,
        post,
    }
}
