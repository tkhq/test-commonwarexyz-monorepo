//! Prove transitions at a checkpoint older than the database's current state.
//!
//! The prover reads the key index, activity bitmap, and grafted tree, which exist only at the
//! database's current state. The log and its operations tree keep history, so a scratch database
//! synced from the live one to the old checkpoint reproduces that state, and the batches applied
//! there can be proved from it.

use super::{CurrentCommitment, TrustedMap, UserBatch, UserOp};
use crate::{
    journal::contiguous::fixed::Config as JournalConfig,
    merkle::{Graftable, full::Config as MerkleConfig, mmb, mmr},
    qmdb::{
        self,
        any::{operation::Operation, value::FixedEncoding},
        current::{FixedConfig, db::sync_boundary, ordered::fixed::Db},
        floor::Proportional,
        sync::{self, Target, engine},
    },
    translator::OneCap,
};
use commonware_codec::Encode as _;
use commonware_cryptography::Sha256;
use commonware_parallel::Sequential;
use commonware_runtime::{Runner as _, Supervisor as _, buffer::paged::CacheRef, deterministic};
use commonware_utils::{NZU16, NZU64, NZUsize, non_empty_range, sequence::FixedBytes};
use std::sync::Arc;

const CHUNK_BYTES: usize = 32;

type Key = FixedBytes<8>;
type Commitment<F> = CurrentCommitment<F, Key, FixedEncoding<Key>, Sha256, CHUNK_BYTES>;
type Batch = UserBatch<Key, Key>;
type TestDb<F> = Db<F, deterministic::Context, Key, Key, Sha256, OneCap, CHUNK_BYTES, Sequential>;

fn key(value: u64) -> Key {
    FixedBytes::new(value.to_be_bytes())
}

fn config(name: &str) -> FixedConfig<OneCap, Sequential> {
    let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
    FixedConfig {
        merkle_config: MerkleConfig {
            journal_partition: format!("{name}-nodes"),
            replay_buffer: NZUsize!(4096),
            metadata_partition: format!("{name}-merkle-metadata"),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            strategy: Sequential,
            page_cache: page_cache.clone(),
        },
        journal_config: JournalConfig {
            partition: format!("{name}-operations"),
            replay_buffer: NZUsize!(4096),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            page_cache,
        },
        grafted_metadata_partition: format!("{name}-grafted"),
        translator: OneCap,
        init_cache: Some(NZUsize!(1024)),
        init_buffer: NZUsize!(4096),
        init_concurrency: (),
    }
}

fn commitment<F: Graftable>(db: &TestDb<F>) -> Commitment<F> {
    Commitment::<F>::from_parts(db.bounds().end, db.root()).unwrap()
}

/// Round `round`'s batch over 400 keys: sets that update and create, a point delete, and a
/// range delete.
fn round_batch(round: u64) -> Batch {
    let sets = (0..40).map(|i| UserOp::set(key((round * 7 + i * 13) % 400), key(round * 1000 + i)));
    let delete = UserOp::delete(key((round * 31) % 400));
    let mut batch = UserBatch::from_iter(sets.chain(core::iter::once(delete)));
    let start = (round * 17) % 380;
    batch.delete_range(key(start)..key(start + 5)).unwrap();
    batch
}

async fn apply<F: Graftable>(db: TestDb<F>, batch: &Batch) -> TestDb<F> {
    let merkleized = batch.merkleize(&db, &mut Proportional).await.unwrap();
    db.apply_batch(merkleized).await.unwrap()
}

/// Record two consecutive transitions, move the live database well past them, then rebuild the
/// first checkpoint from the live database alone and prove both batches from that copy.
fn proves_transitions_at_an_old_checkpoint<F: Graftable>() {
    let executor = deterministic::Runner::default();
    executor.start(|context| async move {
        let mut db = TestDb::<F>::init(context.child("live"), config("live"), None)
            .await
            .unwrap();
        for round in 0..30 {
            db = apply(db, &round_batch(round)).await;
        }
        let (old_ops_root, old_boundary) = (sync::Database::root(&db), db.sync_boundary());
        let mut history = Vec::new();
        for round in 30..32 {
            let batch = round_batch(round);
            let pre = commitment(&db);
            db = apply(db, &batch).await;
            history.push((batch, pre, commitment(&db)));
        }
        let old_size = history[0].1.size();
        for round in 32..90 {
            db = apply(db, &round_batch(round)).await;
        }
        let db = db.commit().await.unwrap();
        assert!(
            db.inactivity_floor_loc() > old_size,
            "the floor passed the checkpoint"
        );

        // The checkpoint's commit, proved against the old size, gives its ops root and floor,
        // and the floor gives the lowest location a sync to that size may need.
        let last = old_size - 1;
        let (proof, ops) = db
            .ops_historical_proof(old_size, last, NZU64!(1))
            .await
            .unwrap();
        let encoded = ops.iter().map(|op| op.encode()).collect::<Vec<_>>();
        let ops_root = proof
            .reconstruct_root(&qmdb::hasher::<Sha256>(), &encoded, last)
            .unwrap();
        assert_eq!(ops_root, old_ops_root);
        let [Operation::CommitFloor(_, old_floor)] = ops.as_slice() else {
            panic!("a checkpoint ends with its commit");
        };
        let start =
            sync_boundary::<F, CHUNK_BYTES>(**old_floor / (8 * CHUNK_BYTES as u64), *old_size);
        assert_eq!(start, old_boundary);

        let live = Arc::new(db);
        let mut copy: TestDb<F> = sync::sync(engine::Config {
            context: context.child("copy"),
            db_config: config("copy"),
            fetch_batch_size: NZU64!(64),
            target: Target {
                root: ops_root,
                range: non_empty_range!(start, old_size),
            },
            source: live.clone(),
            apply_batch_size: NZU64!(1024),
            max_outstanding_requests: NZUsize!(4),
            update_rx: None,
            finish_rx: None,
            reached_target_tx: None,
        })
        .await
        .unwrap();

        // Prove each old batch from the copy, then apply it to reach the next checkpoint.
        for (batch, pre, post) in &history {
            assert_eq!(&commitment(&copy), pre);
            let (bundle, merkleized) = batch.prove(&copy, &mut Proportional).await.unwrap();
            let verified = batch
                .verify(
                    *pre,
                    TrustedMap::empty_cache(*pre),
                    &bundle,
                    *post,
                    &mut Proportional,
                )
                .unwrap();
            assert_eq!(verified.commitment(), post);
            copy = copy.apply_batch(merkleized).await.unwrap();
        }
        copy.destroy().await.unwrap();
        Arc::try_unwrap(live)
            .unwrap_or_else(|_| panic!("the live database is still shared"))
            .destroy()
            .await
            .unwrap();
    });
}

#[test]
fn mmr_proves_transitions_at_an_old_checkpoint() {
    proves_transitions_at_an_old_checkpoint::<mmr::Family>();
}

#[test]
fn mmb_proves_transitions_at_an_old_checkpoint() {
    proves_transitions_at_an_old_checkpoint::<mmb::Family>();
}
