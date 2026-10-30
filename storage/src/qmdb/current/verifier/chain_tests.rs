//! Prove batches on top of pending batches.
//!
//! A batch proved on a pending parent must give the bundle and batch that proving it from the
//! database gives once the parent is applied, whichever of its ancestors the database has applied
//! meanwhile, for fixed and variable values alike.

use super::{
    BatchProofBundle, BuildError, CurrentCommitment, TrustedMap, UserBatch, UserOp,
    tests::{Batch, Key, StopEvery, apply_model, batches, index, key, keyspace},
};
use crate::{
    index::Ordered as OrderedIndex,
    journal::contiguous::{
        Mutable, fixed::Config as JournalConfig, variable::Config as VariableJournalConfig,
    },
    merkle::{Graftable, Location, full::Config as MerkleConfig, mmb, mmr},
    qmdb::{
        self,
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        current::{FixedConfig, VariableConfig, batch::MerkleizedBatch, ordered},
        floor::{Policy, Proportional},
    },
    translator::OneCap,
};
use commonware_codec::{Codec, Encode, RangeCfg};
use commonware_cryptography::{Hasher, Sha256};
use commonware_parallel::{Sequential, Strategy, mocks::Explorer};
use commonware_runtime::{Runner as _, Strategizer as _, buffer::paged::CacheRef, deterministic};
use commonware_utils::{NZU16, NZU64, NZUsize};
use core::{fmt::Debug, ops::Bound};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

type Commitment<F, V> = CurrentCommitment<F, Key, V, Sha256, 32>;
type TestDb<F, S> = ordered::fixed::Db<F, deterministic::Context, Key, Key, Sha256, OneCap, 32, S>;
type VariableDb<F> =
    ordered::variable::Db<F, deterministic::Context, Key, Vec<u8>, Sha256, OneCap, 32, Sequential>;
/// An ordered database with values encoded by `V`.
type AnyDb<F, C, V, I, S> = ordered::db::Db<F, deterministic::Context, C, Key, V, I, Sha256, 32, S>;
type Pending<F, V, S> =
    Arc<MerkleizedBatch<F, <Sha256 as Hasher>::Digest, update::Ordered<Key, V>, 32, S>>;

fn config<S: Strategy>(name: &str, strategy: S) -> FixedConfig<OneCap, S> {
    let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
    FixedConfig {
        merkle_config: MerkleConfig {
            journal_partition: format!("{name}-nodes"),
            replay_buffer: NZUsize!(4096),
            metadata_partition: format!("{name}-merkle-metadata"),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            strategy,
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

/// The codec config of the variable-value database's operations.
type VariableCodec = ((), (RangeCfg<usize>, ()));

/// The variable-value database, with values up to 64 bytes.
fn variable_config(name: &str) -> VariableConfig<OneCap, VariableCodec, Sequential> {
    let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
    VariableConfig {
        merkle_config: MerkleConfig {
            journal_partition: format!("{name}-nodes"),
            replay_buffer: NZUsize!(4096),
            metadata_partition: format!("{name}-merkle-metadata"),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            strategy: Sequential,
            page_cache: page_cache.clone(),
        },
        journal_config: VariableJournalConfig {
            partition: format!("{name}-operations"),
            items_per_section: NZU64!(1024),
            compression: None,
            codec_config: ((), ((..=64).into(), ())),
            page_cache,
            write_buffer: NZUsize!(4096),
            replay_buffer: NZUsize!(4096),
        },
        grafted_metadata_partition: format!("{name}-grafted"),
        translator: OneCap,
        init_cache: Some(NZUsize!(1024)),
        init_buffer: NZUsize!(4096),
        init_concurrency: (),
    }
}

/// `batch` with each value, and its metadata, replaced by 1 to 7 bytes that depend on it.
fn with_variable_values(batch: &Batch) -> UserBatch<Key, Vec<u8>> {
    let bytes = |value: &Key| {
        let value = index(value);
        vec![value as u8; 1 + (value % 7) as usize]
    };
    let mut variable = batch
        .iter()
        .map(|op| match op {
            UserOp::Set { key, value } => UserOp::set(key.clone(), bytes(value)),
            UserOp::Delete { key } => UserOp::delete(key.clone()),
            UserOp::DeleteRange { start, end } => {
                UserOp::delete_range((start.clone(), end.clone()))
            }
        })
        .collect::<UserBatch<_, _>>();
    if let Some(metadata) = batch.metadata() {
        variable.set_metadata(bytes(metadata));
    }
    variable
}

fn commitment<F: Graftable, V: ValueEncoding, S: Strategy>(
    batch: &Pending<F, V, S>,
) -> Commitment<F, V> {
    Commitment::<F, V>::from_parts(batch.bounds().tip.size, batch.root()).unwrap()
}

/// Batches whose changes depend on a pending parent's: keys a parent created, deleted again
/// singly and by a range; a range that deletes nothing, after a key a parent changed; a key a
/// parent deleted, created again; an unbounded range whose least key a parent created; emptying
/// and refilling the state; and an unbounded range whose least key a parent created below the
/// database's, which it deleted.
fn chain_batches() -> Vec<Batch> {
    let mut fill = UserBatch::new();
    for i in 0..300 {
        fill.set(key(i), key(i));
    }
    let mut thin = UserBatch::new();
    thin.delete_range(key(100)..key(200)).unwrap();
    thin.set(key(150), key(1));
    thin.delete(key(5));
    let mut revive = UserBatch::new();
    revive.set(key(5), key(2));
    revive.delete_range(..key(3)).unwrap();
    let mut refill = UserBatch::new();
    for i in 10..20 {
        refill.set(key(i), key(i));
    }
    let mut shift = UserBatch::new();
    shift.delete(key(9));
    shift.set(key(5), key(6));
    vec![
        fill,
        thin,
        // Only the endpoint's own record, which the batch leaves alone, shows the range is empty.
        UserBatch::from([UserOp::delete_range((
            Bound::Excluded(key(150)),
            Bound::Included(key(160)),
        ))]),
        revive,
        UserBatch::from([UserOp::set(key(1), key(3))]),
        // The least key, which the parent created, starts this range.
        UserBatch::from([UserOp::delete_range(..key(2))]),
        UserBatch::from([UserOp::delete_range(..)]),
        UserBatch::from([UserOp::set(key(7), key(4))]),
        // The only key's successor wraps to itself, so a range below it deletes nothing.
        UserBatch::from([UserOp::delete_range(..key(7))]),
        UserBatch::from([UserOp::delete(key(7))]),
        UserBatch::new(),
        UserBatch::from([UserOp::set(key(9), key(5))]),
        refill,
        shift,
        // The database's least key, which the parent deleted, follows the least key, which the
        // parent created.
        UserBatch::from([UserOp::delete_range(..key(12))]),
    ]
}

/// A batch proved and not yet applied.
struct Proved<F: Graftable, V: ValueEncoding, S: Strategy, P> {
    batch: UserBatch<Key, V::Value>,
    /// The writer's policy before this batch.
    policy: P,
    /// The state the batch was proved from.
    pre: Commitment<F, V>,
    /// The bundle, encoded.
    bundle: Vec<u8>,
    pending: Pending<F, V, S>,
}

/// Prove `batches` in order on `db`, each on top of the batch before while it is pending,
/// applying the oldest whenever `depth` are pending, and then the rest.
///
/// Every bundle must verify from an empty cache against its parent's post-state, and when a
/// batch is applied, proving it from the database must give the same bundle and root. With
/// `prune`, the database syncs and prunes after each apply while later batches are pending.
async fn pipeline<F, C, V, I, S, P>(
    mut db: AnyDb<F, C, V, I, S>,
    policy: P,
    depth: usize,
    prune: bool,
    batches: Vec<UserBatch<Key, V::Value>>,
) where
    F: Graftable,
    C: Mutable<Item = Operation<F, update::Ordered<Key, V>>>,
    V: ValueEncoding,
    V::Value: Clone + Debug + PartialEq + Send + Sync,
    I: OrderedIndex<Value = Location<F>> + 'static,
    S: Strategy,
    P: Policy<F, Key, V::Value> + Clone + Debug + PartialEq + Send,
    Operation<F, update::Ordered<Key, V>>: Codec,
    BatchProofBundle<F, Key, V, Sha256, 32>: Encode,
{
    let keys = keyspace(&batches);
    let mut writer = policy.clone();
    let mut verifier = policy;
    let mut expected = BTreeMap::new();
    let mut queue = VecDeque::<Proved<F, V, S, P>>::new();
    for batch in batches {
        if queue.len() == depth {
            let oldest = queue.pop_front().unwrap();
            db = apply(db, oldest, &mut expected, prune).await;
        }
        let before = writer.clone();
        let (pre, (bundle, pending)) = match queue.back() {
            Some(parent) => {
                let proved = batch
                    .prove_on(&parent.pending, &db, &mut writer)
                    .await
                    .unwrap();
                let built = batch
                    .build_batch_proof_on(&parent.pending, &db, &before)
                    .await
                    .unwrap();
                assert_eq!(built.encode(), proved.0.encode());
                (commitment(&parent.pending), proved)
            }
            None => {
                let pre = Commitment::<F, V>::from_parts(db.bounds().end, db.root()).unwrap();
                (pre, batch.prove(&db, &mut writer).await.unwrap())
            }
        };
        let post = commitment(&pending);
        let verified = batch
            .verify(
                pre,
                TrustedMap::empty_cache(pre),
                &bundle,
                post,
                &mut verifier,
            )
            .unwrap();
        assert_eq!(verified.commitment(), &post);
        assert_eq!(writer, verifier);
        queue.push_back(Proved {
            batch,
            policy: before,
            pre,
            bundle: bundle.encode().to_vec(),
            pending,
        });
    }
    while let Some(oldest) = queue.pop_front() {
        db = apply(db, oldest, &mut expected, prune).await;
    }
    for i in 0..keys {
        assert_eq!(
            db.get(&key(i)).await.unwrap().as_ref(),
            expected.get(&key(i))
        );
    }
    db.destroy().await.unwrap();
}

/// Check that proving `proved`'s batch from `db`, now at its pre-state, gives its bundle and
/// root, then apply its pending batch.
async fn apply<F, C, V, I, S, P>(
    db: AnyDb<F, C, V, I, S>,
    proved: Proved<F, V, S, P>,
    expected: &mut BTreeMap<Key, V::Value>,
    prune: bool,
) -> AnyDb<F, C, V, I, S>
where
    F: Graftable,
    C: Mutable<Item = Operation<F, update::Ordered<Key, V>>>,
    V: ValueEncoding,
    V::Value: Clone + Send + Sync,
    I: OrderedIndex<Value = Location<F>> + 'static,
    S: Strategy,
    P: Policy<F, Key, V::Value> + Clone + Send,
    Operation<F, update::Ordered<Key, V>>: Codec,
    BatchProofBundle<F, Key, V, Sha256, 32>: Encode,
{
    let Proved {
        batch,
        mut policy,
        pre,
        bundle,
        pending,
    } = proved;
    assert_eq!(
        Commitment::<F, V>::from_parts(db.bounds().end, db.root()).unwrap(),
        pre
    );
    let (direct, merkleized) = batch.prove(&db, &mut policy).await.unwrap();
    assert_eq!(
        direct.encode().to_vec(),
        bundle,
        "the parent changed the bundle"
    );
    assert_eq!(merkleized.root(), pending.root());
    drop(merkleized);
    let mut db = db.apply_batch(pending).await.unwrap();
    apply_model(expected, &batch);
    if prune {
        db = db.sync().await.unwrap();
        let boundary = db.sync_boundary();
        db = db.prune(boundary).await.unwrap();
    }
    db
}

/// Every chain depth, over both batch lists.
fn chains_match_applied<F: Graftable>() {
    for depth in [2, 3, 8] {
        for list in [chain_batches(), batches()] {
            deterministic::Runner::default().start(|context| async move {
                let db = TestDb::<F, _>::init(context, config("chain", Sequential), None)
                    .await
                    .unwrap();
                pipeline(db, Proportional, depth, false, list).await;
            });
        }
    }
}

#[test]
fn mmr_chains_match_applied() {
    chains_match_applied::<mmr::Family>();
}

#[test]
fn mmb_chains_match_applied() {
    chains_match_applied::<mmb::Family>();
}

/// A policy that keeps state, a parallel strategy, and pruning while batches are pending.
fn chains_match_applied_while_pruning<F: Graftable>() {
    for list in [chain_batches(), batches()] {
        deterministic::Runner::default().start(|context| async move {
            let strategy = Explorer::parallel(&context.strategy(NZUsize!(4)));
            let db = TestDb::<F, _>::init(context, config("chain", strategy), None)
                .await
                .unwrap();
            pipeline(db, StopEvery { period: 3, seen: 0 }, 3, true, list).await;
        });
    }
}

#[test]
fn mmr_chains_match_applied_while_pruning() {
    chains_match_applied_while_pruning::<mmr::Family>();
}

#[test]
fn mmb_chains_match_applied_while_pruning() {
    chains_match_applied_while_pruning::<mmb::Family>();
}

/// Values of varying length, short and deep chains, and pruning while batches are pending.
fn variable_chains_match_applied<F: Graftable>() {
    for (depth, prune) in [(2, false), (3, true), (8, false)] {
        for list in [chain_batches(), batches()] {
            let list = list.iter().map(with_variable_values).collect::<Vec<_>>();
            deterministic::Runner::default().start(|context| async move {
                let db = VariableDb::<F>::init(context, variable_config("chain-variable"), None)
                    .await
                    .unwrap();
                pipeline(db, Proportional, depth, prune, list).await;
            });
        }
    }
}

#[test]
fn mmr_variable_chains_match_applied() {
    variable_chains_match_applied::<mmr::Family>();
}

#[test]
fn mmb_variable_chains_match_applied() {
    variable_chains_match_applied::<mmb::Family>();
}

/// A chain stops being provable once the database leaves it, or once an unapplied ancestor is
/// dropped.
async fn stale<F: Graftable>(context: deterministic::Context) {
    let mut db = TestDb::<F, Sequential>::init(context, config("stale", Sequential), None)
        .await
        .unwrap();
    let mut fill = UserBatch::new();
    for i in 0..50 {
        fill.set(key(i), key(i));
    }
    let (_, filled) = fill.prove(&db, &mut Proportional).await.unwrap();
    db = db.apply_batch(filled).await.unwrap();
    let first = UserBatch::from([UserOp::set(key(1), key(10))]);
    let second = UserBatch::from([UserOp::set(key(2), key(20))]);
    let next = UserBatch::from([UserOp::set(key(4), key(40))]);

    let (_, parent) = first.prove(&db, &mut Proportional).await.unwrap();
    let (_, child) = second
        .prove_on(&parent, &db, &mut Proportional)
        .await
        .unwrap();
    // A sibling of the parent applied leaves the whole chain stale.
    let sibling = UserBatch::from([UserOp::set(key(3), key(30))]);
    let (_, sibling) = sibling.prove(&db, &mut Proportional).await.unwrap();
    db = db.apply_batch(sibling).await.unwrap();
    for tip in [&parent, &child] {
        assert!(matches!(
            next.prove_on(tip, &db, &mut Proportional).await,
            Err(BuildError::Database(qmdb::Error::StaleRead))
        ));
    }

    // A dropped unapplied ancestor leaves its descendants unprovable.
    let (_, parent) = first.prove(&db, &mut Proportional).await.unwrap();
    let (_, child) = second
        .prove_on(&parent, &db, &mut Proportional)
        .await
        .unwrap();
    drop(parent);
    assert!(matches!(
        next.prove_on(&child, &db, &mut Proportional).await,
        Err(BuildError::Database(qmdb::Error::StaleBatch))
    ));
    db.destroy().await.unwrap();
}

#[test]
fn mmr_stale_chains_are_refused() {
    deterministic::Runner::default().start(stale::<mmr::Family>);
}

#[test]
fn mmb_stale_chains_are_refused() {
    deterministic::Runner::default().start(stale::<mmb::Family>);
}
