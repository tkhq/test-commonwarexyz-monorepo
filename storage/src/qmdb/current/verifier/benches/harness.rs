//! Seeded honest workloads shared by the verifier fuzzer and benchmarks.
//!
//! The benchmarks declare it as a module and the fuzzer includes it with `#[path]`; it is not part
//! of the verifier module or its public API. The action list is retained beside the normalized batch so
//! callers can apply an independent model in arrival order and catch normalization mistakes.

use commonware_cryptography::Sha256;
use commonware_parallel::{Sequential, Strategy};
use commonware_runtime::{buffer::paged::CacheRef, deterministic};
use commonware_storage::{
    journal::contiguous::fixed::Config as JournalConfig,
    merkle::{Graftable, full::Config as MerkleConfig},
    qmdb::current::{FixedConfig, ordered::fixed::Db, verifier::UserBatch},
    translator::EightCap,
};
use commonware_utils::{NZU16, NZU64, NZUsize, TestRng, sequence::FixedBytes};
use rand::TryRng as _;
use std::{
    collections::BTreeMap,
    ops::{Bound, RangeBounds as _},
};

/// Bitmap chunk size used by the shared workload.
pub const CHUNK_BYTES: usize = 32;

/// Fixed-size key and value used by the fixture.
pub type Key = FixedBytes<8>;

/// Fixed-size key and value used by the fixture.
pub type Value = FixedBytes<8>;

/// Storage-backed ordered Current database used by both targets.
pub type ScenarioDb<F, E = deterministic::Context, S = Sequential> =
    Db<F, E, Key, Value, Sha256, EightCap, CHUNK_BYTES, S>;

/// A caller action before it is normalized into a [`UserBatch`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Action {
    /// Set one key to a value.
    Set { key: Key, value: Value },
    /// Delete one key.
    Delete { key: Key },
    /// Delete a key interval. The generator's are inclusive; a fuzz case may open either end.
    DeleteRange { start: Bound<Key>, end: Bound<Key> },
}

impl Action {
    /// Delete the inclusive interval from `start` to `end`.
    pub const fn delete_range(start: Key, end: Key) -> Self {
        Self::DeleteRange {
            start: Bound::Included(start),
            end: Bound::Included(end),
        }
    }
}

/// Original actions and the normalized batch they produce.
#[derive(Clone, Debug)]
pub struct LogicalBatch {
    /// Actions in the order selected by the seeded generator.
    pub actions: Vec<Action>,
    /// The same actions after eager last-write-wins normalization.
    pub batch: UserBatch<Key, Value>,
}

impl LogicalBatch {
    /// Apply the original actions to an independent model in arrival order.
    pub fn apply_model(&self, model: &mut BTreeMap<Key, Value>) {
        for action in &self.actions {
            match action {
                Action::Set { key, value } => {
                    model.insert(key.clone(), value.clone());
                }
                Action::Delete { key } => {
                    model.remove(key);
                }
                Action::DeleteRange { start, end } => {
                    model.retain(|key, _| !(start.as_ref(), end.as_ref()).contains(key));
                }
            }
        }
    }
}

/// Deterministic point and range workload generator.
///
/// The generator deliberately emits absent deletes, existing-key deletes, ranges, and occasional
/// `range -> set -> delete` sequences. The latter exercises both range splitting and rejoining;
/// the independent model must consume the original action sequence rather than the normalized
/// batch. The seed and key-space size fully determine the output.
pub struct SeededWorkload {
    rng: TestRng,
    keyspace: u64,
}

impl SeededWorkload {
    /// Create a generator for keys in `0..keyspace`.
    pub fn new(seed: u64, keyspace: u64) -> Self {
        assert!(keyspace > 0, "the workload key space must be non-empty");
        Self {
            rng: TestRng::new(seed),
            keyspace,
        }
    }

    /// Create the initial population with keys `0..count`.
    pub fn initial_batch(&self, count: u64) -> LogicalBatch {
        assert!(
            count <= self.keyspace,
            "initial population exceeds key space"
        );
        let actions = (0..count)
            .map(|index| Action::Set {
                key: key(index),
                value: value(index ^ 0x9e37_79b9),
            })
            .collect();
        logical_batch(actions)
    }

    /// Generate the next batch with at most `action_count` original actions.
    pub fn next_batch(&mut self, action_count: usize) -> LogicalBatch {
        let mut actions = Vec::with_capacity(action_count);
        let mut remaining = action_count;
        while remaining != 0 {
            // Keep a visible proportion of batches focused on range normalization. A set inside
            // the range is followed by a delete often enough to exercise both last-write cases.
            if remaining >= 2 && self.bounded(8) == 0 {
                let (start, end) = self.range();
                let middle = start + self.bounded(end - start + 1);
                actions.push(Action::delete_range(key(start), key(end)));
                actions.push(Action::Set {
                    key: key(middle),
                    value: value(self.next_u64()),
                });
                remaining -= 2;
                if remaining != 0 && self.bounded(2) == 0 {
                    actions.push(Action::Delete { key: key(middle) });
                    remaining -= 1;
                }
                continue;
            }

            let selected = self.bounded(100);
            if selected < 55 {
                actions.push(Action::Set {
                    key: self.random_key(),
                    value: value(self.next_u64()),
                });
            } else if selected < 75 {
                actions.push(Action::Delete {
                    key: self.random_key(),
                });
            } else {
                let (start, end) = self.range();
                actions.push(Action::delete_range(key(start), key(end)));
            }
            remaining -= 1;
        }
        let mut logical = logical_batch(actions);
        if self.bounded(4) == 0 {
            logical.batch.set_metadata(value(self.next_u64()));
        }
        logical
    }

    fn random_key(&mut self) -> Key {
        key(self.bounded(self.keyspace))
    }

    fn range(&mut self) -> (u64, u64) {
        let start = self.bounded(self.keyspace);
        let width = self.bounded(65).min(self.keyspace - 1 - start);
        (start, start + width)
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.try_next_u64().expect("TestRng is infallible")
    }

    fn bounded(&mut self, upper: u64) -> u64 {
        debug_assert!(upper > 0);
        self.next_u64() % upper
    }
}

/// Convert a logical integer into the fixture's ordered key encoding.
pub fn key(index: u64) -> Key {
    FixedBytes::from(index.to_be_bytes())
}

/// Convert a logical integer into the fixture's value encoding.
pub fn value(seed: u64) -> Value {
    FixedBytes::from(seed.to_be_bytes())
}

/// Open a fresh deterministic database using the normal benchmark translator.
#[allow(dead_code)]
pub async fn open_db<F: Graftable>(
    context: deterministic::Context,
    partition_prefix: &str,
) -> ScenarioDb<F> {
    open_db_with_strategy(context, partition_prefix, Sequential).await
}

/// Open the same database fixture with a caller-selected runtime and CPU strategy.
pub async fn open_db_with_strategy<
    F: Graftable,
    E: commonware_storage::Context + commonware_runtime::Spawner,
    S: Strategy,
>(
    context: E,
    partition_prefix: &str,
    strategy: S,
) -> ScenarioDb<F, E, S> {
    let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
    let config = FixedConfig {
        merkle_config: MerkleConfig {
            journal_partition: format!("{partition_prefix}-nodes"),
            replay_buffer: NZUsize!(4096),
            metadata_partition: format!("{partition_prefix}-node-metadata"),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            strategy,
            page_cache: page_cache.clone(),
        },
        journal_config: JournalConfig {
            partition: format!("{partition_prefix}-operations"),
            replay_buffer: NZUsize!(4096),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            page_cache,
        },
        grafted_metadata_partition: format!("{partition_prefix}-grafted"),
        translator: EightCap,
        init_cache: Some(NZUsize!(1024)),
        init_buffer: NZUsize!(4096),
        init_concurrency: (),
    };
    Db::<F, _, Key, Value, Sha256, _, CHUNK_BYTES, S>::init(context, config, None)
        .await
        .expect("fixture database initialization succeeds")
}

fn logical_batch(actions: Vec<Action>) -> LogicalBatch {
    let mut batch = UserBatch::new();
    for action in &actions {
        match action {
            Action::Set { key, value } => batch.set(key.clone(), value.clone()),
            Action::Delete { key } => batch.delete(key.clone()),
            Action::DeleteRange { start, end } => batch
                .delete_range((start.clone(), end.clone()))
                .expect("generated ranges are ordered"),
        }
    }
    LogicalBatch { actions, batch }
}
