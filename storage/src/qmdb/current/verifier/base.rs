//! The state a proof starts from: a database's applied state, or the tip of a chain of
//! pending batches built on it.
//!
//! A pending batch is merkleized but not applied. Reads through a chain consult the chain's
//! changes newest first and then the database, as the database's own batch reads do (see
//! [`crate::qmdb::chain`]). Without a tip, every read goes to the database directly.

use super::{
    BitmapProof, BuildError, CurrentCommitment, PreStateOpeningClaim, PreStateTerminalClaim,
    bitmap::build_bitmap_proof, ordered::PreparedOrderedBatch,
};
use crate::{
    Context,
    index::Ordered as OrderedIndex,
    journal::contiguous::{Contiguous as _, Mutable},
    merkle::{self, Family, Graftable, Location, Position, Readable, storage::Storage},
    qmdb::{
        self,
        any::{
            ValueEncoding,
            batch::{DiffEntry, lookup_sorted},
            operation::{Operation, update},
        },
        chain,
        current::{
            batch::{MerkleizedBatch, UnmerkleizedBatch},
            db::compute_ops_root_witness,
            grafting, ordered,
        },
        operation::Key,
    },
};
use commonware_codec::Codec;
use commonware_cryptography::Hasher;
use commonware_parallel::Strategy;
use commonware_utils::{NZU64, futures::try_join_all, range::contains_cyclic};
use core::ops::{Bound, RangeBounds as _};
use futures::{StreamExt as _, pin_mut};
use std::{collections::BTreeSet, sync::Arc};

type OrderedUpdate<K, V> = update::Ordered<K, V>;
type ExactRecord<F, K, V> = (Location<F>, OrderedUpdate<K, V>);
type Db<F, E, C, K, V, I, H, const N: usize, S> = ordered::db::Db<F, E, C, K, V, I, H, N, S>;
type Tip<F, K, V, H, const N: usize, S> =
    MerkleizedBatch<F, <H as Hasher>::Digest, OrderedUpdate<K, V>, N, S>;
type Pending<F, K, V, H, const N: usize, S> = Arc<Tip<F, K, V, H, N, S>>;
type Opening<F, V, H, const N: usize> = PreStateOpeningClaim<F, V, <H as Hasher>::Digest, N>;

/// The state a proof starts from.
pub(super) struct Base<'a, F, E, C, K, V, I, H, const N: usize, S>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
    /// The database: the whole state without a tip, and the state below the chain with one.
    pub(super) db: &'a Db<F, E, C, K, V, I, H, N, S>,
    /// The chain's newest pending batch, if the state is a chain's tip.
    tip: Option<&'a Pending<F, K, V, H, N, S>>,
}

impl<F, E, C, K, V, I, H, const N: usize, S> Clone for Base<'_, F, E, C, K, V, I, H, N, S>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, E, C, K, V, I, H, const N: usize, S> Copy for Base<'_, F, E, C, K, V, I, H, N, S>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
}

impl<'a, F, E, C, K, V, I, H, const N: usize, S> Base<'a, F, E, C, K, V, I, H, N, S>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
    /// The database's applied state.
    pub(super) const fn db(db: &'a Db<F, E, C, K, V, I, H, N, S>) -> Self {
        Self { db, tip: None }
    }

    /// The state after `tip`, a pending batch `db` created. Returns
    /// [`qmdb::Error::StaleRead`] unless `db` is at a state of `tip`'s chain, and
    /// [`qmdb::Error::StaleBatch`] if an unapplied batch in the chain was dropped.
    ///
    /// The second check is the one merkleizing a child of `tip` makes. Merkle node reads
    /// follow the chain's weak parent links, and past a dropped batch they would find the
    /// database's older nodes.
    pub(super) fn chain(
        db: &'a Db<F, E, C, K, V, I, H, N, S>,
        tip: &'a Pending<F, K, V, H, N, S>,
    ) -> Result<Self, BuildError<F>> {
        let current = db.any.commitment();
        tip.inner
            .bounds
            .compatible(&db.any, current)
            .map_err(BuildError::Database)?;
        let held = chain::parent_and_ancestors(Some(&tip.inner), |batch| batch.ancestors())
            .collect::<Vec<_>>();
        let boundary = chain::effective_boundary(
            tip.inner.bounds.db,
            held.last().map(|oldest| oldest.bounds.base),
        );
        chain::merkleizable(
            &db.any,
            current,
            boundary,
            held.iter().map(|batch| batch.commitment()),
        )
        .map_err(BuildError::Database)?;
        Ok(Self { db, tip: Some(tip) })
    }

    /// The commitment to this state.
    pub(super) fn commitment(&self) -> Result<CurrentCommitment<F, K, V, H, N>, BuildError<F>> {
        Ok(match self.tip {
            None => CurrentCommitment::from_parts(self.db.bounds().end, self.db.root())?,
            Some(tip) => CurrentCommitment::from_parts(tip.bounds().tip.size, tip.root())?,
        })
    }

    /// A writer for the batch after this state.
    pub(super) fn new_batch(&self) -> UnmerkleizedBatch<F, H, OrderedUpdate<K, V>, N, S> {
        self.tip
            .map_or_else(|| self.db.new_batch(), |tip| tip.new_batch::<H>())
    }

    /// The chain's changes, newest first, that a read consults before the database: none
    /// without a tip.
    fn diffs(self) -> impl Iterator<Item = &'a [(K, DiffEntry<F, V::Value>)]> {
        let db_size = self.db.bounds().end;
        self.tip
            .into_iter()
            .flat_map(move |tip| tip.inner.read_diffs(db_size))
    }

    /// The operation at `location` if the chain appended it.
    fn appended(&self, location: Location<F>) -> Option<&'a Operation<F, OrderedUpdate<K, V>>> {
        let tip = self.tip?;
        let (base, items) = tip.inner.chunks().find(|(base, _)| *location >= *base)?;
        items.get(usize::try_from(*location - base).ok()?)
    }

    /// The operation at `location` in this state's log.
    pub(super) async fn read(
        &self,
        location: Location<F>,
    ) -> Result<Operation<F, OrderedUpdate<K, V>>, BuildError<F>> {
        if let Some(operation) = self.appended(location) {
            return Ok(operation.clone());
        }
        Ok(self
            .db
            .any
            .log
            .read(location.as_u64())
            .await
            .map_err(qmdb::Error::Journal)?)
    }

    /// The operations at `locations`, which must be strictly increasing, in this state's log.
    ///
    /// The database's come from one batched read, which serves cached ones without I/O. The
    /// chain's operations follow the database's in the log, and come from memory.
    pub(super) async fn read_many(
        &self,
        locations: &[Location<F>],
    ) -> Result<Vec<Operation<F, OrderedUpdate<K, V>>>, BuildError<F>> {
        let stored = locations
            .iter()
            .take_while(|location| self.appended(**location).is_none())
            .map(|location| location.as_u64())
            .collect::<Vec<_>>();
        let mut operations = self
            .db
            .any
            .log
            .read_many(&stored)
            .await
            .map_err(qmdb::Error::Journal)?;
        for location in locations.iter().skip(stored.len()) {
            let Some(operation) = self.appended(*location) else {
                return Err(BuildError::InvalidState(
                    "a read location is outside the pending chain's operations",
                ));
            };
            operations.push(operation.clone());
        }
        Ok(operations)
    }

    /// The update the chain appended at `location`, the location of an active diff entry.
    fn appended_update(
        &self,
        location: Location<F>,
    ) -> Result<&'a OrderedUpdate<K, V>, BuildError<F>> {
        match self.appended(location) {
            Some(Operation::Update(update)) => Ok(update),
            _ => Err(BuildError::InvalidState(
                "a pending batch's active entry does not hold an update",
            )),
        }
    }

    /// `key`'s record in this state, with its location, or `None` if `key` is absent.
    pub(super) async fn member(
        &self,
        key: &K,
    ) -> Result<Option<ExactRecord<F, K, V>>, BuildError<F>> {
        // The newest change to `key` decides; the database holds keys the chain left alone.
        if let Some(entry) = self.diffs().find_map(|diff| lookup_sorted(diff, key)) {
            return match entry.loc() {
                Some(location) => Ok(Some((location, self.appended_update(location)?.clone()))),
                None => Ok(None),
            };
        }
        Ok(self
            .db
            .any
            .get_with_loc(key)
            .await?
            .map(|(update, location)| (location, update)))
    }

    /// The record whose cyclic span `[key, next_key)` holds `key`, or `None` if this state
    /// is empty.
    pub(super) async fn span_owner(
        &self,
        key: &K,
    ) -> Result<Option<ExactRecord<F, K, V>>, BuildError<F>> {
        if let Some(tip) = self.tip {
            if tip.inner.total_active_keys == 0 {
                return Ok(None);
            }
            if let Some(owner) = self.changed_owner::<true>(key)? {
                return Ok(Some(owner));
            }
        }
        Ok(self.db.any.get_span(key).await?)
    }

    /// The record whose successor is `key`, an active key.
    pub(super) async fn predecessor(&self, key: &K) -> Result<ExactRecord<F, K, V>, BuildError<F>> {
        if let Some(owner) = self.changed_owner::<false>(key)? {
            if owner.1.next_key != *key {
                return Err(BuildError::InvalidState(
                    "a pending batch's span owner does not link to an active key",
                ));
            }
            return Ok(owner);
        }
        predecessor_record(self.db, key).await
    }

    /// The record a chain change emitted whose cyclic span holds `key`, if one does: with
    /// `AT`, the span from its key up to its successor; otherwise from just past its key
    /// through its successor.
    ///
    /// Membership changes emit the affected predecessors and the created keys, so the newest
    /// layer with such a record owns the span in this state. In each layer only the nearest
    /// active entry at or below `key` (wrapping to its last) can own it: a later active key
    /// would end an earlier one's span first.
    fn changed_owner<const AT: bool>(
        &self,
        key: &K,
    ) -> Result<Option<ExactRecord<F, K, V>>, BuildError<F>> {
        for diff in self.diffs() {
            let end = diff.partition_point(|(candidate, _)| {
                if AT {
                    candidate <= key
                } else {
                    candidate < key
                }
            });
            let (below, above) = diff.split_at(end);
            let Some(location) = below
                .iter()
                .rev()
                .chain(above.iter().rev())
                .find_map(|(_, entry)| entry.loc())
            else {
                continue;
            };
            let update = self.appended_update(location)?;
            let owns = if AT {
                contains_cyclic(
                    (
                        Bound::Included(&update.key),
                        Bound::Excluded(&update.next_key),
                    ),
                    key,
                )
            } else {
                contains_cyclic(
                    (
                        Bound::Excluded(&update.key),
                        Bound::Included(&update.next_key),
                    ),
                    key,
                )
            };
            if owns {
                return Ok(Some((location, update.clone())));
            }
        }
        Ok(None)
    }

    /// The active keys in `range` in this state, ascending.
    ///
    /// The read follows this state's successor links from the range's first active key, one
    /// key lookup per active key in the range. Under a chain, a scan of the database's keys
    /// would also read every key the chain deleted.
    pub(super) async fn keys_in(
        &self,
        range: (Bound<K>, Bound<K>),
    ) -> Result<Vec<K>, BuildError<F>> {
        let mut keys = Vec::new();
        let mut next = match &range.0 {
            Bound::Unbounded => self.first_key().await?,
            Bound::Included(start) | Bound::Excluded(start) => {
                self.span_owner(start).await?.map(|(_, owner)| {
                    if owner.key == *start && range.contains(start) {
                        keys.push(owner.key);
                    }
                    owner.next_key
                })
            }
        };
        // The links end the range past its end, or where they wrap around to the least key.
        while let Some(key) =
            next.take_if(|key| range.contains(key) && keys.last().is_none_or(|last| *key > *last))
        {
            let Some((_, update)) = self.member(&key).await? else {
                return Err(BuildError::InvalidState(
                    "an active record links to an absent key",
                ));
            };
            next = Some(update.next_key);
            keys.push(key);
        }
        Ok(keys)
    }

    /// The least active key in this state, if any.
    pub(super) async fn first_key(&self) -> Result<Option<K>, BuildError<F>> {
        let stream = self.db.keys(..);
        pin_mut!(stream);
        let first = stream.next().await.transpose()?;
        let Some(tip) = self.tip else {
            return Ok(first);
        };
        if tip.inner.total_active_keys == 0 {
            return Ok(None);
        }
        // The database's least key, if the chain kept it. If the chain deleted it, its span
        // owner is either the greatest key, which links to the least, or a lesser key the chain
        // created.
        let mut kept = None;
        if let Some(first) = first {
            let Some((_, owner)) = self.span_owner(&first).await? else {
                return Err(BuildError::InvalidState(
                    "a nonempty pending state has no span owner",
                ));
            };
            if owner.key == first {
                kept = Some(first);
            } else if owner.next_key <= owner.key {
                return Ok(Some(owner.next_key));
            }
        }
        let created = self.first_created();
        Ok(match (kept, created) {
            (Some(kept), Some(created)) => Some(kept.min(created.clone())),
            (kept, created) => kept.or_else(|| created.cloned()),
        })
    }

    /// The least key whose newest change in the chain leaves it active.
    fn first_created(&self) -> Option<&'a K> {
        let diffs = self.diffs().collect::<Vec<_>>();
        diffs
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(layer, diff)| {
                diff.iter()
                    .find(|(key, entry)| {
                        entry.loc().is_some()
                            && diffs
                                .iter()
                                .take(layer)
                                .all(|newer| lookup_sorted(newer, key).is_none())
                    })
                    .map(|(key, _)| key)
            })
            .min()
    }

    /// The opening of this state: its terminal commit's raw proof, the frontier before it,
    /// and the root witness. Also returns its inactivity floor.
    pub(super) async fn opening(
        &self,
        commitment: CurrentCommitment<F, K, V, H, N>,
    ) -> Result<(Opening<F, V, H, N>, Location<F>), BuildError<F>> {
        let Some(tip) = self.tip else {
            return build_opening(self.db, commitment).await;
        };
        let size = commitment.size();
        if size == Location::new(1) {
            return Ok((PreStateOpeningClaim::Initial, Location::new(0)));
        }
        let terminal_location = size - 1;
        let Operation::CommitFloor(metadata, floor) = self.read(terminal_location).await? else {
            return Err(BuildError::InvalidState(
                "the committed operation log does not end in CommitFloor",
            ));
        };
        let nodes = ChainNodes {
            chain: &tip.inner.journal_batch,
            below: &self.db.any.log.merkle,
        };
        let grafted = ChainGrafted {
            chain: &tip.grafted,
            below: &*self.db.grafted_tree,
        };
        let storage =
            grafting::Storage::<F, H, _, _>::new(&grafted, grafting::height::<N>(), &nodes);
        let raw_proof = merkle::verification::historical_range_proof(
            &self.db.any.log.hasher,
            &nodes,
            size,
            terminal_location..terminal_location + 1,
            F::inactive_peaks(size, floor),
        )
        .await?;
        let witness =
            compute_ops_root_witness::<F, H, _, _, N>(&tip.bitmap, &storage, size, floor).await?;
        let pins_before_terminal =
            try_join_all(F::nodes_to_pin(terminal_location).map(|position| {
                let nodes = &nodes;
                async move {
                    nodes
                        .get_node(position)
                        .await?
                        .ok_or(merkle::Error::ElementPruned(position))
                }
            }))
            .await?;
        Ok((
            PreStateOpeningClaim::Proven {
                ops_root: tip.ops_root(),
                witness,
                terminal: PreStateTerminalClaim {
                    metadata,
                    floor,
                    raw_proof,
                    pins_before_terminal,
                },
            },
            floor,
        ))
    }

    /// The set bits from `floor` up to `size`, except the terminal at `size - 1`, until
    /// `entries` of them are outside `superseded`.
    pub(super) fn candidate_locations(
        &self,
        floor: Location<F>,
        size: Location<F>,
        superseded: &BTreeSet<Location<F>>,
        entries: u64,
    ) -> Vec<Location<F>> {
        self.tip.map_or_else(
            || scan_bits(self.db.bitmap(), floor, size, superseded, entries),
            |tip| scan_bits(&tip.bitmap, floor, size, superseded, entries),
        )
    }

    /// One raw multi-proof of `records`, ascending, against this state's operations tree.
    pub(super) async fn records_proof(
        &self,
        inactive_peaks: usize,
        records: &[Location<F>],
    ) -> Result<merkle::Proof<F, H::Digest>, BuildError<F>> {
        Ok(match self.tip {
            None => {
                merkle::verification::multi_proof(&self.db.any.log.merkle, inactive_peaks, records)
                    .await?
            }
            Some(tip) => {
                let nodes = ChainNodes {
                    chain: &tip.inner.journal_batch,
                    below: &self.db.any.log.merkle,
                };
                merkle::verification::multi_proof(&nodes, inactive_peaks, records).await?
            }
        })
    }

    /// The bitmap proof for `prepared`, whose floor scan ended at `scan_end`, against this
    /// state of `size` operations and inactivity floor `floor`.
    pub(super) async fn bitmap_proof(
        &self,
        size: Location<F>,
        floor: Location<F>,
        scan_end: Location<F>,
        prepared: &PreparedOrderedBatch<F, K, V>,
        claimed: &BTreeSet<Location<F>>,
    ) -> Result<BitmapProof<F, H::Digest, N>, BuildError<F>> {
        let Some(tip) = self.tip else {
            return build_bitmap_proof::<F, K, V, H, _, _, _, N>(
                size,
                floor,
                scan_end,
                prepared,
                claimed,
                self.db.bitmap(),
                &self.db.any.log.merkle,
                &self.db.grafted_storage(),
            )
            .await;
        };
        let nodes = ChainNodes {
            chain: &tip.inner.journal_batch,
            below: &self.db.any.log.merkle,
        };
        let grafted = ChainGrafted {
            chain: &tip.grafted,
            below: &*self.db.grafted_tree,
        };
        let storage =
            grafting::Storage::<F, H, _, _>::new(&grafted, grafting::height::<N>(), &nodes);
        build_bitmap_proof::<F, K, V, H, _, _, _, N>(
            size,
            floor,
            scan_end,
            prepared,
            claimed,
            &tip.bitmap,
            &nodes,
            &storage,
        )
        .await
    }
}

/// Collect set bits for [`Base::candidate_locations`] from `bitmap`.
fn scan_bits<F: Family, B: commonware_utils::bitmap::Readable<N>, const N: usize>(
    bitmap: &B,
    floor: Location<F>,
    size: Location<F>,
    superseded: &BTreeSet<Location<F>>,
    entries: u64,
) -> Vec<Location<F>> {
    let terminal_location = size - 1;
    let mut locations = Vec::new();
    let mut live = 0u64;
    if entries == 0 {
        return locations;
    }
    for raw_location in bitmap
        .ones_iter_from(floor.as_u64())
        .take_while(|location| *location < size.as_u64())
    {
        let location = Location::new(raw_location);
        if location == terminal_location {
            continue;
        }
        locations.push(location);
        if superseded.contains(&location) {
            continue;
        }
        live += 1;
        if live == entries {
            break;
        }
    }
    locations
}

/// Merkle nodes a chain of pending batches appended or rewrote, over the database's own.
///
/// The database's nodes come from its full Merkle storage, which holds every node it has not
/// pruned. A batch's own proofs read only the database's in-memory nodes, which a later flush
/// may drop; a proof of old records needs nodes anywhere in the tree.
struct ChainNodes<'a, R, B> {
    chain: &'a R,
    below: &'a B,
}

impl<F, R, B> Storage<F> for ChainNodes<'_, R, B>
where
    F: Family,
    R: Readable<Family = F>,
    B: Storage<F, Digest = R::Digest>,
{
    type Digest = R::Digest;

    fn size(&self) -> Position<F> {
        self.chain.size()
    }

    async fn get_node(
        &self,
        position: Position<F>,
    ) -> Result<Option<Self::Digest>, merkle::Error<F>> {
        if let Some(node) = self.chain.get_node(position) {
            return Ok(Some(node));
        }
        self.below.get_node(position).await
    }

    async fn get_nodes(
        &self,
        positions: &[Position<F>],
    ) -> Result<Vec<Self::Digest>, merkle::Error<F>> {
        // Fetch what the chain lacks in one request, keeping the storage's batching.
        let chained = positions
            .iter()
            .map(|position| self.chain.get_node(*position))
            .collect::<Vec<_>>();
        let missing = positions
            .iter()
            .zip(&chained)
            .filter(|(_, node)| node.is_none())
            .map(|(position, _)| *position)
            .collect::<Vec<_>>();
        let mut fetched = self.below.get_nodes(&missing).await?.into_iter();
        chained
            .into_iter()
            .zip(positions)
            .map(|(node, position)| {
                node.or_else(|| fetched.next())
                    .ok_or(merkle::Error::ElementPruned(*position))
            })
            .collect()
    }
}

/// Grafted-tree nodes a chain of pending batches changed, over the database's grafted tree.
struct ChainGrafted<'a, R, M> {
    chain: &'a R,
    below: &'a M,
}

impl<R, M> Readable for ChainGrafted<'_, R, M>
where
    R: Readable,
    M: Readable<Family = R::Family, Digest = R::Digest>,
{
    type Family = R::Family;
    type Digest = R::Digest;

    fn size(&self) -> Position<Self::Family> {
        self.chain.size()
    }

    fn get_node(&self, position: Position<Self::Family>) -> Option<Self::Digest> {
        self.chain
            .get_node(position)
            .or_else(|| self.below.get_node(position))
    }
}

/// Locate the immediate cyclic predecessor needed when deleting a member.
/// The index groups keys by translated prefixes, so inspect both the member's bucket
/// and the previous bucket (wrapping at the minimum), then match the exact successor.
/// Its ordinary membership proof will authenticate this link for the recipient.
/// For a bucket of at most `l` locations, O(i + l * log(l + 2)) CPU, O(l)
/// temporary locations, and O(l) record reads; `i` is index-lookup work. Bounded
/// occupancy gives O(i) typical work, while all `m` active keys can share a bucket.
async fn predecessor_record<F, E, C, K, V, I, H, const N: usize, S>(
    db: &Db<F, E, C, K, V, I, H, N, S>,
    key: &K,
) -> Result<ExactRecord<F, K, V>, BuildError<F>>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
    let mut locations = db.any.snapshot.get(key).copied().collect::<Vec<_>>();
    if let Some((iter, _)) = db.any.snapshot.prev_translated_key(key) {
        locations.extend(iter.copied());
    }
    locations.sort();
    locations.dedup();
    for location in locations {
        let operation = db
            .any
            .log
            .read(*location)
            .await
            .map_err(qmdb::Error::Journal)?;
        let Operation::Update(update) = operation else {
            return Err(BuildError::InvalidState(
                "the active ordered index referenced a non-update operation",
            ));
        };
        if update.next_key == *key {
            return Ok((location, update));
        }
    }
    Err(BuildError::InvalidState(
        "the active ordered cycle has no predecessor for a member",
    ))
}

/// Obtain the terminal's raw inclusion proof and the frontier immediately before it.
/// The root witness binds the raw log to Current's bitmap-aware commitment. Together
/// these let the verifier authenticate the old floor and reconstruct the append tip.
async fn build_opening<F, E, C, K, V, I, H, const N: usize, S>(
    db: &Db<F, E, C, K, V, I, H, N, S>,
    commitment: CurrentCommitment<F, K, V, H, N>,
) -> Result<(Opening<F, V, H, N>, Location<F>), BuildError<F>>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
    if commitment.size() == Location::new(1) {
        return Ok((PreStateOpeningClaim::Initial, Location::new(0)));
    }
    let terminal_location = commitment.size() - 1;
    let (raw_proof, mut operations) = db
        .ops_historical_proof(commitment.size(), terminal_location, NZU64!(1))
        .await?;
    let Some(Operation::CommitFloor(metadata, floor)) = operations.pop() else {
        return Err(BuildError::InvalidState(
            "the committed operation log does not end in CommitFloor",
        ));
    };
    let witness = db.ops_root_witness().await?;
    let pins_before_terminal = db.pinned_nodes_at(terminal_location).await?;
    Ok((
        PreStateOpeningClaim::Proven {
            ops_root: db.ops_root(),
            witness,
            terminal: PreStateTerminalClaim {
                metadata,
                floor,
                raw_proof,
                pins_before_terminal,
            },
        },
        floor,
    ))
}
