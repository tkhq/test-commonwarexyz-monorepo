//! Turn authenticated cyclic adjacency into the canonical physical operation suffix.
//!
//! An ordered update stores its key, value, and immediate cyclic successor. Inserting
//! or deleting a key may therefore require rewriting a surviving predecessor even
//! when the caller did not change its value. The proof bundle supplies pre-state
//! records establishing membership or absence. This module authenticates those facts,
//! then combines them with the trusted cache and the caller's `UserBatch` mutations
//! to construct the new updates, deletes, and predecessor rewrites in canonical order.
//! Floor copies and the terminal commit are derived in the floor stage that follows.

use super::{
    CandidateOperation, InsufficientPreState,
    OptionTrusted::{KnownTrusted, Unknown},
    PreStateKeyClaim, TrustedEntry, TrustedMap, TrustedState, UserOp, VerifyError,
    bitmap::VerifiedBitmapView,
    opening::AuthenticatedPreState,
    trusted::strict_span_contains,
    user::{ByStart, range_contains},
};
use crate::{
    merkle::{self, Graftable, Location},
    qmdb::{
        self,
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        operation::Key,
    },
};
use alloc::{
    collections::{BTreeMap, BTreeSet, btree_map},
    vec::Vec,
};
use commonware_codec::{Codec, Encode as _};
use commonware_cryptography::{Digest, Hasher};
use core::ops::Bound;

/// Each mutated key with the value a set writes, or `None` for a delete, after range deletes
/// are expanded into the members they cover.
pub(crate) type PointMutations<'a, K, V> = BTreeMap<K, Option<&'a V>>;

/// Install a fully authenticated active update without contradicting retained facts.
///
/// Member and exclusion proofs can reveal fields omitted from the caller's cache.
/// Merge those fields so operation derivation can use the record's location, value,
/// and successor. Compare every previously known field before replacing the entry.
/// Rejecting conflicts prevents execution from combining incompatible versions of a
/// record. Cache insertion also checks adjacency consistency.
///
/// This helper checks consistency only. Its caller must first authenticate the
/// update and location against the cache's pre-state commitment.
fn merge_member<F, K, V, H, const N: usize>(
    map: &mut TrustedMap<F, K, V, H, N>,
    location: Location<F>,
    update: update::Ordered<K, V>,
) -> Result<(), VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    if let Some(existing) = map.entry(&update.key) {
        let conflict = matches!(existing.location, KnownTrusted(old) if old != location)
            || matches!(&existing.next_key, KnownTrusted(old) if old != &update.next_key)
            || matches!(&existing.value, KnownTrusted(old) if old.encode() != update.value.encode());
        if conflict {
            return Err(VerifyError::ConflictingPreState);
        }
    }
    map.replace_entry(TrustedEntry {
        key: update.key,
        location: KnownTrusted(location),
        value: KnownTrusted(update.value),
        next_key: KnownTrusted(update.next_key),
    })
    .map_err(|_| VerifyError::ConflictingPreState)
}

/// A claim's location and record, and whether it is an exclusion.
type ClaimedRecord<F, K, V> = (Location<F>, update::Ordered<K, V>, bool);

/// The record a claim supplies. An exclusion's record must span its queried key.
fn claim_record<F, K, V>(
    claim: &PreStateKeyClaim<F, K, V>,
) -> Result<ClaimedRecord<F, K, V>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
{
    match claim {
        PreStateKeyClaim::Member(claim) => Ok((
            claim.location,
            update::Ordered {
                key: claim.key.clone(),
                value: claim.value.clone(),
                next_key: claim.next_key.clone(),
            },
            false,
        )),
        PreStateKeyClaim::Exclusion(claim) => {
            if !strict_span_contains(&claim.owner.key, &claim.owner.next_key, &claim.queried_key) {
                return Err(VerifyError::InvalidExclusionProof);
            }
            Ok((claim.location, claim.owner.clone(), true))
        }
    }
}

/// Authenticate the payload of every pre-state record the bundle supplies, the key claims'
/// records and the floor candidates, with the bundle's one shared raw multi-proof, and merge
/// the claims into the cache.
///
/// Inclusion alone does not make a record current: a superseded record stays in the log.
/// The caller checks each claim's activity bit against the verified bitmap, with
/// [`check_claim_activity`], before accepting the transition, and the candidates'
/// completeness with [`super::floor::verify_candidates`]. A location that two claims, or a
/// claim and a candidate, share must carry one payload. Return the claims' distinct
/// locations, whose chunks the bitmap proof must cover.
///
/// Claims merge in their original order. An error may follow successful earlier merges, so
/// the caller must discard the working cache on failure.
pub(crate) fn authenticate_and_merge_records<F, K, V, H, const N: usize>(
    pre: &AuthenticatedPreState<F, K, V, H, N>,
    map: &mut TrustedMap<F, K, V, H, N>,
    claims: &[PreStateKeyClaim<F, K, V>],
    candidates: &[CandidateOperation<F, K, V>],
    proof: Option<&merkle::Proof<F, H::Digest>>,
) -> Result<BTreeSet<Location<F>>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    let mut records = BTreeMap::new();
    let mut claimed = Vec::with_capacity(claims.len());
    for claim in claims {
        let (location, update, exclusion) = claim_record(claim)?;
        if location < pre.old_floor || location >= pre.terminal_location {
            return Err(if exclusion {
                VerifyError::InvalidExclusionProof
            } else {
                VerifyError::InvalidMemberProof
            });
        }
        insert_record(&mut records, location, Operation::Update(update.clone()))?;
        claimed.push((location, update, exclusion));
    }
    for (location, operation) in candidates {
        insert_record(&mut records, *location, operation.clone())?;
    }
    let operations = records.into_iter().collect::<Vec<_>>();
    let authentic = proof.map_or(operations.is_empty(), |raw| {
        !operations.is_empty()
            && raw.leaves == pre.commitment.size()
            && raw.inactive_peaks == F::inactive_peaks(pre.commitment.size(), pre.old_floor)
            && qmdb::verify_multi_proof::<H, F, _>(raw, &operations, &pre.ops_root)
    });
    if !authentic {
        return Err(VerifyError::InvalidRecordProof);
    }
    let mut locations = BTreeSet::new();
    for (location, update, exclusion) in claimed {
        locations.insert(location);
        let edge = exclusion.then(|| (update.key.clone(), update.next_key.clone()));
        merge_member(map, location, update)?;
        if let Some((key, next_key)) = edge {
            map.install_span(key, next_key)
                .map_err(|_| VerifyError::ConflictingPreState)?;
        }
    }
    Ok(locations)
}

/// Add one supplied record to the set the shared proof covers; a location supplied twice
/// must carry the same payload.
fn insert_record<F, K, V>(
    records: &mut BTreeMap<Location<F>, Operation<F, update::Ordered<K, V>>>,
    location: Location<F>,
    operation: Operation<F, update::Ordered<K, V>>,
) -> Result<(), VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    match records.entry(location) {
        btree_map::Entry::Vacant(entry) => {
            entry.insert(operation);
            Ok(())
        }
        btree_map::Entry::Occupied(entry) if entry.get().encode() == operation.encode() => Ok(()),
        btree_map::Entry::Occupied(_) => Err(VerifyError::ConflictingPreState),
    }
}

/// Require every claimed record to be active in the pre-state, as the verified bitmap shows.
/// The bitmap proof covers each claim's chunk, so an uncovered bit is a defect of the proof.
pub(crate) fn check_claim_activity<F, D, const N: usize>(
    bitmap: &VerifiedBitmapView<F, D, N>,
    claimed: &BTreeSet<Location<F>>,
) -> Result<(), VerifyError>
where
    F: Graftable,
    D: Digest,
{
    if claimed.iter().all(|location| bitmap.bit(*location) == Some(true)) {
        Ok(())
    } else {
        Err(VerifyError::InactiveClaim)
    }
}

/// Cached fields needed to invalidate a record and derive its replacement or deletion.
/// The location identifies the activity bit to clear. The successor identifies the
/// next old key when repairing links. The value is optional because only rewrites
/// that preserve a key's old value need it.
struct ExactEntry<'a, F: Graftable, K: Key, V> {
    location: Location<F>,
    value: Option<&'a V>,
    next_key: &'a K,
}

/// Canonical application operations derived from authorized mutations and trusted facts.
///
/// These results connect logical execution to floor replay and root reconstruction.
/// `application_ops` contains the operations to append before floor copies and the
/// terminal, with at most one operation per key. Every appended update is live until
/// floor replay copies it. `superseded` identifies the old activity bits to clear.
///
/// `deleted_keys` identifies cache entries to remove. `post_empty` requires a trusted
/// empty database or authenticated successor links forming a cycle of deleted keys.
/// Floor replay hands the floor policy [`Self::made_inactive`], as the writer does.
pub(crate) struct PreparedOrderedBatch<F: Graftable, K: Key, V: ValueEncoding> {
    pub(crate) application_ops: Vec<Operation<F, update::Ordered<K, V>>>,
    pub(crate) superseded: BTreeSet<Location<F>>,
    pub(crate) deleted_keys: BTreeSet<K>,
    pub(crate) post_empty: bool,
}

impl<F: Graftable, K: Key, V: ValueEncoding> PreparedOrderedBatch<F, K, V> {
    /// The operations this batch makes inactive, counted as the writer counts them for its floor
    /// policy: one per superseded record, predecessor rewrites included, and one more per delete,
    /// which is itself inactive.
    pub(crate) fn made_inactive(&self) -> usize {
        self.superseded.len() + self.deleted_keys.len()
    }
}

/// Gather the cached fields needed to replace or delete an existing record.
///
/// Operation derivation needs its location to clear the old activity bit and count
/// a superseded record among the operations the batch made inactive, which the floor
/// policy turns into its entry limit. It needs the old successor to
/// derive links that bypass deleted keys or include newly created keys. Require both
/// fields before execution. A retained absence span can supply the successor even
/// when the entry omits it, because that span records an immediate successor link.
///
/// Borrow the optional value: a set supplies a new value, and a delete needs no value.
/// A predecessor rewrite must require the old value so changing its link preserves
/// its contents. This helper reads already trusted facts and checks that the needed
/// fields are available. It does not authenticate new evidence.
fn exact_entry<'a, F, K, V, H, const N: usize>(
    map: &'a TrustedMap<F, K, V, H, N>,
    key: &K,
) -> Result<ExactEntry<'a, F, K, V::Value>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    let entry = map.entry(key).ok_or(InsufficientPreState::Location)?;
    let KnownTrusted(location) = entry.location else {
        return Err(VerifyError::InsufficientPreState(
            InsufficientPreState::Location,
        ));
    };
    let next_key = match &entry.next_key {
        KnownTrusted(next_key) => Some(next_key),
        Unknown => None,
    }
    .or_else(|| map.partial().and_then(|partial| partial.known_edge(key)))
    .ok_or(InsufficientPreState::NextKey)?;
    let value = match &entry.value {
        KnownTrusted(value) => Some(value),
        Unknown => None,
    };
    Ok(ExactEntry {
        location,
        value,
        next_key,
    })
}

/// Return the least key strictly greater than `key`, wrapping to the first key.
/// Operation derivation uses this to find the next newly created key in cyclic order.
/// It then compares that key with the surviving old successor to choose the final
/// link. In an initially empty database, new keys alone define the cycle.
/// An empty set returns `None`. A singleton returns its only key.
fn cyclic_next<'a, K: Key, T>(keys: &'a BTreeMap<K, T>, key: &K) -> Option<&'a K> {
    keys.range((Bound::Excluded(key), Bound::Unbounded))
        .next()
        .or_else(|| keys.first_key_value())
        .map(|(key, _)| key)
}

/// Find the first old neighbor that survives a run of adjacent deletions.
///
/// Deleting B and C from A -> B -> C -> D requires a new A -> D link. Walking
/// successors finds D as the new link target. Walking predecessors finds A as the
/// record that needs rewriting. Both searches must follow every authenticated link
/// across the deleted run: choosing the next cached key could skip a live key whose
/// record is absent from the cache.
///
/// Starting at `anchor`, use `next` to follow immediate old successors or predecessors.
/// Return the first neighbor outside `deleted`. Return `None` if the walk closes a
/// cycle of deleted keys, which establishes that no old key survives. A missing link
/// returns an error because the search cannot establish the surviving neighbor.
/// Cache each visited anchor's result to avoid repeating the same deletion-chain walk.
/// The lookup and memoized results must use the same pre-state, deletion set, and direction.
fn surviving_neighbor<K: Key, T>(
    anchor: &K,
    deleted: &BTreeMap<K, T>,
    memoized: &mut BTreeMap<K, Option<K>>,
    next: impl Fn(&K) -> Option<K>,
) -> Result<Option<K>, VerifyError> {
    let mut current = anchor.clone();
    let mut visited = BTreeSet::new();
    let result = loop {
        if let Some(result) = memoized.get(&current) {
            break result.clone();
        }
        if !visited.insert(current.clone()) {
            break None;
        }
        let neighbor = next(&current).ok_or(InsufficientPreState::OrderedEdge)?;
        if !deleted.contains_key(&neighbor) {
            break Some(neighbor);
        }
        current = neighbor;
    };
    for key in visited {
        memoized.insert(key, result.clone());
    }
    Ok(result)
}

/// Expand normalized delete intervals into the exact pre-state members they cover, and
/// return them with the batch's point mutations.
///
/// A range delete authorizes deleting every member within its bounds. Enumerating only
/// cached keys would let an incomplete cache hide members. Instead, establish the first
/// possible member using membership at the lower bound or an authenticated absence
/// edge covering it. Follow immediate successor edges to enumerate every remaining
/// member until the upper bound is crossed or the cyclic order wraps. A wrap proves
/// there are no further keys in this non-wrapping interval. Missing membership or
/// successor facts reject the proof, even when other cached keys could be enumerated.
///
/// A range with no lower bound starts at the least key, which the authenticated edge
/// wrapping from the greatest key establishes. With no upper bound, only the wrap ends
/// the walk. An excluded lower bound skips its member, preserving a later set at that key.
/// The caller has normalized intervals to exclude all surviving sets. An authenticated
/// empty database needs no traversal. The resulting point deletes pass through ordinary
/// operation derivation, predecessor rewrites, bitmap updates, and floor replay. Thus
/// completeness here and the child-root check together prove the whole range transition.
/// With the module's fixed-size model, O(k * log(c + p + k + 2)) worst-case work
/// and O(k) temporary operations. Each covered member is visited once because the
/// ranges are disjoint. A small number of wide ranges can still visit all active keys.
pub(super) fn expand_range_deletes<'a, F, K, V, H, const N: usize>(
    map: &TrustedMap<F, K, V, H, N>,
    user_ops: &'a BTreeSet<ByStart<K, V::Value>>,
) -> Result<PointMutations<'a, K, V::Value>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    let mut points = BTreeMap::new();
    let mut ranges = Vec::new();
    for ByStart(op) in user_ops {
        match op {
            UserOp::Set { key, value } => {
                points.insert(key.clone(), Some(value));
            }
            UserOp::Delete { key } => {
                points.insert(key.clone(), None);
            }
            UserOp::DeleteRange { start, end } => ranges.push((start, end)),
        }
    }
    if ranges.is_empty() || matches!(map.state(), TrustedState::EmptyDatabase) {
        return Ok(points);
    }
    let partial = map.partial().ok_or(VerifyError::InvalidTrustedState)?;
    for (lower, upper) in ranges {
        let mut current = match lower {
            Bound::Unbounded => partial
                .first_key()
                .ok_or(InsufficientPreState::OrderedEdge)?
                .clone(),
            Bound::Included(start) if partial.is_claimed_member(start) => start.clone(),
            Bound::Excluded(start) if partial.is_claimed_member(start) => {
                let next = partial
                    .known_edge(start)
                    .ok_or(InsufficientPreState::OrderedEdge)?;
                if next <= start {
                    continue;
                }
                next.clone()
            }
            Bound::Included(start) | Bound::Excluded(start) => {
                let (_, next) = partial
                    .containing_edge(start)
                    .ok_or(InsufficientPreState::Classification)?;
                if next <= start {
                    continue;
                }
                next.clone()
            }
        };
        while range_contains(lower, upper, &current) {
            let next = exact_entry(map, &current)?.next_key;
            let wrapped = next <= &current;
            points.insert(current, None);
            if wrapped {
                break;
            }
            current = next.clone();
        }
    }
    Ok(points)
}

/// Construct the application operations required by the caller's logical mutations.
///
/// The post-state root commits to encoded operations in log order, including successor
/// links and predecessor rewrites. Deriving these operations from trusted old records
/// and `user_ops` establishes which physical changes the requested batch requires.
///
/// Classify each mutated key using membership or an absence interval. Missing
/// classification is an error. Ignore deletes of keys proven absent. Determine
/// emptiness only from a trusted empty database or a complete successor cycle of
/// deleted members, so the floor stage can safely use its empty-database rule.
///
/// Emit deletes, updates of existing keys, and creations, each in key order. Compute
/// each final successor by skipping deleted old neighbors and choosing any nearer
/// new key in cyclic order. Then traverse created keys followed by deleted keys,
/// each group in key order. Rewrite each affected surviving predecessor once,
/// retaining its authenticated value. This traversal order fixes the physical log
/// and therefore its root. A predecessor directly mutated by the caller already has
/// its final update or delete. The three mutation groups are disjoint, and each
/// remaining predecessor is rewritten once, so the suffix has at most one operation
/// per key. Floor replay relies on every update in this suffix being live.
///
/// Record every invalidated old location. The suffix order determines appended
/// locations for the bitmap and floor stages. All cache facts must belong to the
/// trusted pre-state.
/// Worst-case work is O(k * log(c + p + k + 2)) and additional memory is O(k)
/// for fixed-size records. Memoization makes overlapping deleted-neighbor walks
/// visit O(k) anchors in total; this does not scan unrelated cached entries.
pub(crate) fn derive_ordered_transition<F, K, V, H, const N: usize>(
    map: &TrustedMap<F, K, V, H, N>,
    points: &PointMutations<'_, K, V::Value>,
) -> Result<PreparedOrderedBatch<F, K, V>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    // Existing keys the batch deletes or sets, and absent keys it creates with the key whose
    // successor edge covers each; a delete of an absent key changes nothing.
    let mut deleted = BTreeMap::new();
    let mut updated = BTreeMap::new();
    let mut created = BTreeMap::new();
    for (key, value) in points {
        let member = map
            .partial()
            .is_some_and(|partial| partial.is_claimed_member(key));
        if member {
            let entry = exact_entry(map, key)?;
            if let Some(value) = value {
                updated.insert(key.clone(), (*value, entry));
            } else {
                deleted.insert(key.clone(), entry);
            }
            continue;
        }
        let anchor = if matches!(map.state(), TrustedState::EmptyDatabase) {
            None
        } else {
            Some(
                map.partial()
                    .and_then(|partial| partial.containing_edge(key))
                    .ok_or(InsufficientPreState::Classification)?
                    .0
                    .clone(),
            )
        };
        if let Some(value) = value {
            created.insert(key.clone(), (*value, anchor));
        }
    }

    let post_empty = if !updated.is_empty() || !created.is_empty() {
        false
    } else if matches!(map.state(), TrustedState::EmptyDatabase) {
        true
    } else if let Some(first) = deleted.keys().next() {
        let mut current = first;
        let mut visited = BTreeSet::new();
        loop {
            match deleted.get(current) {
                Some(entry) if visited.insert(current) => current = entry.next_key,
                _ => break current == first && visited.len() == deleted.len(),
            }
        }
    } else {
        false
    };

    let mut operations = Vec::new();
    let mut superseded = BTreeSet::new();
    let mut surviving_successors = BTreeMap::new();
    let mut final_next = |query: &K, anchor: &K| -> Result<K, VerifyError> {
        let survivor = surviving_neighbor(anchor, &deleted, &mut surviving_successors, |key| {
            map.partial()
                .and_then(|partial| partial.known_edge(key))
                .cloned()
        })?;
        let next = cyclic_next(&created, query);
        survivor.map_or_else(
            || {
                next.cloned()
                    .ok_or(InsufficientPreState::OrderedEdge.into())
            },
            |survivor| {
                Ok(next
                    .filter(|next| strict_span_contains(query, &survivor, next))
                    .cloned()
                    .unwrap_or(survivor))
            },
        )
    };

    for (key, entry) in &deleted {
        operations.push(Operation::Delete(key.clone()));
        superseded.insert(entry.location);
    }
    let mut push_update = |key: K, value: V::Value, next_key: K| {
        operations.push(Operation::Update(update::Ordered {
            key,
            value,
            next_key,
        }));
    };
    for (key, (value, entry)) in &updated {
        let next_key = final_next(key, key)?;
        push_update(key.clone(), (*value).clone(), next_key);
        superseded.insert(entry.location);
    }
    for (key, (value, anchor)) in &created {
        let next_key = if let Some(anchor) = anchor {
            final_next(key, anchor)?
        } else {
            cyclic_next(&created, key)
                .cloned()
                .ok_or(VerifyError::ConflictingPreState)?
        };
        push_update(key.clone(), (*value).clone(), next_key);
    }

    // Created keys' anchors, then deleted keys' predecessors, each group in key order.
    let created_predecessors = created
        .values()
        .filter_map(|(_, anchor)| anchor.clone().map(Ok));
    let deleted_predecessors = deleted.keys().map(|key| {
        map.partial()
            .and_then(|partial| partial.predecessor_of(key))
            .cloned()
            .ok_or(VerifyError::from(InsufficientPreState::OrderedEdge))
    });
    let mut rewritten = BTreeSet::new();
    let mut surviving_predecessors = BTreeMap::new();
    for candidate in created_predecessors.chain(deleted_predecessors) {
        let candidate = candidate?;
        let predecessor = if !deleted.contains_key(&candidate) {
            Some(candidate)
        } else {
            surviving_neighbor(&candidate, &deleted, &mut surviving_predecessors, |key| {
                map.partial()
                    .and_then(|partial| partial.predecessor_of(key))
                    .cloned()
            })?
        };
        let Some(predecessor) = predecessor else {
            continue;
        };
        // A surviving predecessor the caller set already carries its final link.
        if updated.contains_key(&predecessor) || !rewritten.insert(predecessor.clone()) {
            continue;
        }

        let entry = exact_entry(map, &predecessor)?;
        let next_key = final_next(&predecessor, &predecessor)?;
        push_update(
            predecessor,
            entry.value.ok_or(InsufficientPreState::Value)?.clone(),
            next_key,
        );
        superseded.insert(entry.location);
    }

    Ok(PreparedOrderedBatch {
        application_ops: operations,
        superseded,
        deleted_keys: deleted.into_keys().collect(),
        post_empty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        merkle::mmr,
        qmdb::{
            any::value::FixedEncoding,
            current::verifier::{CurrentCommitment, PartialTrustedState, UserBatch},
        },
    };
    use alloc::vec;
    use commonware_cryptography::Sha256;
    use commonware_utils::sequence::FixedBytes;

    type F = mmr::Family;
    type Key = FixedBytes<8>;
    type Map = TrustedMap<F, Key, FixedEncoding<Key>, Sha256, 32>;
    type Batch = UserBatch<Key, Key>;

    /// An application operation as its key and, for an update, its value and successor.
    type Plain = (u64, Option<(u64, u64)>);

    fn key(index: u64) -> Key {
        FixedBytes::new(index.to_be_bytes())
    }

    fn index(key: &Key) -> u64 {
        u64::from_be_bytes(key.as_ref().try_into().unwrap())
    }

    /// Bind `state` to a ten-operation checkpoint.
    fn bind(state: TrustedState<F, Key, Key>) -> Map {
        let commitment =
            CurrentCommitment::from_parts(Location::new(10), Sha256::hash(&[b"root".as_slice()]))
                .unwrap();
        TrustedMap::from_state(commitment, state).unwrap()
    }

    fn partial(entries: impl IntoIterator<Item = TrustedEntry<F, Key, Key>>) -> Map {
        bind(TrustedState::Partial(
            PartialTrustedState::new(entries, []).unwrap(),
        ))
    }

    /// Every fact about a database whose keys 10, 20, 30 and 40 sit at locations 1 to 4, each
    /// holding its own key as value.
    fn full() -> Map {
        let keys = [10, 20, 30, 40];
        partial((0..keys.len()).map(|i| TrustedEntry {
            key: key(keys[i]),
            location: KnownTrusted(Location::new(i as u64 + 1)),
            value: KnownTrusted(key(keys[i])),
            next_key: KnownTrusted(key(keys[(i + 1) % keys.len()])),
        }))
    }

    /// Expand the batch's ranges and derive its transition.
    fn derive(
        map: &Map,
        batch: &Batch,
    ) -> Result<PreparedOrderedBatch<F, Key, FixedEncoding<Key>>, VerifyError> {
        derive_ordered_transition(map, &expand_range_deletes(map, batch.ops())?)
    }

    fn plain(prepared: &PreparedOrderedBatch<F, Key, FixedEncoding<Key>>) -> Vec<Plain> {
        prepared
            .application_ops
            .iter()
            .map(|op| match op {
                Operation::Delete(key) => (index(key), None),
                Operation::Update(update) => (
                    index(&update.key),
                    Some((index(&update.value), index(&update.next_key))),
                ),
                Operation::CommitFloor(..) => unreachable!("derivation appends no commit"),
            })
            .collect()
    }

    fn superseded(prepared: &PreparedOrderedBatch<F, Key, FixedEncoding<Key>>) -> Vec<u64> {
        prepared
            .superseded
            .iter()
            .map(|location| **location)
            .collect()
    }

    fn set(index: u64, value: u64) -> UserOp<Key, Key> {
        UserOp::set(key(index), key(value))
    }

    fn delete(index: u64) -> UserOp<Key, Key> {
        UserOp::delete(key(index))
    }

    #[test]
    fn a_creation_rewrites_its_predecessor() {
        let prepared = derive(&full(), &Batch::from([set(25, 99)])).unwrap();
        assert_eq!(
            plain(&prepared),
            vec![(25, Some((99, 30))), (20, Some((20, 25)))]
        );
        assert_eq!(superseded(&prepared), vec![2]);
        assert_eq!(prepared.made_inactive(), 1);
        assert!(!prepared.post_empty);
    }

    #[test]
    fn a_creation_after_the_last_key_wraps() {
        let prepared = derive(&full(), &Batch::from([set(50, 99)])).unwrap();
        assert_eq!(
            plain(&prepared),
            vec![(50, Some((99, 10))), (40, Some((40, 50)))]
        );
    }

    #[test]
    fn deleting_a_run_links_around_it_once() {
        let prepared = derive(&full(), &Batch::from([delete(20), delete(30)])).unwrap();
        assert_eq!(
            plain(&prepared),
            vec![(20, None), (30, None), (10, Some((10, 40)))]
        );
        assert_eq!(superseded(&prepared), vec![1, 2, 3]);
        // Each delete is itself inactive, beside the three records it supersedes.
        assert_eq!(prepared.made_inactive(), 5);
    }

    #[test]
    fn a_set_predecessor_carries_its_new_link() {
        let prepared = derive(&full(), &Batch::from([set(10, 77), set(15, 88)])).unwrap();
        assert_eq!(
            plain(&prepared),
            vec![(10, Some((77, 15))), (15, Some((88, 20)))]
        );
        assert_eq!(superseded(&prepared), vec![1]);
    }

    #[test]
    fn deleting_every_key_empties_the_database() {
        let mut batch = Batch::new();
        batch.delete_range(key(0)..=key(100)).unwrap();
        let prepared = derive(&full(), &batch).unwrap();
        assert_eq!(
            plain(&prepared),
            vec![(10, None), (20, None), (30, None), (40, None)]
        );
        assert!(prepared.post_empty);
    }

    #[test]
    fn creations_in_an_empty_database_form_a_cycle() {
        let empty = bind(TrustedState::EmptyDatabase);
        let mut batch = Batch::from([set(5, 1), set(7, 2), delete(9)]);
        batch.delete_range(key(20)..=key(30)).unwrap();
        let prepared = derive(&empty, &batch).unwrap();
        assert_eq!(plain(&prepared), vec![(5, Some((1, 7))), (7, Some((2, 5)))]);
        assert!(prepared.superseded.is_empty() && !prepared.post_empty);
    }

    #[test]
    fn ranges_enumerate_members_through_successors() {
        let expanded = |batch: &Batch| {
            expand_range_deletes(&full(), batch.ops())
                .unwrap()
                .into_iter()
                .map(|(key, value)| (index(&key), value.is_some()))
                .collect::<Vec<_>>()
        };
        let mut batch = Batch::new();
        batch.delete_range(key(15)..=key(35)).unwrap();
        assert_eq!(expanded(&batch), vec![(20, false), (30, false)]);
        // A later set at 20 leaves the range starting just after it.
        batch.delete_range(key(20)..=key(40)).unwrap();
        batch.set(key(20), key(1));
        assert_eq!(expanded(&batch), vec![(20, true), (30, false), (40, false)]);
        let mut batch = Batch::new();
        batch.delete_range(key(5)..=key(15)).unwrap();
        batch.delete_range(key(45)..=key(60)).unwrap();
        assert_eq!(expanded(&batch), vec![(10, false)]);
    }

    #[test]
    fn open_ranges_start_at_the_least_key_and_end_at_the_wrap() {
        let expanded = |map: &Map, range: (Bound<Key>, Bound<Key>)| {
            let mut batch = Batch::new();
            batch.delete_range(range).unwrap();
            expand_range_deletes(map, batch.ops())
                .map(|points| points.keys().map(index).collect::<Vec<_>>())
        };
        let full = full();
        assert_eq!(
            expanded(&full, (Bound::Unbounded, Bound::Included(key(25)))),
            Ok(vec![10, 20])
        );
        assert_eq!(
            expanded(&full, (Bound::Excluded(key(20)), Bound::Unbounded)),
            Ok(vec![30, 40])
        );
        assert_eq!(
            expanded(&full, (Bound::Unbounded, Bound::Unbounded)),
            Ok(vec![10, 20, 30, 40])
        );
        // A range below the least key deletes nothing, but still needs the wrap to say so.
        assert_eq!(
            expanded(&full, (Bound::Unbounded, Bound::Excluded(key(10)))),
            Ok(vec![])
        );
        // Without the greatest key's record, nothing says which key is least: 10 is not
        // enough, since another key could precede it.
        let without_wrap = partial([10, 20, 30].map(|i| TrustedEntry {
            key: key(i),
            location: KnownTrusted(Location::new(i / 10)),
            value: KnownTrusted(key(i)),
            next_key: KnownTrusted(key(i + 10)),
        }));
        assert_eq!(
            expanded(&without_wrap, (Bound::Unbounded, Bound::Included(key(25)))),
            Err(VerifyError::InsufficientPreState(
                InsufficientPreState::OrderedEdge
            ))
        );
    }

    #[test]
    fn missing_facts_are_errors_not_guesses() {
        let missing =
            |fact| -> Result<(), VerifyError> { Err(VerifyError::InsufficientPreState(fact)) };
        let derived = |map: &Map, batch: Batch| derive(map, &batch).map(|_| ());
        // Key 20 at location 2, holding 20 and followed by 30, without the facts marked false.
        let twenty = |location: bool, value: bool, next: bool| TrustedEntry {
            key: key(20),
            location: if location {
                KnownTrusted(Location::new(2))
            } else {
                Unknown
            },
            value: if value {
                KnownTrusted(key(20))
            } else {
                Unknown
            },
            next_key: if next { KnownTrusted(key(30)) } else { Unknown },
        };

        // Nothing classifies 25 as a member or a non-member.
        assert_eq!(
            derived(&partial([]), Batch::from([set(25, 1)])),
            missing(InsufficientPreState::Classification)
        );
        // Updating 20 needs its location.
        assert_eq!(
            derived(
                &partial([twenty(false, true, true)]),
                Batch::from([set(20, 1)])
            ),
            missing(InsufficientPreState::Location)
        );
        // Deleting 20 needs its predecessor, to link around it.
        assert_eq!(
            derived(
                &partial([twenty(true, true, true)]),
                Batch::from([delete(20)])
            ),
            missing(InsufficientPreState::OrderedEdge)
        );
        // Creating 25 rewrites 20, which needs 20's value.
        assert_eq!(
            derived(
                &partial([twenty(true, false, true)]),
                Batch::from([set(25, 1)])
            ),
            missing(InsufficientPreState::Value)
        );
        // A range cannot skip past a member whose successor is unknown.
        let no_successor = partial([
            TrustedEntry {
                key: key(10),
                location: KnownTrusted(Location::new(1)),
                value: KnownTrusted(key(10)),
                next_key: KnownTrusted(key(20)),
            },
            twenty(true, true, false),
        ]);
        let mut batch = Batch::new();
        batch.delete_range(key(15)..=key(35)).unwrap();
        assert_eq!(
            derived(&no_successor, batch),
            missing(InsufficientPreState::NextKey)
        );
    }
}
