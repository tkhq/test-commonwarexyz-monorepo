//! Turn authenticated cyclic adjacency into the canonical physical operation suffix.
//!
//! An ordered update stores its key, value, and immediate cyclic successor. Inserting
//! or deleting a key may therefore require rewriting a surviving predecessor even
//! when the caller did not change its value. The proof bundle supplies pre-state
//! records and proofs of membership or absence. This module authenticates those facts,
//! then combines them with the trusted cache and the caller's `UserBatch` mutations
//! to construct the new updates, deletes, and predecessor rewrites in canonical order.
//! Floor copies and the terminal commit are derived in the floor stage that follows.

use super::{
    InsufficientPreState,
    OptionTrusted::{KnownTrusted, Unknown},
    PreStateKeyClaim, TrustedEntry, TrustedMap, TrustedState, UserOp, VerifyError,
    opening::{AuthenticatedPreState, operation_proof_matches_pre_state},
    trusted::strict_span_contains,
    user::{ByStart, range_contains},
};
use crate::{
    merkle::{Graftable, Location},
    qmdb::{
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        current::ordered::proof::constant::ExclusionProof,
        operation::Key,
    },
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use commonware_codec::{Codec, Encode as _};
use commonware_cryptography::Hasher;
use commonware_parallel::Strategy;
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

/// None authenticates an empty database; Some carries an active record and whether
/// its edge was requested explicitly as an exclusion. This is never proof input.
type AuthenticatedClaim<F, K, V> = Option<(Location<F>, update::Ordered<K, V>, bool)>;

/// Authenticate one member or exclusion before it may be merged into the cache.
///
/// Operation derivation needs to distinguish updates and deletes of existing keys
/// from creations and deletes of absent keys. A member proof establishes a live
/// update at a location between the floor and terminal. An exclusion establishes
/// an active predecessor whose open successor interval contains the queried key,
/// or an empty database whose terminal is at the floor.
///
/// Require the same root components as the pre-state opening before adding facts to
/// the cache. The activity check prevents an old, superseded record from establishing
/// current membership or absence. An exclusion's predecessor also supplies its
/// location, value, and successor for any required predecessor rewrite.
/// Cache conflicts are checked by the ordered merge after authentication.
fn authenticate_claim<F, K, V, H, const N: usize>(
    pre: &AuthenticatedPreState<F, K, V, H, N>,
    claim: &PreStateKeyClaim<F, K, V, H::Digest, N>,
) -> Result<AuthenticatedClaim<F, K, V>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    let (proof, update, exclusion) = match claim {
        PreStateKeyClaim::Member(claim) => (
            &claim.proof.proof,
            update::Ordered {
                key: claim.key.clone(),
                value: claim.value.clone(),
                next_key: claim.proof.next_key.clone(),
            },
            false,
        ),
        PreStateKeyClaim::Exclusion(claim) => match &claim.proof {
            ExclusionProof::KeyValue(proof, update) => {
                if !strict_span_contains(&update.key, &update.next_key, &claim.queried_key) {
                    return Err(VerifyError::InvalidExclusionProof);
                }
                (proof, update.clone(), true)
            }
            ExclusionProof::Commit(proof, metadata) => {
                if proof.loc != pre.terminal_location
                    || pre.old_floor != proof.loc
                    || !operation_proof_matches_pre_state(
                        pre,
                        proof,
                        &Operation::<F, update::Ordered<K, V>>::CommitFloor(
                            metadata.clone(),
                            proof.loc,
                        ),
                    )
                {
                    return Err(VerifyError::InvalidExclusionProof);
                }
                return Ok(None);
            }
        },
    };
    let operation = Operation::Update(update);
    let authentic = proof.loc >= pre.old_floor
        && proof.loc < pre.terminal_location
        && operation_proof_matches_pre_state(pre, proof, &operation);
    match operation {
        Operation::Update(update) if authentic => Ok(Some((proof.loc, update, exclusion))),
        _ if exclusion => Err(VerifyError::InvalidExclusionProof),
        _ => Err(VerifyError::InvalidMemberProof),
    }
}

/// Authenticate supplied facts independently, then merge them in their original order.
/// Workers never mutate the cache, and no fact is merged unless all authentication
/// succeeds. Missing or invalid proofs reject the batch under every strategy.
/// The resulting cache supplies the membership, absence, and adjacency facts used
/// to derive the application operations. The caller rejects repeated queried keys
/// before this loop. An error may follow successful earlier merges, so the caller
/// must discard the working cache on failure.
pub(crate) fn verify_and_merge_claims<F, K, V, H, const N: usize, S: Strategy>(
    pre: &AuthenticatedPreState<F, K, V, H, N>,
    map: &mut TrustedMap<F, K, V, H, N>,
    claims: &[PreStateKeyClaim<F, K, V, H::Digest, N>],
    strategy: &S,
) -> Result<(), VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    let verified = strategy.try_map_collect_vec(claims, |claim| authenticate_claim(pre, claim))?;
    for claim in verified {
        if let Some((location, update, exclusion)) = claim {
            let edge = exclusion.then(|| (update.key.clone(), update.next_key.clone()));
            merge_member(map, location, update)?;
            if let Some((key, next_key)) = edge {
                map.install_span(key, next_key)
                    .map_err(|_| VerifyError::ConflictingPreState)?;
            }
        } else {
            if let TrustedState::Partial(partial) = map.state()
                && (partial.entries().next().is_some() || !partial.absences().is_empty())
            {
                return Err(VerifyError::ConflictingPreState);
            }
            map.set_state(TrustedState::EmptyDatabase);
        }
    }
    Ok(())
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
