//! Compose proof authentication, deterministic execution, and cache rebasing.
//!
//! Run the stages with one authenticated pre-state and the caller's batch. Return
//! the verified transition only after the derived state matches the claimed commitment;
//! rebasing the cache onto that state is left to the caller.

use super::{
    BatchProofBundle, CurrentCommitment,
    OptionTrusted::KnownTrusted,
    PartialTrustedState, TrustedEntry, TrustedMap, TrustedState, UserBatch, UserOp, VerifyError,
    bitmap::verify_bitmap,
    commitment::validate_profile,
    floor::{FloorResult, advance_floor, verify_candidates},
    opening::authenticate_pre_state,
    ordered::{
        PreparedOrderedBatch, derive_ordered_transition, expand_range_deletes,
        verify_and_merge_claims,
    },
    root::derive_child_commitment,
};
use crate::{
    merkle::{Graftable, Location},
    qmdb::{
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        floor::Policy,
        operation::Key,
    },
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use commonware_codec::Codec;
use commonware_cryptography::Hasher;
use commonware_parallel::Strategy;
use core::fmt;

/// A batch transition the verifier accepted.
///
/// It holds the child commitment the batch produced, which equals the claimed
/// post-commitment, and the facts verification established at the parent: the caller's
/// cache plus every record and absence the bundle authenticated. [`Self::into_cache`]
/// carries those facts across the transition for the next verification. A caller that keeps
/// no cache can drop this value instead and skip that work.
pub struct Verified<F, K, V, H, const N: usize>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    parent: TrustedMap<F, K, V, H, N>,
    child: CurrentCommitment<F, K, V, H, N>,
    prepared: PreparedOrderedBatch<F, K, V>,
    floor: FloorResult<F, K, V>,
    set_keys: Vec<K>,
}

impl<F, K, V, H, const N: usize> Verified<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    /// Return the child commitment, which equals the claimed post-commitment.
    pub const fn commitment(&self) -> &CurrentCommitment<F, K, V, H, N> {
        &self.child
    }

    /// Return the facts established at the parent commitment, for example to check what a
    /// key held before the batch. They are bound to the parent, not the child.
    pub const fn parent(&self) -> &TrustedMap<F, K, V, H, N> {
        &self.parent
    }

    /// Carry the parent facts across the transition and return them bound to the child.
    ///
    /// The last derived update for a key determines its new location, value, and edge,
    /// including floor copies. Deleted entries are removed and rewritten ones replaced.
    /// Absence intervals cut by insertions or changed adjacency are removed, and only
    /// intervals justified by final updates are restored. When the child is empty, every
    /// record is discarded and that fact retained instead.
    ///
    /// # Errors
    ///
    /// Returns [`VerifyError::InvalidTrustedState`] when a retained fact contradicts the
    /// transition, such as a live record left below the new floor that the batch neither
    /// moved nor deleted. That happens only if the caller's cache held a fact that was not
    /// true at the parent.
    pub fn into_cache(self) -> Result<TrustedMap<F, K, V, H, N>, VerifyError> {
        let base_size = self.parent.commitment().size();
        let mut cache = self.parent;
        cache.rebind(self.child);
        rebase_cache(
            &mut cache,
            &self.prepared,
            &self.floor,
            &self.set_keys,
            base_size,
        )?;
        Ok(cache)
    }
}

impl<F, K, V, H, const N: usize> fmt::Debug for Verified<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Verified")
            .field("parent", self.parent.commitment())
            .field("child", &self.child)
            .finish_non_exhaustive()
    }
}

/// Carry old trusted facts across the already verified physical suffix.
///
/// A successful root check establishes the new state, but facts cached at the old
/// root can now have stale locations or successor links. Update or remove those
/// facts before returning a cache that later transitions can safely trust.
/// `set_keys` are the keys the batch set. Reject a retained live location below the new
/// floor unless the suffix moved or deleted it. The caller has rebound the owned cache to
/// the child commitment and must discard it if rebasing fails.
fn rebase_cache<F, K, V, H, const N: usize>(
    map: &mut TrustedMap<F, K, V, H, N>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    floor: &FloorResult<F, K, V>,
    set_keys: &[K],
    base_size: Location<F>,
) -> Result<(), VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    if prepared.post_empty {
        map.set_state(TrustedState::EmptyDatabase);
        return Ok(());
    }
    // The authenticated transition bounds every operation in this derived suffix.
    let updates = prepared
        .application_ops
        .iter()
        .chain(&floor.copies)
        .zip(0u64..)
        .filter_map(|(operation, index)| match operation {
            Operation::Update(update) => Some((update.key.clone(), (base_size + index, update))),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let was_empty = matches!(map.state(), TrustedState::EmptyDatabase);
    if was_empty {
        map.set_state(TrustedState::Partial(PartialTrustedState::default()));
    }
    let Some(partial) = map.partial() else {
        return Err(VerifyError::InvalidTrustedState);
    };
    if partial
        .keys_below_location(floor.floor)
        .any(|key| !prepared.deleted_keys.contains(key) && !updates.contains_key(key))
    {
        return Err(VerifyError::InvalidTrustedState);
    }

    // Keys whose final update must re-establish its successor span, with that successor.
    let mut spans_to_remove = BTreeSet::new();
    let mut explicit_update_spans = BTreeMap::new();
    if was_empty {
        explicit_update_spans.extend(
            updates
                .iter()
                .map(|(key, (_, update))| (key.clone(), &update.next_key)),
        );
    }
    for (key, (_, update)) in &updates {
        if partial.absences().successor(key).is_some() || partial.absences().get(key).is_some() {
            explicit_update_spans.insert(key.clone(), &update.next_key);
        }
        if partial
            .absences()
            .successor(key)
            .is_some_and(|successor| successor != &update.next_key)
        {
            spans_to_remove.insert(key.clone());
        }
    }
    for key in &prepared.deleted_keys {
        spans_to_remove.insert(key.clone());
        if let Some(predecessor) = partial.predecessor_of(key) {
            spans_to_remove.insert(predecessor.clone());
        }
    }
    for key in set_keys {
        if let Some((predecessor, _)) = partial.containing_edge(key) {
            spans_to_remove.insert(predecessor.clone());
        }
    }
    for predecessor in spans_to_remove
        .iter()
        .filter(|key| partial.absences().successor(key).is_some())
    {
        let final_update = updates.get_key_value(predecessor).or_else(|| {
            super::trusted::containing_edge(&updates, predecessor, |(_, update)| &update.next_key)
        });
        if let Some((key, (_, update))) = final_update {
            explicit_update_spans.insert(key.clone(), &update.next_key);
        }
    }

    for predecessor in spans_to_remove {
        map.remove_span(&predecessor);
    }
    for key in &prepared.deleted_keys {
        map.remove_entry(key);
    }
    for key in updates.keys() {
        map.remove_entry(key);
    }
    for (key, (location, update)) in &updates {
        map.replace_entry(TrustedEntry {
            key: key.clone(),
            location: KnownTrusted(*location),
            value: KnownTrusted(update.value.clone()),
            next_key: KnownTrusted(update.next_key.clone()),
        })?;
    }
    for (key, next_key) in explicit_update_spans {
        map.install_span(key, next_key.clone())?;
    }
    Ok(())
}

/// Verify one transition, and return it with the facts it established at the parent.
///
/// Require the cache to already be bound to `pre_commitment`. Authenticate the opening and
/// key claims, derive ordered operations, authenticate scan coverage and payloads,
/// and replay floor advancement. Reconstruct the child root from those results.
/// Comparing the derived bounds and commitment with the claims establishes that the
/// claimed transition follows from the authorized batch, including floor movement.
/// Only after all comparisons succeed is the transition returned.
///
/// The caller supplies authorized mutations, a valid trusted pre-state, and true
/// cache facts. Each stage consumes results authenticated or derived during this call.
pub(super) fn verify_batch<F, K, V, H, P, const N: usize, S: Strategy>(
    pre_commitment: CurrentCommitment<F, K, V, H, N>,
    mut cache: TrustedMap<F, K, V, H, N>,
    batch: &UserBatch<K, V::Value>,
    proof_bundle: &BatchProofBundle<F, K, V, H, N>,
    claimed_post: CurrentCommitment<F, K, V, H, N>,
    strategy: &S,
    policy: &mut P,
) -> Result<Verified<F, K, V, H, N>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    P: Policy<F, K, V::Value>,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    validate_profile::<H, N>();
    if cache.commitment() != &pre_commitment {
        return Err(VerifyError::TrustedCommitmentMismatch);
    }
    if proof_bundle.claims.pre_size != pre_commitment.size() {
        return Err(VerifyError::ClaimedBoundsMismatch);
    }
    let mut claim_keys = BTreeSet::new();
    for claim in &proof_bundle.pre_state_claims {
        let key = match claim {
            super::PreStateKeyClaim::Member(claim) => &claim.key,
            super::PreStateKeyClaim::Exclusion(claim) => &claim.queried_key,
        };
        if !claim_keys.insert(key) {
            return Err(VerifyError::ConflictingPreState);
        }
    }
    let pre = authenticate_pre_state(pre_commitment, &proof_bundle.pre_state)?;
    if proof_bundle.claims.pre_floor != pre.old_floor {
        return Err(VerifyError::ClaimedBoundsMismatch);
    }
    let authenticated_empty = pre.old_floor == pre.terminal_location;
    match (authenticated_empty, cache.state()) {
        (true, TrustedState::Partial(partial)) => {
            if partial.entries().next().is_some() || !partial.absences().is_empty() {
                return Err(VerifyError::InvalidTrustedState);
            }
            cache.set_state(TrustedState::EmptyDatabase);
        }
        (false, TrustedState::EmptyDatabase) => {
            return Err(VerifyError::InvalidTrustedState);
        }
        _ => {}
    }
    if cache
        .partial()
        .is_some_and(|partial| !partial.locations_within(pre.old_floor, pre.terminal_location))
    {
        return Err(VerifyError::InvalidTrustedState);
    }
    verify_and_merge_claims(&pre, &mut cache, &proof_bundle.pre_state_claims, strategy)?;
    let points = expand_range_deletes(&cache, batch.ops())?;
    let prepared = derive_ordered_transition(&cache, &points)?;
    let bitmap = verify_bitmap(&pre, &prepared, &proof_bundle.bitmap, strategy)?;
    let candidates =
        verify_candidates(&pre, &bitmap, &proof_bundle.pre_state_candidates)?;
    let floor = advance_floor(
        &pre,
        &prepared,
        &bitmap,
        &candidates,
        batch.metadata().cloned(),
        policy,
    )?;
    let derived = derive_child_commitment(&pre, &prepared, &floor, bitmap, strategy)?;
    if proof_bundle.claims.post_size != derived.size()
        || proof_bundle.claims.post_floor != floor.floor
    {
        return Err(VerifyError::ClaimedBoundsMismatch);
    }
    if derived != claimed_post {
        return Err(VerifyError::PostCommitmentMismatch);
    }
    let set_keys = batch
        .ops()
        .iter()
        .filter_map(|op| match &op.0 {
            UserOp::Set { key, .. } => Some(key.clone()),
            UserOp::Delete { .. } | UserOp::DeleteRange { .. } => None,
        })
        .collect();
    Ok(Verified {
        parent: cache,
        child: derived,
        prepared,
        floor,
        set_keys,
    })
}
