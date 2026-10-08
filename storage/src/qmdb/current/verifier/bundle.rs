//! Pre-state records and Merkle proofs supplied for one batch verification.

use super::BitmapProof;
use crate::{
    merkle::{self, Family, Graftable, Location},
    qmdb::{
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        current::{ordered, proof::OpsRootWitness},
        operation::Key,
    },
};
use alloc::vec::Vec;
use commonware_cryptography::{Digest, Hasher};

/// An old log location paired with the update payload needed for floor relocation.
/// The raw multi-proof authenticates the payload. The independently authenticated
/// bitmap establishes that its location must be considered by the scan.
pub type CandidateOperation<F, K, V> = (Location<F>, Operation<F, update::Ordered<K, V>>);

/// Bounds repeated by the bundle and checked for consistency, never trusted as inputs.
/// `pre_size` comes from the trusted commitment and `pre_floor` from its terminal
/// proof. `post_size` and `post_floor` come from local execution. These fields add
/// no independent evidence: a mismatch rejects the bundle even if other proofs pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClaimedBatchBounds<F: Family> {
    /// Claimed pre-state operation count.
    pub pre_size: Location<F>,
    /// Claimed pre-state inactivity floor.
    pub pre_floor: Location<F>,
    /// Claimed post-state operation count.
    pub post_size: Location<F>,
    /// Claimed post-state inactivity floor.
    pub post_floor: Location<F>,
}

/// Supply the root components and terminal proof for a Current commitment.
/// The verifier checks the root witness and terminal inclusion before using either
/// the supplied operations root or floor to derive the transition.
#[derive(Clone)]
pub enum PreStateOpeningClaim<F, V, D, const N: usize>
where
    F: Graftable,
    V: ValueEncoding,
    D: Digest,
{
    /// The fixed one-operation initial state.
    Initial,
    /// Claimed opening of a non-initial state, authenticated during verification.
    Proven {
        /// Claimed raw operations root.
        ops_root: D,
        /// Witness binding the raw root to the canonical Current root.
        witness: OpsRootWitness<F, D>,
        /// Terminal commit and raw append-frontier proof.
        terminal: PreStateTerminalClaim<F, V, D>,
    },
}

/// Prove the last log operation is `CommitFloor(metadata, floor)` and recover append nodes.
/// The verifier derives its location as `pre_size - 1`. The raw proof authenticates
/// both that payload and the pinned nodes immediately before its append. Replaying
/// the append yields the raw frontier for extending the committed log.
#[derive(Clone)]
pub struct PreStateTerminalClaim<F, V, D>
where
    F: Family,
    V: ValueEncoding,
    D: Digest,
{
    /// Claimed terminal metadata.
    pub metadata: Option<V::Value>,
    /// Claimed exact inactivity floor.
    pub floor: Location<F>,
    /// Raw inclusion proof for the derived terminal location.
    pub raw_proof: merkle::Proof<F, D>,
    /// Raw digests in `F::nodes_to_pin(pre_size - 1)` order, checked with `raw_proof`.
    pub pins_before_terminal: Vec<D>,
}

/// Claim that `key` has `value` at an active location in the pre-state.
/// The membership proof supplies the location and immediate cyclic successor. The
/// verifier authenticates the reconstructed update, its activity bit, and its
/// agreement with the common pre-state opening before adding it to the cache.
#[derive(Clone)]
pub struct PreStateMemberClaim<F, K, V, D, const N: usize>
where
    F: Graftable,
    K: Key,
    D: Digest,
{
    /// Claimed member key.
    pub key: K,
    /// Claimed member value.
    pub value: V,
    /// Ordered Current membership proof containing the location and successor.
    pub proof: ordered::proof::constant::KeyValueProof<F, K, D, N>,
}

/// Claim that `queried_key` is absent from the pre-state's active keys.
/// A key-value proof authenticates the active predecessor and its successor, whose
/// open cyclic interval must contain the queried key. A commit proof establishes
/// an empty database by placing the terminal commit at the inactivity floor.
#[derive(Clone)]
pub struct PreStateExclusionClaim<F, K, V, D, const N: usize>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: Digest,
{
    /// Key claimed to be absent.
    pub queried_key: K,
    /// Ordered Current exclusion proof containing a predecessor or an empty-state commit.
    pub proof: ordered::proof::constant::ExclusionProof<F, K, V, D, N>,
}

/// One untrusted ordered pre-state fact supplied by the prover.
#[derive(Clone)]
pub enum PreStateKeyClaim<F, K, V, D, const N: usize>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: Digest,
{
    /// Claimed active record.
    Member(PreStateMemberClaim<F, K, V::Value, D, N>),
    /// Claimed absent key.
    Exclusion(PreStateExclusionClaim<F, K, V, D, N>),
}

/// Raw payload proof for every pre-state live bit in the proposed floor scan.
/// Locations must equal the bitmap's set bits in `[old_floor, floor_scan_end)`,
/// excluding the terminal commit. This excludes omitted candidates as well as
/// extras. Inclusion alone would not prove scan completeness. All payloads must
/// be updates. The floor stage decides which remain live after the user batch.
#[derive(Clone)]
pub struct PreStateCandidateProof<F, K, V, D>
where
    F: Family,
    K: Key,
    V: ValueEncoding,
    D: Digest,
{
    /// Claimed operations, required to be strictly increasing by location.
    pub operations: Vec<CandidateOperation<F, K, V>>,
    /// Shared raw operations-tree proof, present exactly when `operations` is non-empty.
    pub raw_multi_proof: Option<merkle::Proof<F, D>>,
}

/// Evidence supplied by a prover for verification with [`super::UserBatch::verify`].
///
/// The opening supplies the old root components and terminal commit. Key claims
/// supply records and proofs of membership or absence to fill gaps in the cache.
/// Bitmap proofs and candidate payloads let the verifier replay floor movement and
/// compute the new root. Verification checks the claimed bounds and scan endpoint
/// against that execution.
///
/// These fields describe the pre-state and claim bounds for the transition. The
/// caller supplies the requested mutations and metadata through `UserBatch`.
/// The verifier uses both inputs to construct the new physical operations.
/// Required proofs depend on the batch and the facts already retained in the cache.
/// Missing facts produce an error.
/// For range deletes, supply the lower-bound classification and the successor chain
/// through every deleted member, plus the surviving predecessors needed for rewrites.
/// Existing member and exclusion claims carry these facts; range completeness is
/// established from the chain before the ordinary transition and floor checks.
pub struct BatchProofBundle<F, K, V, H, const N: usize>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    /// Bounds compared with the authenticated pre-state and locally derived post-state.
    pub claims: ClaimedBatchBounds<F>,
    /// Root components, terminal commit, and proof of the raw append frontier.
    pub pre_state: PreStateOpeningClaim<F, V, H::Digest, N>,
    /// Member and exclusion claims to authenticate before adding facts to the cache.
    pub pre_state_claims: Vec<PreStateKeyClaim<F, K, V, H::Digest, N>>,
    /// Proof covering every bitmap chunk needed by the transition.
    pub bitmap: BitmapProof<F, H::Digest, N>,
    /// Old operation payloads and inclusion proof for the floor scan's set bits.
    pub pre_state_candidates: PreStateCandidateProof<F, K, V, H::Digest>,
}

impl<F, K, V, H, const N: usize> Clone for BatchProofBundle<F, K, V, H, N>
where
    F: Graftable + Clone,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    /// Copy the untrusted claims and witness vectors without authenticating them.
    fn clone(&self) -> Self {
        Self {
            claims: self.claims,
            pre_state: self.pre_state.clone(),
            pre_state_claims: self.pre_state_claims.clone(),
            bitmap: self.bitmap.clone(),
            pre_state_candidates: self.pre_state_candidates.clone(),
        }
    }
}

impl<F, K, V, H, const N: usize> BatchProofBundle<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    /// Package proof claims for a later call to `UserBatch::verify`.
    /// This constructor performs no authentication or completeness checks. The verifier
    /// checks all components together against the pre-commitment and the authorized batch.
    pub const fn new(
        claims: ClaimedBatchBounds<F>,
        pre_state: PreStateOpeningClaim<F, V, H::Digest, N>,
        pre_state_claims: Vec<PreStateKeyClaim<F, K, V, H::Digest, N>>,
        bitmap: BitmapProof<F, H::Digest, N>,
        pre_state_candidates: PreStateCandidateProof<F, K, V, H::Digest>,
    ) -> Self {
        Self {
            claims,
            pre_state,
            pre_state_claims,
            bitmap,
            pre_state_candidates,
        }
    }
}
