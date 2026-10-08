//! Authenticate the root components that all later pre-state proofs must share.
//!
//! The terminal commit fixes the old floor. Its raw proof also authenticates the
//! nodes needed to append to the log. Membership and bitmap proofs must agree with
//! this opening's raw root, grafted root, size, and trailing bitmap components.

use super::{
    CurrentCommitment, PreStateOpeningClaim, VerifyError, bitmap::set_chunk_bit,
    initial_commitment, sparse::SparseForest,
};
use crate::{
    merkle::{Graftable, Location, PendingChunk as _, Position, hasher::Hasher as _},
    qmdb::{
        self,
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        current::{grafting, proof::constant::OperationProof},
        operation::Key,
    },
};
use alloc::collections::BTreeMap;
use commonware_codec::{Codec, Encode as _};
use commonware_cryptography::Hasher;
use commonware_utils::bitmap::BitMap;

/// Authenticated root components and log bounds used throughout one transition.
/// The witness binds the bitmap components. The terminal proof binds `old_floor`.
/// `raw_frontier` contains authenticated raw nodes after appending that terminal.
pub(crate) struct AuthenticatedPreState<F, K, V, H, const N: usize>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    pub(crate) commitment: CurrentCommitment<F, K, V, H, N>,
    pub(crate) ops_root: H::Digest,
    pub(crate) old_floor: Location<F>,
    pub(crate) terminal_location: Location<F>,
    pub(crate) witness: crate::qmdb::current::proof::OpsRootWitness<F, H::Digest>,
    pub(crate) raw_frontier: BTreeMap<Position<F>, H::Digest>,
    /// Inactive-peak count for grafted proofs, rounded down so the whole prefix ends
    /// at a complete bitmap-chunk boundary. Every operation proof must carry this
    /// count; compute it once from the authenticated size and floor for all claims.
    pub(crate) grafted_inactive_peaks: usize,
    pub(crate) initial: bool,
}

/// Authenticate the root components, terminal operation, and raw append frontier.
///
/// Later stages need the old floor to bound live locations and start the floor scan.
/// They also need the old tree's append nodes to compute the child root. Authenticate
/// those inputs here, and return one set of root components against which membership,
/// bitmap, and candidate proofs can be checked.
///
/// For size one, require the deterministic initial commitment and synthesize its
/// single active commit bit. Otherwise bind the supplied raw root through the
/// Current witness, check which pending and partial chunk components the log size
/// requires, and prove that `CommitFloor(metadata, floor)` occupies `size - 1`.
/// Check the pinned nodes from just before the terminal with that inclusion proof.
/// Replaying its append recovers the authenticated nodes needed to extend the current log.
/// The caller has validated the chunk profile and trusts the checkpoint's history.
pub(crate) fn authenticate_pre_state<F, K, V, H, const N: usize>(
    commitment: CurrentCommitment<F, K, V, H, N>,
    claim: &PreStateOpeningClaim<F, V, H::Digest, N>,
) -> Result<AuthenticatedPreState<F, K, V, H, N>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    let size = commitment.size();
    if size == Location::new(1) {
        if commitment != initial_commitment::<F, K, V, H, N>()
            || !matches!(claim, PreStateOpeningClaim::Initial)
        {
            return Err(VerifyError::InvalidPreStateOpening);
        }
        let terminal = Operation::CommitFloor(None, Location::new(0));
        let ops_root = crate::qmdb::any::initial_root::<F, update::Ordered<K, V>, H>();
        let hasher = qmdb::hasher::<H>();
        let encoded = terminal.encode();
        let leaf_pos = Position::try_from(Location::<F>::new(0))
            .map_err(|_| VerifyError::InvalidPreStateOpening)?;
        let mut chunk = BitMap::<N>::EMPTY_CHUNK;
        set_chunk_bit(&mut chunk, 0, true)?;
        let partial = H::hash(&[chunk.as_slice()]);
        let witness = crate::qmdb::current::proof::OpsRootWitness {
            grafted_root: ops_root,
            pending_chunk_digest: None
                .try_into()
                .map_err(|_| VerifyError::InvalidPreStateOpening)?,
            partial_chunk: Some((1, partial)),
        };
        return Ok(AuthenticatedPreState {
            commitment,
            ops_root,
            old_floor: Location::new(0),
            terminal_location: Location::new(0),
            witness,
            raw_frontier: BTreeMap::from([(leaf_pos, hasher.leaf_digest(leaf_pos, &encoded))]),
            grafted_inactive_peaks: 0,
            initial: true,
        });
    }

    let PreStateOpeningClaim::Proven {
        ops_root,
        witness,
        terminal,
    } = claim
    else {
        return Err(VerifyError::InvalidPreStateOpening);
    };
    if !witness.verify::<H>(ops_root, commitment.root()) {
        return Err(VerifyError::InvalidPreStateOpening);
    }
    let chunk_bits = BitMap::<N>::CHUNK_SIZE_BITS;
    let complete = size.as_u64() / chunk_bits;
    let next_bit = size.as_u64() % chunk_bits;
    let graftable = grafting::graftable_chunks::<F>(*size, grafting::height::<N>()).min(complete);
    if witness.pending_chunk_digest.as_ref().is_some() != (complete > graftable) {
        return Err(VerifyError::InvalidPreStateOpening);
    }
    match (&witness.partial_chunk, next_bit) {
        (None, 0) => {}
        (Some((claimed, _)), expected) if *claimed == expected => {}
        _ => return Err(VerifyError::InvalidPreStateOpening),
    }
    let terminal_location = size.checked_sub(1).ok_or(VerifyError::Overflow)?;
    if terminal.floor > terminal_location
        || terminal.raw_proof.leaves != size
        || terminal.raw_proof.inactive_peaks != F::inactive_peaks(size, terminal.floor)
    {
        return Err(VerifyError::InvalidPreStateOpening);
    }
    let terminal_operation = Operation::CommitFloor(terminal.metadata.clone(), terminal.floor);
    let grafted_inactive_peaks =
        grafting::chunk_aligned_inactive_peaks::<F>(size, terminal.floor, grafting::height::<N>())
            .map_err(|_| VerifyError::InvalidPreStateOpening)?;
    if !qmdb::verify_proof_and_pinned_nodes::<H, F, _>(
        &terminal.raw_proof,
        terminal_location,
        core::slice::from_ref(&terminal_operation),
        &terminal.pins_before_terminal,
        ops_root,
    ) {
        return Err(VerifyError::InvalidPreStateOpening);
    }
    let nodes = F::nodes_to_pin(terminal_location)
        .zip(terminal.pins_before_terminal.iter().copied())
        .collect::<BTreeMap<_, _>>();
    let mut raw = SparseForest::new(terminal_location, nodes);
    let hasher = qmdb::hasher::<H>();
    let encoded = terminal_operation.encode();
    let leaf_position =
        Position::try_from(terminal_location).map_err(|_| VerifyError::InvalidPreStateOpening)?;
    raw.append_digest(
        hasher.leaf_digest(leaf_position, &encoded),
        |forest, position, height| forest.parent_digest(&hasher, position, height),
    )?;
    // The pinned proof already authenticated the terminal append and its resulting root.
    Ok(AuthenticatedPreState {
        commitment,
        ops_root: *ops_root,
        old_floor: terminal.floor,
        terminal_location,
        witness: witness.clone(),
        raw_frontier: raw.into_nodes(),
        grafted_inactive_peaks,
        initial: false,
    })
}

/// Authenticate an active operation against this transition's pre-state opening.
///
/// Member and exclusion claims use this check before adding a record's fields to
/// the trusted cache. Inclusion alone would allow superseded records. Requiring a
/// set activity bit establishes that the record is live at the trusted commitment.
///
/// Require the opening's size, raw root, trailing bitmap digests, and floor-derived
/// inactive-peak count. The operation proof checks inclusion and a set activity bit.
/// Its reconstructed grafted root must also equal the opening's grafted root. Callers
/// separately restrict the operation location and payload to the kind of fact needed.
pub(crate) fn operation_proof_matches_pre_state<F, K, V, H, const N: usize>(
    pre: &AuthenticatedPreState<F, K, V, H, N>,
    proof: &OperationProof<F, H::Digest, N>,
    operation: &Operation<F, update::Ordered<K, V>>,
) -> bool
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    let range = &proof.range_proof;
    let partial_digest = pre
        .witness
        .partial_chunk
        .as_ref()
        .map(|(_, digest)| *digest);
    range.proof.leaves == pre.commitment.size()
        && range.ops_root == pre.ops_root
        && range.pending_chunk_digest == pre.witness.pending_chunk_digest
        && range.partial_chunk_digest == partial_digest
        && range.proof.inactive_peaks == pre.grafted_inactive_peaks
        && proof.verify_and_reconstruct_grafted_root::<H, _>(operation, pre.commitment.root())
            == Some(pre.witness.grafted_root)
}
