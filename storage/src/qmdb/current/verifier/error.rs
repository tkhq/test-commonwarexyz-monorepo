//! Verifier errors.

use thiserror::Error;

/// Failure to construct a transition proof from a committed database.
#[derive(Debug, Error)]
#[cfg(feature = "std")]
pub enum BuildError<F: crate::merkle::Family> {
    /// Storage could not supply the required records or Merkle nodes.
    #[error("database proof construction failed: {0}")]
    Database(#[from] crate::qmdb::Error<F>),
    /// The database did not satisfy the committed-state contract.
    #[error("invalid committed database state: {0}")]
    InvalidState(&'static str),
    /// The floor policy evicts or limits skips, which a verifier cannot replay.
    #[error("unsupported floor policy")]
    UnsupportedFloorPolicy,
    /// Generated evidence failed the verifier's checks.
    #[error("generated proof is inconsistent: {0}")]
    Inconsistent(#[from] VerifyError),
}

#[cfg(feature = "std")]
impl<F: crate::merkle::Family> From<crate::merkle::Error<F>> for BuildError<F> {
    fn from(error: crate::merkle::Error<F>) -> Self {
        Self::Database(crate::qmdb::Error::Merkle(error))
    }
}

/// A pre-state fact needed to derive the ordered transition has not been established.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum InsufficientPreState {
    /// Membership or absence of a mutated key was not established.
    #[error("missing key classification")]
    Classification,
    /// An ordered predecessor or successor edge was not established.
    #[error("missing ordered edge")]
    OrderedEdge,
    /// An active operation location was not established.
    #[error("missing active location")]
    Location,
    /// An active value was not established.
    #[error("missing active value")]
    Value,
    /// An active successor key was not established.
    #[error("missing successor key")]
    NextKey,
}

/// The check a bitmap proof failed.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum BitmapFault {
    /// Only the initial state takes an initial proof, and every other state a proven one.
    #[error("wrong proof kind")]
    Kind,
    /// The claimed floor scan ends below the old floor or past the log.
    #[error("scan end out of bounds")]
    ScanBounds,
    /// The proof claims a pre-state size other than the commitment's.
    #[error("pre-state size mismatch")]
    PreSize,
    /// The units do not partition the required chunk intervals: a unit is missing, repeated,
    /// out of order, extra, or past the log, or a zero subtree runs past its interval.
    #[error("coverage mismatch")]
    Coverage,
    /// A chunk carries a raw subtree digest exactly when it is not graftable.
    #[error("raw subtree digest presence mismatch")]
    RawDigest,
    /// The partial chunk sets bits past the log or differs from its committed digest.
    #[error("partial chunk mismatch")]
    PartialChunk,
    /// The pending chunk differs from its committed digest.
    #[error("pending chunk mismatch")]
    PendingChunk,
    /// A zero subtree does not start at a multiple of its width.
    #[error("misaligned zero subtree")]
    Misaligned,
    /// A zero subtree reaches past the graftable chunks.
    #[error("zero subtree past the graftable chunks")]
    Ungraftable,
    /// A zero subtree covers a chunk whose bytes the transition reads or changes.
    #[error("zero subtree over a chunk that needs its bytes")]
    NeedsBytes,
    /// A raw or grafted proof is missing, unexpected, bound to another size or floor, or has
    /// the wrong number of digests.
    #[error("proof shape mismatch")]
    ProofShape,
    /// A target is not a node of the tree at this size.
    #[error("target is not a tree node")]
    Geometry,
    /// The raw proof does not reconstruct the operations root.
    #[error("raw root mismatch")]
    RawRoot,
    /// The grafted proof does not reconstruct the grafted root.
    #[error("grafted root mismatch")]
    GraftedRoot,
    /// The grafted append frontier has the wrong shape or does not reconstruct the grafted root.
    #[error("grafted append frontier mismatch")]
    Frontier,
    /// Two authenticated sources disagree about the digest of one node.
    #[error("conflicting nodes")]
    NodeConflict,
    /// The old terminal or a superseded record is not active, or its chunk is not covered.
    #[error("inactive or uncovered record")]
    Inactive,
    /// The claimed scan end differs from where floor replay stopped.
    #[error("scan end differs from floor replay")]
    ScanEnd,
}

/// The check the floor candidates failed.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum CandidateFault {
    /// The candidate locations differ from the set bits in the floor scan: one is missing,
    /// repeated, out of order, or extra.
    #[error("locations differ from the scanned bits")]
    Locations,
    /// A candidate payload is not an update.
    #[error("payload is not an update")]
    NotUpdate,
    /// The candidates' raw proof is missing, unexpected, bound to another size or floor, or
    /// does not authenticate their payloads.
    #[error("payload proof mismatch")]
    Proof,
    /// A batch that empties the database claims a floor scan or candidates.
    #[error("scan claimed for an emptied database")]
    EmptiedScan,
    /// Floor replay received candidates out of ascending order.
    #[error("candidates out of order")]
    Unordered,
}

/// Errors returned while verifying a claimed state transition.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum VerifyError {
    /// A typed commitment or proof field has invalid bounds, or a delete range is reversed.
    #[error("invalid bounds")]
    InvalidBounds,
    /// The trusted cache is not bound to the supplied pre-commitment.
    #[error("trusted cache commitment mismatch")]
    TrustedCommitmentMismatch,
    /// Cached facts conflict or do not fit the commitment's size or authenticated floor.
    #[error("invalid trusted state")]
    InvalidTrustedState,
    /// The pre-state opening does not authenticate the trusted commitment.
    #[error("invalid pre-state opening")]
    InvalidPreStateOpening,
    /// A membership claim fails authentication or disagrees with the pre-state opening.
    #[error("invalid member proof")]
    InvalidMemberProof,
    /// An exclusion claim fails authentication or disagrees with the pre-state opening.
    #[error("invalid exclusion proof")]
    InvalidExclusionProof,
    /// Pre-state claims repeat a queried key or conflict with retained facts.
    #[error("conflicting pre-state facts")]
    ConflictingPreState,
    /// The transition cannot be derived from the available trusted facts.
    #[error("insufficient pre-state: {0}")]
    InsufficientPreState(#[from] InsufficientPreState),
    /// Bitmap coverage or authentication is invalid.
    #[error("invalid bitmap proof: {0}")]
    InvalidBitmapProof(#[from] BitmapFault),
    /// Floor candidates fail authentication or conflict with the scan.
    #[error("invalid floor candidates: {0}")]
    InvalidCandidates(#[from] CandidateFault),
    /// The floor policy evicts or limits skips, which replay cannot check from the scan alone.
    #[error("unsupported floor policy")]
    UnsupportedFloorPolicy,
    /// A sparse Merkle update lacks an authenticated input or has invalid geometry.
    #[error("invalid sparse update")]
    InvalidSparseUpdate,
    /// Checked size, location, or height arithmetic overflowed.
    #[error("arithmetic overflow")]
    Overflow,
    /// Bundle bounds disagree with authenticated pre-state or locally derived post-state bounds.
    #[error("claimed batch bounds mismatch")]
    ClaimedBoundsMismatch,
    /// The derived child commitment differs from the claimed commitment.
    #[error("post-commitment mismatch")]
    PostCommitmentMismatch,
}
