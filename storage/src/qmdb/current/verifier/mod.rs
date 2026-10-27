//! Verify an ordered Current-QMDB commit without opening a database.
//!
//! [`UserBatch::verify`] derives the physical operations, activity bitmap, inactivity
//! floor, and root for the caller's [`UserBatch`] mutations and metadata. A
//! [`BatchProofBundle`] supplies old records, Merkle proofs, and claimed transition
//! bounds. The verifier authenticates those records, constructs the operations to
//! append, and checks the claimed bounds against the resulting state.
//! The result is a [`Verified`] transition, returned only if the derived commitment
//! equals the caller's claimed post-state. [`Verified::into_cache`] turns it into a
//! [`TrustedMap`] bound to that commitment for the next batch.
//!
//! A storage-backed writer uses `UserBatch::prove` to compute the proposed post-state
//! through the ordinary database writer and, from that merkleized batch, complete evidence
//! for an empty client cache. `UserBatch::build_batch_proof` returns only the evidence, and
//! `UserBatch::merkleize` only the batch; used apart, both must use the same database
//! checkpoint. The client independently calls [`UserBatch::verify`] before accepting that
//! state. Proof production never changes the database.
//!
//! The commitment does not record how far a batch advanced the inactivity floor, so the
//! verifier and the writer must use the same [`Policy`](crate::qmdb::floor::Policy). The
//! verifier replays policies that never evict and that either hold the floor or limit only
//! entries, such as `Proportional`, `Hold` and `Bounded` with unlimited skips; it rejects
//! the rest with [`VerifyError::UnsupportedFloorPolicy`].
//!
//! # Execution and features
//!
//! Both verifier entry points are synchronous and perform no database I/O.
//! [`UserBatch::verify`] runs sequentially. [`UserBatch::verify_with_strategy`]
//! accepts a `commonware_parallel::Rayon` strategy to authenticate independent
//! proofs and hash operations on a CPU pool. Cache updates, floor replay, and
//! appends retain their deterministic order. Both paths check the same evidence.
//!
//! Disable this crate's default features to use the verifier with `core` and `alloc`,
//! without the databases, prover, or storage runtime. For Rayon, enable the `std`
//! feature on `commonware-parallel` and pass its Rayon strategy. This does not enable
//! this crate's `std` feature.
//!
//! This crate's default `std` feature retains the full storage API, including the
//! database and prover. Cargo combines feature requests: another dependency enabling
//! `commonware-storage/std` also enables storage in that build. The verifier API and
//! implementation are the same in both configurations.
//!
//! # Wire format
//!
//! [`CurrentCommitment`] and [`BatchProofBundle`] implement the codec traits, so a writer
//! can send a bundle and its claimed post-commitment as bytes. Decode a bundle with a
//! [`BundleConfig`] whose limits bound every length in it: decoding allocates no more than
//! they allow and checks only the encoding, which leaves every claim to verification.
//!
//! # Trust boundary
//!
//! The caller must trust the pre-commitment as a valid ordered Current database and
//! must authorize the batch and its metadata. Verification does not establish who
//! created the checkpoint, authorize writes, or audit the database's earlier history.
//! Soundness relies on the hash function and the existing Merkle proof primitives.
//! Start with [`TrustedMap::empty_cache`] when no previously authenticated facts are
//! available. Cache constructors and mutators accept facts the caller already trusts.
//! Their consistency checks do not authenticate those facts against a commitment.
//! Proof-bundle fields are untrusted inputs to the checks described below.
//!
//! # What is committed
//!
//! The raw operations tree commits to the append-only physical log. The activity
//! bitmap identifies its live updates and terminal commit. Each chunk contains `N`
//! bytes and covers `8 * N` log locations. At tree height `log2(8 * N)`,
//! a complete graftable chunk replaces the corresponding raw subtree digest `r`
//! with `hash(chunk || r)`, except that an all-zero chunk leaves `r` unchanged.
//! Ancestors use the family's normal positional hashing. A complete chunk awaiting
//! its subtree in MMB is pending. An incomplete chunk is partial. Their digests are
//! bound separately alongside the raw and grafted roots by Current's root combiner.
//! The final log operation authenticates the inactivity floor and commit metadata.
//!
//! # Verification argument
//!
//! 1. Check the supplied raw root and bitmap components against the trusted root.
//!    Authenticate the final `CommitFloor` and the raw append frontier (the nodes
//!    needed to append to the tree), binding the old floor and log size.
//! 2. Authenticate active ordered records and exclusion intervals against that same
//!    opening. Merge only consistent facts into the cache. An active record's cyclic
//!    successor proves that the open interval between the two keys contains no active key.
//! 3. Classify each requested mutation and derive the application operations,
//!    including predecessor rewrites. The batch keeps delete ranges normalized;
//!    follow authenticated successor edges through each interval to enumerate every
//!    deleted member. A range with no lower bound starts at the least key, which the
//!    greatest key's wrapping successor edge proves. Missing facts cause an error,
//!    never a guess.
//! 4. Authenticate every bitmap region needed for invalidations, appends, and the
//!    floor scan. Exact coverage prevents omitted live bits. Raw and grafted proofs
//!    jointly authenticate explicit chunks and compressed all-zero intervals.
//! 5. Require candidate payloads at every set bit in the scan, except the separately
//!    authenticated terminal. Replay the caller's floor policy to derive the floor and
//!    the copies, and require the claimed scan endpoint to match the replay's stopping
//!    point.
//! 6. Append the derived operations, clear superseded bits, and recompute affected
//!    Merkle paths using authenticated siblings. Combine all child root components
//!    and compare both size and root with the claimed post-commitment.
//! 7. On [`Verified::into_cache`], rebase retained facts through those same
//!    operations. Remove invalidated absence intervals and replace moved records
//!    before returning the cache.
//!
//! Each private stage documents the facts it consumes and establishes. Authenticated
//! intermediate values must stay within the transition that produced them.
//!
//! # Complexity
//!
//! These are conservative upper bounds for MMR and MMB, counting total CPU work,
//! not parallel elapsed time. Keys, values, digests, and `N` are fixed-size in this
//! model; comparisons, clones, encoding, and hashing one such item cost O(1).
//! For variable-size values, additionally count the bytes cloned, encoded, decoded,
//! and hashed, including repeated processing of the same value. Memory likewise
//! includes the bytes of retained and temporary payloads. Backend I/O latency,
//! decompression, and custom index/strategy costs are separate.
//!
//! - `n`: pre-state log operations; `m`: active database keys.
//! - `b`: normalized batch operations; `r`: old members enumerated by range deletes.
//!   Write `k = b + r + 1`. Application operations are O(k). Floor copies are bounded by
//!   the policy's entry limit, which is O(k) for `Proportional`; the bounds below assume
//!   a policy whose entry limit is O(k).
//! - `c`: entries plus explicit spans in the incoming trusted cache.
//! - `p`: supplied member/exclusion claims; `f`: supplied floor candidates.
//! - `u`: supplied bitmap coverage units (explicit chunks or zero subtrees).
//! - `h`: maximum Merkle height during the transition, O(log(n + k + 2)).
//!   Write `t = (p + f + u + k + 1) * h` as an upper bound on Merkle-path work
//!   and temporary node storage. Shared paths can make actual storage much smaller.
//!
//! Verification has worst-case CPU cost
//! `O((p + k + f) * log(c + p + k + f + 2) + t * log(t + 2))`
//! and peak memory `O(c + k + p + f + u + t)`, including the owned working cache
//! and result, but excluding the borrowed batch and proof bundle. Cloning the cache
//! before calling adds O(c) work and memory. There is no database I/O. The logarithmic
//! factors account for the implementation's B-tree maps/sets; counting only hashes
//! would omit this work. Supplied claims and candidates are independent input sizes:
//! an oversized or ultimately rejected bundle can cost more than an honest proof.
//! There is no worst-case bound in `b` and `n` alone for arbitrary supplied bundles.
//!
//! No distribution-independent average case exists. For a typical honest workload
//! with `p`, `f`, and `u` all O(k), the bound becomes
//! `O(k * log(c + k + 2) + k * h * log(k * h + 2))` CPU and `O(c + k * h)` memory.
//! A retained cache can reduce `p`; it does not remove bitmap authentication or floor
//! replay. Range deletes still cost at least O(r), even when `b` is one.
//!
//! Child-root reconstruction hashes shared ancestors of changed bitmap chunks once.
//! For `d > 0` changed chunks among `L` graftable chunks, this stage visits
//! O(d * (1 + log(L / d)) + h) nodes, with logarithmic B-tree access costs and
//! O(d + h) temporary space beyond the node maps. Updating every chunk takes O(L + h)
//! visits. The maps are consumed directly. Authentication of the supplied proofs
//! remains separate, so this saving does not change the overall upper bounds above.
//! Floor-candidate authentication shares both witness planning and hashing. For
//! `f > 0` candidates its tree traversal visits O(f * (1 + log(n / f)) + h) nodes;
//! sorting and witness-position searches add logarithmic factors. Individual key
//! proofs still authenticate their own paths, so the overall upper bounds above
//! remain conservative. See [`crate::merkle::Proof::verify_multi_inclusion`].
//!
//! The bundled prover supplies complete key proofs and proves each run of clear bitmap
//! chunks with the fewest aligned zero subtrees. The floor-copy policy's entry limit
//! bounds the live records the scan reaches, not the inactive bits it crosses, but those
//! bits cost O(log n) units per run. Thus `p`, `f`, and `u` are O(k) up to that logarithmic
//! factor, and an honest bundle does not grow with the old log for a small batch. The
//! prover still compares every scanned chunk in memory. See `UserBatch::prove` for database
//! lookup, collision, and I/O costs in addition to these verification bounds.

// Bundles are untrusted input: report a malformed one as an error, never a panic.
#![cfg_attr(
    not(test),
    deny(
        clippy::as_conversions,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::unreachable,
        clippy::unwrap_used
    )
)]

#[cfg(test)]
#[cfg(feature = "std")]
mod attack_tests;
mod bitmap;
#[cfg(feature = "std")]
mod builder;
mod bundle;
mod codec;
mod commitment;
mod error;
mod floor;
mod opening;
mod ordered;
mod root;
mod sparse;
mod subtree;
#[cfg(test)]
#[cfg(feature = "std")]
mod tests;
mod transition;
mod trusted;
mod user;

pub use bitmap::{BitmapProof, BitmapUnit, ProvenBitmapProof};
pub use bundle::{
    BatchProofBundle, CandidateOperation, ClaimedBatchBounds, PreStateCandidateProof,
    PreStateExclusionClaim, PreStateKeyClaim, PreStateMemberClaim, PreStateOpeningClaim,
    PreStateTerminalClaim,
};
pub use codec::{BundleConfig, BundleConfigFor};
pub use commitment::{CurrentCommitment, initial_commitment};
#[cfg(feature = "std")]
pub use error::BuildError;
pub use error::{BitmapFault, CandidateFault, InsufficientPreState, VerifyError};
pub use transition::Verified;
pub use trusted::{
    OptionTrusted, PartialTrustedState, TrustedAbsenceMap, TrustedEntry, TrustedMap, TrustedState,
};
pub use user::{UserBatch, UserBatchIntoIter, UserBatchIter, UserOp};
