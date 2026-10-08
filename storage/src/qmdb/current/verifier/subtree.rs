//! Authenticate subtree digests against their positions in a Merkle tree.
//!
//! Each target names a complete subtree and its claimed digest. Together with the
//! proof's sibling and peak digests, these targets form the frontier from which the
//! verifier reconstructs the root. Matching that root authenticates the targets and
//! the nodes later used to update the tree.

use super::{BitmapFault, VerifyError};
use crate::merkle::{self, Family, Location, Position, hasher::Hasher};
use alloc::{collections::BTreeMap, vec::Vec};
use commonware_cryptography::Digest;

/// A claimed digest for the perfect subtree of `2^height` leaves at `leaf_start`.
/// Bitmap coverage supplies aligned, ordered, non-overlapping targets within the tree;
/// the inclusion verifier must still authenticate each digest against its root.
#[derive(Clone, Copy)]
pub(crate) struct SubtreeTarget<F: Family, D: Digest> {
    pub(crate) leaf_start: Location<F>,
    pub(crate) height: u32,
    pub(crate) digest: D,
}

type Parent<F> = (Position<F>, Position<F>, Position<F>);

/// Read the missing frontier nodes in the exact order consumed by verification.
/// Reusing the same subtree plan binds witness placement to the target geometry.
#[cfg(feature = "std")]
pub(super) async fn create_multi_subtree_proof<F: Family, D: Digest>(
    storage: &impl merkle::storage::Storage<F, Digest = D>,
    inactive_peaks: usize,
    targets: &[SubtreeTarget<F, D>],
) -> Result<merkle::Proof<F, D>, merkle::Error<F>> {
    let leaves = Location::try_from(storage.size())?;
    let mut frontier = BTreeMap::new();
    let mut parents = Vec::new();
    let mut start = 0;
    let mut remaining = targets;
    for (position, height) in F::peaks(storage.size()) {
        let end = start + (1u64 << height);
        let split = remaining.partition_point(|target| target.leaf_start.as_u64() < end);
        let (peak_targets, rest) = remaining.split_at(split);
        plan(
            position,
            height,
            start,
            peak_targets,
            &mut frontier,
            &mut parents,
        )
        .map_err(|_| merkle::Error::InvalidProof)?;
        remaining = rest;
        start = end;
    }
    let mut digests = Vec::new();
    for (position, target) in frontier {
        if target.is_none() {
            digests.push(
                storage
                    .get_node(position)
                    .await?
                    .ok_or(merkle::Error::ElementPruned(position))?,
            );
        }
    }
    Ok(merkle::Proof {
        leaves,
        inactive_peaks,
        digests,
    })
}

/// Partition one peak into supplied targets and missing witness subtrees.
///
/// The bitmap proof provides subtree digests instead of individual leaves. Build a
/// reconstruction plan that places each target and witness at its required position,
/// so root verification authenticates exactly those subtrees. Shared paths appear
/// once in the plan, allowing reconstruction to reuse their digests.
///
/// Stop at an exact target or at a subtree containing no targets. Otherwise recurse
/// through both children and record their parent after them, so reconstruction hashes
/// children first. Reject targets crossing a child boundary or descending past a leaf.
/// The caller supplies sorted, disjoint, aligned targets bounded by the full tree.
fn plan<F: Family, D: Digest>(
    position: Position<F>,
    height: u32,
    start: u64,
    targets: &[SubtreeTarget<F, D>],
    frontier: &mut BTreeMap<Position<F>, Option<D>>,
    parents: &mut Vec<Parent<F>>,
) -> Result<(), VerifyError> {
    match targets {
        [] => {
            frontier.insert(position, None);
        }
        [target] if target.leaf_start.as_u64() == start && target.height == height => {
            frontier.insert(position, Some(target.digest));
        }
        _ => {
            let child_height = height.checked_sub(1).ok_or(BitmapFault::Geometry)?;
            let midpoint = start + (1u64 << child_height);
            let split = targets.partition_point(|target| target.leaf_start.as_u64() < midpoint);
            let (left_targets, right_targets) = targets.split_at(split);
            let crosses = left_targets.last().is_some_and(|last| {
                1u64.checked_shl(last.height)
                    .and_then(|width| last.leaf_start.as_u64().checked_add(width))
                    .is_none_or(|end| end > midpoint)
            });
            if crosses {
                return Err(BitmapFault::Geometry.into());
            }
            let (left, right) = F::children(position, height);
            plan(left, child_height, start, left_targets, frontier, parents)?;
            plan(
                right,
                child_height,
                midpoint,
                right_targets,
                frontier,
                parents,
            )?;
            parents.push((position, left, right));
        }
    }
    Ok(())
}

/// Bind pre-hashed targets to their positions by reconstructing every peak and the root.
///
/// Bitmap verification calls this for both the raw and grafted trees. Authenticating
/// an explicit chunk's raw digest and its bitmap-adjusted digest binds the chunk bytes.
/// Authenticating a zero subtree's shared digest under both roots establishes its zero
/// bits under Current's hash assumptions. The returned nodes also supply the siblings
/// needed to recompute changed paths without loading the rest of the database.
///
/// Plan each peak, fill missing frontier nodes from `proof.digests` in position order,
/// and reject a digest count mismatch or a target past the last peak. Hash parents in
/// postorder and combine peaks with the proof's size and inactive-peak count. Return the
/// authenticated nodes only if the reconstructed root matches, allowing later sparse
/// updates to reuse siblings; otherwise report `mismatch`, which names the tree.
/// Bitmap verification must already have checked target ordering, alignment, coverage,
/// and bounds, and bound the proof size and inactive peaks to the trusted pre-state.
pub(crate) fn verify_multi_subtree_inclusion<F: Family, H: Hasher<F>>(
    proof: &merkle::Proof<F, H::Digest>,
    hasher: &H,
    targets: &[SubtreeTarget<F, H::Digest>],
    root: &H::Digest,
    mismatch: BitmapFault,
) -> Result<BTreeMap<Position<F>, H::Digest>, VerifyError> {
    let peaks = F::peaks(F::location_to_position(proof.leaves)).collect::<Vec<_>>();
    let mut frontier = BTreeMap::new();
    let mut parents = Vec::new();
    let mut start = 0;
    let mut remaining = targets;
    for &(position, height) in &peaks {
        let end = start + (1u64 << height);
        let split = remaining.partition_point(|target| target.leaf_start.as_u64() < end);
        let (peak_targets, rest) = remaining.split_at(split);
        plan(
            position,
            height,
            start,
            peak_targets,
            &mut frontier,
            &mut parents,
        )?;
        remaining = rest;
        start = end;
    }
    if !remaining.is_empty() {
        return Err(BitmapFault::Geometry.into());
    }
    let mut witnesses = proof.digests.iter().copied();
    let mut nodes = BTreeMap::new();
    for (position, target) in frontier {
        let digest = target
            .or_else(|| witnesses.next())
            .ok_or(BitmapFault::ProofShape)?;
        nodes.insert(position, digest);
    }
    if witnesses.next().is_some() {
        return Err(BitmapFault::ProofShape.into());
    }
    // The plan records every parent after both of its children.
    for (parent, left, right) in parents {
        let (Some(left), Some(right)) = (nodes.get(&left), nodes.get(&right)) else {
            return Err(BitmapFault::Geometry.into());
        };
        let digest = hasher.node_digest(parent, left, right);
        nodes.insert(parent, digest);
    }
    let peak_digests = peaks
        .iter()
        .map(|(position, _)| nodes.get(position))
        .collect::<Option<Vec<_>>>()
        .ok_or(BitmapFault::Geometry)?;
    let reconstructed = hasher
        .root(proof.leaves, proof.inactive_peaks, peak_digests)
        .map_err(|_| BitmapFault::ProofShape)?;
    if reconstructed != *root {
        return Err(mismatch.into());
    }
    Ok(nodes)
}
