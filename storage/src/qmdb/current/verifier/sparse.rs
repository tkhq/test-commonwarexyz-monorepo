//! Recompute Merkle roots using only the nodes needed for appends and bitmap changes.

use super::VerifyError;
use crate::merkle::{Family, Location, Position, hasher::Hasher};
use alloc::{collections::BTreeMap, vec::Vec};
use commonware_cryptography::Digest;

/// Partial operation-space Merkle forest used to recompute affected paths.
/// Stored nodes are authenticated old digests or locally derived replacements. Missing
/// nodes are unknown. The caller must authenticate supplied nodes against the same
/// pre-state before using this forest to derive a child root.
pub(crate) struct SparseForest<F: Family, D: Digest> {
    pub(super) leaves: Location<F>,
    nodes: BTreeMap<Position<F>, D>,
    append_start: Position<F>,
    appended: Vec<D>,
}

impl<F: Family, D: Digest> SparseForest<F, D> {
    /// Start a sparse forest from authenticated nodes and a valid leaf count.
    /// This stores the partial tree used by terminal replay or child-root reconstruction.
    /// It checks no proof. The caller must take the count from a checked commitment
    /// (or its pre-terminal prefix). Nodes come from verified proofs or the fixed
    /// initial state.
    pub(crate) fn new(leaves: Location<F>, nodes: BTreeMap<Position<F>, D>) -> Self {
        Self {
            leaves,
            nodes,
            append_start: F::location_to_position(leaves),
            appended: Vec::new(),
        }
    }

    /// Append one already position-hashed leaf and build the family's newly formed parents.
    /// Terminal replay uses this to recover the old append frontier. Child-root
    /// reconstruction uses it to append each derived operation to the raw and grafted
    /// trees. The result retains every unchanged subtree and hashes the new append path.
    ///
    /// Compute each new parent through `parent_digest`. The raw tree hashes its children.
    /// The grafted tree can reuse newly computed raw nodes below the grafting height,
    /// where both trees are identical, then incorporates final bitmap bytes at that
    /// height and hashes grafted children above it. Existing siblings remain unchanged.
    /// The caller supplies the correct leaf digest for the current append position.
    /// Both supported families append consecutive postorder positions, so new nodes
    /// use a dense vector with O(1) insertion and lookup. Old authenticated siblings
    /// remain in the sparse map. Update old subtrees before starting appends.
    /// On failure the forest may be partially modified and must be discarded.
    pub(crate) fn append_digest<T>(
        &mut self,
        leaf_digest: D,
        mut parent_digest: T,
    ) -> Result<(), VerifyError>
    where
        T: FnMut(&Self, Position<F>, u32) -> Result<D, VerifyError>,
    {
        let leaf_position =
            Position::try_from(self.leaves).map_err(|_| VerifyError::InvalidSparseUpdate)?;
        debug_assert_eq!(
            Some(leaf_position.as_u64()),
            u64::try_from(self.appended.len())
                .ok()
                .and_then(|appended| self.append_start.as_u64().checked_add(appended))
        );
        self.appended.push(leaf_digest);
        let mut next_position = leaf_position.checked_add(1).ok_or(VerifyError::Overflow)?;
        for height in F::parent_heights(self.leaves) {
            let digest = parent_digest(self, next_position, height)?;
            self.appended.push(digest);
            next_position = next_position.checked_add(1).ok_or(VerifyError::Overflow)?;
        }
        self.leaves = self.leaves.checked_add(1).ok_or(VerifyError::Overflow)?;
        Ok(())
    }

    /// Hash a parent's authenticated or locally derived children at its exact position.
    /// Append callbacks use this wherever the raw and grafted trees need their own hash.
    pub(crate) fn parent_digest<H: Hasher<F, Digest = D>>(
        &self,
        hasher: &H,
        position: Position<F>,
        height: u32,
    ) -> Result<D, VerifyError> {
        let (left, right) = F::children(position, height);
        Ok(hasher.node_digest(position, &self.node(left)?, &self.node(right)?))
    }

    /// Replace sorted, disjoint subtrees of one height and rebuild their shared paths once.
    /// Child-root reconstruction supplies the final digest for each changed bitmap chunk.
    /// Recursion descends only into branches containing changes; untouched siblings must
    /// already be authenticated or derived. Missing siblings reject the update.
    ///
    /// The caller must supply strictly increasing, aligned starts for existing subtrees
    /// at `height`. These are locally derived changes, not prover-selected replacements.
    /// Stored descendants below a replaced subtree are not refreshed; subsequent reads
    /// must use that subtree root or its ancestors. Discard the forest on failure.
    /// With `d > 0` changed chunks among `L` possible chunks and tree height `h`, at most
    /// O(d * (1 + log(L / d)) + h) nodes are visited, rather than O(d * h) separate
    /// path steps. B-tree accesses add a logarithmic factor in stored node count.
    /// Recursion uses O(h) stack space; the caller stores O(d) changes.
    pub(crate) fn update_subtrees<H: Hasher<F, Digest = D>>(
        &mut self,
        hasher: &H,
        height: u32,
        mut changes: &[(Location<F>, D)],
    ) -> Result<(), VerifyError> {
        debug_assert!(
            self.appended.is_empty(),
            "update old subtrees before appending"
        );
        let mut start = 0;
        for (position, level) in F::peaks(F::location_to_position(self.leaves)) {
            if changes.is_empty() {
                break;
            }
            let end = start + (1u64 << level);
            let split = changes.partition_point(|(location, _)| **location < end);
            let (peak_changes, rest) = changes.split_at(split);
            if !peak_changes.is_empty() {
                self.rebuild(hasher, position, level, start, height, peak_changes)?;
            }
            changes = rest;
            start = end;
        }
        Ok(())
    }

    /// Hash a changed branch after both children have their final digests.
    /// Slice partitioning assigns each replacement to exactly one child. Returning
    /// child digests directly avoids looking them up again in the sparse node map.
    fn rebuild<H: Hasher<F, Digest = D>>(
        &mut self,
        hasher: &H,
        position: Position<F>,
        level: u32,
        start: u64,
        height: u32,
        changes: &[(Location<F>, D)],
    ) -> Result<D, VerifyError> {
        let digest = match changes {
            [] => return self.node(position),
            [(_, digest)] if level == height => *digest,
            _ => {
                // Disjoint changes at `height` cannot share a subtree of that height.
                let child_level = level
                    .checked_sub(1)
                    .filter(|child_level| *child_level >= height)
                    .ok_or(VerifyError::InvalidSparseUpdate)?;
                let midpoint = start + (1u64 << child_level);
                let split = changes.partition_point(|(location, _)| **location < midpoint);
                let (left_changes, right_changes) = changes.split_at(split);
                let (left, right) = F::children(position, level);
                let left = self.rebuild(hasher, left, child_level, start, height, left_changes)?;
                let right =
                    self.rebuild(hasher, right, child_level, midpoint, height, right_changes)?;
                hasher.node_digest(position, &left, &right)
            }
        };
        self.nodes.insert(position, digest);
        Ok(digest)
    }

    /// Read a known positioned digest, rejecting missing proof material.
    /// Appends and path updates use this for their child and sibling inputs. A missing
    /// node prevents root reconstruction because its digest has not been established.
    pub(super) fn node(&self, position: Position<F>) -> Result<D, VerifyError> {
        if position >= self.append_start {
            return self
                .appended
                .get(
                    usize::try_from(position.as_u64() - self.append_start.as_u64())
                        .map_err(|_| VerifyError::InvalidSparseUpdate)?,
                )
                .copied()
                .ok_or(VerifyError::InvalidSparseUpdate);
        }
        self.nodes
            .get(&position)
            .copied()
            .ok_or(VerifyError::InvalidSparseUpdate)
    }

    /// Export authenticated and appended nodes for the next verification stage.
    /// Appended nodes occupy consecutive positions in both supported families.
    pub(crate) fn into_nodes(mut self) -> BTreeMap<Position<F>, D> {
        self.nodes.extend(
            self.appended
                .into_iter()
                .zip(0u64..)
                .map(|(digest, index)| (self.append_start + index, digest)),
        );
        self.nodes
    }

    /// Combine the current peaks with the family's size and inactive-peak root encoding.
    /// This produces a raw or grafted root component for the derived Current commitment.
    /// Every peak must be available. The caller derives `inactive_peaks` from the exact
    /// floor for the raw tree or the chunk-aligned floor for the grafted tree.
    pub(crate) fn root<H: Hasher<F, Digest = D>>(
        &self,
        hasher: &H,
        inactive_peaks: usize,
    ) -> Result<D, VerifyError> {
        let peaks = F::peaks(F::location_to_position(self.leaves))
            .map(|(position, _)| self.node(position))
            .collect::<Result<Vec<_>, _>>()?;
        hasher
            .root(self.leaves, inactive_peaks, peaks.iter())
            .map_err(|_| VerifyError::InvalidSparseUpdate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        merkle::{Bagging, Graftable, mmb, mmr},
        qmdb::{self, current::grafting},
    };
    use commonware_cryptography::{Hasher as _, Sha256};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Clone, Default)]
    struct CountingHasher(Arc<AtomicUsize>);

    impl<F: Family> Hasher<F> for CountingHasher {
        type Digest = <Sha256 as commonware_cryptography::Hasher>::Digest;

        fn hash(&self, parts: &[&[u8]]) -> Self::Digest {
            self.0.fetch_add(1, Ordering::Relaxed);
            Sha256::hash(parts)
        }

        fn root_bagging(&self) -> Bagging {
            qmdb::hash::ROOT_BAGGING
        }

        fn node_digest_pair(
            &self,
            nodes: [(Position<F>, &Self::Digest, &Self::Digest); 2],
        ) -> (Self::Digest, Self::Digest) {
            let [(a, al, ar), (b, bl, br)] = nodes;
            (self.node_digest(a, al, ar), self.node_digest(b, bl, br))
        }
    }

    fn check_shared_paths<F: Graftable>() {
        let hasher = CountingHasher::default();
        let height = 3;
        for size in [255, 256, 257, 1023, 1024, 1025] {
            let chunks = grafting::graftable_chunks::<F>(size, height);
            for stride in [1, 3, 17] {
                let mut original = SparseForest::<F, _>::new(Location::new(0), BTreeMap::new());
                let mut expected = SparseForest::<F, _>::new(Location::new(0), BTreeMap::new());
                for leaf in 0..size {
                    let position = Position::try_from(Location::<F>::new(leaf)).unwrap();
                    let changed =
                        (leaf >> height) < chunks && (leaf >> height).is_multiple_of(stride);
                    original
                        .append_digest(hasher.leaf_digest(position, &[0]), |forest, pos, height| {
                            forest.parent_digest(&hasher, pos, height)
                        })
                        .unwrap();
                    expected
                        .append_digest(
                            hasher.leaf_digest(position, &[u8::from(changed)]),
                            |forest, pos, height| forest.parent_digest(&hasher, pos, height),
                        )
                        .unwrap();
                }
                let changes: Vec<_> = (0..chunks)
                    .step_by(stride as usize)
                    .map(|chunk| {
                        let start = Location::new(chunk << height);
                        (
                            start,
                            expected
                                .node(F::subtree_root_position(start, height))
                                .unwrap(),
                        )
                    })
                    .collect();
                // A single changed chunk cannot reconstruct its path without siblings.
                let mut missing = SparseForest::<F, _>::new(Location::new(size), BTreeMap::new());
                assert!(
                    missing
                        .update_subtrees(&hasher, height, &changes[..1])
                        .is_err()
                );
                let mut original = SparseForest::new(Location::new(size), original.into_nodes());
                hasher.0.store(0, Ordering::Relaxed);
                original.update_subtrees(&hasher, height, &changes).unwrap();
                let hashes = hasher.0.load(Ordering::Relaxed);
                // Dense replacement must hash fewer than one parent per chunk overall.
                if stride == 1 {
                    assert!(hashes < chunks as usize);
                }
                assert_eq!(
                    original.root(&hasher, 0).unwrap(),
                    expected.root(&hasher, 0).unwrap()
                );
            }
        }
    }

    fn check_dense_appends<F: Family>() {
        let hasher = qmdb::hasher::<Sha256>();
        let mut full = crate::merkle::mem::Mem::<F, _>::new();
        let mut sparse = SparseForest::<F, _>::new(Location::new(0), BTreeMap::new());
        for leaf in 0..1025u64 {
            let element = leaf.to_be_bytes();
            let batch = full
                .new_batch()
                .add(&hasher, &element)
                .merkleize(&full, &hasher);
            full.apply_batch(&batch).unwrap();
            let position = F::location_to_position(Location::new(leaf));
            sparse
                .append_digest(
                    hasher.leaf_digest(position, &element),
                    |forest, pos, height| forest.parent_digest(&hasher, pos, height),
                )
                .unwrap();
            assert_eq!(
                sparse.root(&hasher, 0).unwrap(),
                full.root(&hasher, 0).unwrap()
            );
            if leaf % 257 == 0 {
                // Start another append epoch from only the authenticated frontier.
                let nodes = F::nodes_to_pin(sparse.leaves)
                    .map(|pos| (pos, full.get_node(pos).unwrap()))
                    .collect();
                sparse = SparseForest::new(sparse.leaves, nodes);
            }
        }
    }

    #[test]
    fn dense_appends_match_full_tree() {
        check_dense_appends::<mmr::Family>();
        check_dense_appends::<mmb::Family>();
    }

    #[test]
    fn shared_paths_match_full_reconstruction() {
        check_shared_paths::<mmr::Family>();
        check_shared_paths::<mmb::Family>();
    }
}
