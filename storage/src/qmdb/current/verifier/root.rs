//! Compute the post-state commitment from authenticated old nodes and derived operations.
//!
//! Unchanged subtrees retain their authenticated digests. Only appended log paths
//! and paths above changed activity chunks are rebuilt. Pending and partial bitmap
//! chunks remain separate root components until the family's geometry allows grafting.

use super::{
    BitmapFault, CurrentCommitment, VerifyError,
    bitmap::{VerifiedBitmapView, set_chunk_bit},
    floor::FloorResult,
    opening::AuthenticatedPreState,
    ordered::PreparedOrderedBatch,
    sparse::SparseForest,
};
use crate::{
    merkle::{Graftable, Location, PendingChunk as _, hasher::Hasher as _},
    qmdb::{
        self,
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        current::{grafting, roots::combine_roots},
        operation::Key,
    },
};
use alloc::{collections::BTreeMap, vec::Vec};
use commonware_codec::{Codec, Write as _};
use commonware_cryptography::Hasher;
use commonware_parallel::Strategy;
use commonware_utils::bitmap::BitMap;

/// Recompute all Current root components for the locally derived physical suffix.
///
/// The final comparison with `claimed_post` depends on this function computing the
/// root from authenticated old nodes and the required operations. Reusing unchanged
/// subtrees preserves the rest of the log and bitmap, while rebuilding changed paths
/// commits to every application operation, floor copy, and activity change.
///
/// Append application operations, floor copies, then the terminal. Initially mark
/// updates and the terminal active. Clear superseded records, the old terminal, and
/// copy sources, including any sources in this suffix. Start old chunks from verified
/// bytes and newly allocated chunks from zero so untouched activity bits are preserved.
///
/// First replace changed chunks that are already graftable in the pre-state, using
/// their final bitmap bytes, and rebuild their shared ancestors. Then append the same
/// encoded leaves to both forests. When an append creates a grafting-height node,
/// graft its final chunk bytes once; later parents consume that final digest. This
/// also handles old pending chunks that become graftable. Compute roots with the new
/// floor's inactive-peak counts. Include pending and partial chunk digests according
/// to the new log size. The terminal payload commits to the exact new floor and metadata.
/// All inputs must be the authenticated or derived outputs of this transition.
/// With `v` authenticated nodes in the bitmap view, `k` as defined by the module,
/// and Merkle height `h`, worst-case CPU is O(v + k * h * log(v + k * h + 2))
/// and memory is O(v + k * h), for fixed-size records and chunks. The authenticated
/// node maps are consumed, not cloned. For `d > 0` changed graftable chunks among
/// `L` possible chunks, their update visits O(d * (1 + log(L / d)) + h) nodes and
/// hashes each shared ancestor once. This replaces O(d * h) independent path walks;
/// sparse B-tree accesses still cost logarithmic time. Appends are a separate cost.
/// New append nodes use dense storage with O(1) lookup and insertion; old sibling
/// lookups still use the sparse map. Raw and grafted appends share every new hash below
/// the grafting height, where activity bits do not affect the tree. Independent leaf hashes
/// use per-worker encoding buffers; appends stay in log order.
pub(crate) fn derive_child_commitment<F, K, V, H, const N: usize, S: Strategy>(
    pre: &AuthenticatedPreState<F, K, V, H, N>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    floor: &FloorResult<F, K, V>,
    bitmap: VerifiedBitmapView<F, H::Digest, N>,
    strategy: &S,
) -> Result<CurrentCommitment<F, K, V, H, N>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    // Floor advancement has already checked the suffix bounds and copy destinations.
    let suffix = prepared
        .application_ops
        .iter()
        .chain(&floor.copies)
        .chain(core::iter::once(&floor.terminal));
    let chunk_bits = BitMap::<N>::CHUNK_SIZE_BITS;
    let mut chunks = BTreeMap::new();
    let mut mutate_bit = |location: Location<F>, value: bool| -> Result<(), VerifyError> {
        let index = location.as_u64() / chunk_bits;
        let chunk = match chunks.entry(index) {
            alloc::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            alloc::collections::btree_map::Entry::Vacant(entry) => {
                let bytes = if index < pre.commitment.size().as_u64().div_ceil(chunk_bits) {
                    *bitmap
                        .explicit_chunks
                        .get(&index)
                        .ok_or(BitmapFault::Coverage)?
                } else {
                    BitMap::<N>::EMPTY_CHUNK
                };
                entry.insert(bytes)
            }
        };
        set_chunk_bit(chunk, location.as_u64(), value)
    };
    for (operation, index) in suffix.clone().zip(0u64..) {
        let location = pre
            .commitment
            .size()
            .checked_add(index)
            .ok_or(VerifyError::Overflow)?;
        let active = matches!(
            operation,
            Operation::Update(_) | Operation::CommitFloor(_, _)
        );
        mutate_bit(location, active)?;
    }
    for location in prepared
        .superseded
        .iter()
        .copied()
        .chain(core::iter::once(pre.terminal_location))
        .chain(floor.copy_locations.iter().copied())
    {
        mutate_bit(location, false)?;
    }
    let get_chunk = |index: &u64| {
        chunks
            .get(index)
            .or_else(|| bitmap.explicit_chunks.get(index))
            .ok_or(BitmapFault::Coverage)
    };
    let hasher = qmdb::hasher::<H>();
    let mut raw = SparseForest::new(pre.commitment.size(), bitmap.raw_nodes);
    let mut grafted = SparseForest::new(pre.commitment.size(), bitmap.grafted_nodes);
    let grafting_height = grafting::height::<N>();
    let old_graftable =
        grafting::graftable_chunks::<F>(pre.commitment.size().as_u64(), grafting_height)
            .min(pre.commitment.size().as_u64() / chunk_bits);
    let mut changes = Vec::new();
    for (&chunk_index, chunk) in chunks.range(..old_graftable) {
        let leaf_start = chunk_index * chunk_bits;
        let raw_digest = raw.node(F::subtree_root_position(
            Location::new(leaf_start),
            grafting_height,
        ))?;
        let grafted_digest = if *chunk == [0u8; N] {
            raw_digest
        } else {
            hasher.hash(&[chunk.as_slice(), raw_digest.as_ref()])
        };
        changes.push((Location::new(leaf_start), grafted_digest));
    }
    grafted.update_subtrees(&hasher, grafting_height, &changes)?;

    let leaves = strategy.map_init_collect_vec(
        suffix.zip(0u64..).collect::<Vec<_>>(),
        bytes::BytesMut::new,
        |encoded, (operation, index)| {
            encoded.clear();
            operation.write(encoded);
            let position = F::location_to_position(pre.commitment.size() + index);
            hasher.leaf_digest(position, encoded)
        },
    );
    for leaf_digest in leaves {
        raw.append_digest(leaf_digest, |forest, position, height| {
            forest.parent_digest(&hasher, position, height)
        })?;
        grafted.append_digest(leaf_digest, |forest, position, height| {
            if height > grafting_height {
                return forest.parent_digest(&hasher, position, height);
            }
            let digest = raw.node(position)?;
            if height < grafting_height {
                return Ok(digest);
            }
            let leaf_start = F::leftmost_leaf(position, height);
            let chunk_index = leaf_start.as_u64() >> grafting_height;
            let chunk = get_chunk(&chunk_index)?;
            if *chunk == [0u8; N] {
                Ok(digest)
            } else {
                Ok(hasher.hash(&[chunk.as_slice(), digest.as_ref()]))
            }
        })?;
    }

    let child_graftable =
        grafting::graftable_chunks::<F>(floor.child_size.as_u64(), grafting_height)
            .min(floor.child_size.as_u64() / chunk_bits);

    let raw_root = raw.root(&hasher, F::inactive_peaks(floor.child_size, floor.floor))?;
    let grafted_inactive =
        grafting::chunk_aligned_inactive_peaks::<F>(floor.child_size, floor.floor, grafting_height)
            .map_err(|_| VerifyError::InvalidSparseUpdate)?;
    let grafted_root = grafted.root(&hasher, grafted_inactive)?;
    let complete_chunks = floor.child_size.as_u64() / chunk_bits;
    let pending_digest = if complete_chunks > child_graftable {
        let chunk = get_chunk(&child_graftable)?;
        Some(hasher.digest(chunk))
    } else {
        None
    };
    let pending: F::PendingChunk<H::Digest> = pending_digest
        .try_into()
        .map_err(|_| VerifyError::InvalidSparseUpdate)?;
    let next_bit = floor.child_size.as_u64() % chunk_bits;
    let partial_digest = if next_bit == 0 {
        None
    } else {
        let chunk = get_chunk(&complete_chunks)?;
        Some(hasher.digest(chunk))
    };
    let root = combine_roots::<H>(
        &raw_root,
        &grafted_root,
        pending.as_ref(),
        partial_digest.as_ref().map(|digest| (next_bit, digest)),
    );
    CurrentCommitment::from_parts(floor.child_size, root)
        .map_err(|_| VerifyError::InvalidSparseUpdate)
}
