//! Authenticate activity bits and the Merkle nodes needed to change them.
//!
//! The floor scan needs complete coverage so a prover cannot hide a live operation.
//! Changed chunks need their old bytes so unrelated bits survive the transition.
//! Graftable chunks are checked in both raw and grafted trees at operation-tree positions.
//! Pending and partial chunks are checked against the pre-state root opening.

#[cfg(feature = "std")]
use super::subtree::create_multi_subtree_proof;
use super::{
    BitmapFault, VerifyError,
    opening::AuthenticatedPreState,
    ordered::PreparedOrderedBatch,
    subtree::{SubtreeTarget, verify_multi_subtree_inclusion},
};
use crate::{
    merkle::{self, Family, Graftable, Location, PendingChunk as _, Position, hasher::Hasher as _},
    qmdb::{self, any::ValueEncoding, current::grafting, operation::Key},
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use commonware_cryptography::{Digest, Hasher};
use commonware_parallel::Strategy;
use commonware_utils::bitmap::BitMap;
use core::marker::PhantomData;

type ChunkIntervals = Vec<(u64, u64)>;
type RequiredChunks = (ChunkIntervals, BTreeSet<u64>);

/// The fewest raw and grafted targets for which [`verify_bitmap`] authenticates the two trees
/// on the strategy; smaller proofs cost less to check than to dispatch.
const PARALLEL_TARGETS: usize = 32;

/// Produce exact coverage with chunk bytes, zero subtrees, and shared subtree inclusion paths.
/// The verifier's interval calculation also drives production, so floor-scan gaps
/// and unrelated extra chunks cannot be introduced by a different coverage rule.
/// Each run of clear, graftable chunks whose bytes the verifier does not need becomes the
/// fewest aligned zero subtrees whose roots exist, as [`zero_blocks`] splits it; every other
/// chunk is sent in full. A batch's coverage therefore costs O(k * log n) units for `k`
/// changed chunks and records reached by the floor walk, however many clear chunks the
/// scan crosses: each clear run between them takes O(log n) blocks. CPU is O(w * N + (u + 1) * h * log((u + 1) * h + 2)) for `w` scanned
/// chunks compared in memory, `u` emitted units, and height `h`; memory and node reads are
/// O((u + 1) * h + e * N) for `e` explicit chunks, excluding backend costs.
#[cfg(feature = "std")]
pub(super) async fn build_bitmap_proof<F, K, V, H, B, R, G, const N: usize>(
    pre_size: Location<F>,
    old_floor: Location<F>,
    floor_scan_end: Location<F>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    bitmap: &B,
    raw_storage: &R,
    grafted_storage: &G,
) -> Result<BitmapProof<F, H::Digest, N>, super::BuildError<F>>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    B: commonware_utils::bitmap::Readable<N>,
    R: merkle::storage::Storage<F, Digest = H::Digest>,
    G: merkle::storage::Storage<F, Digest = H::Digest>,
{
    if pre_size == Location::new(1) {
        return Ok(BitmapProof::Initial { floor_scan_end });
    }
    let chunk_bits = BitMap::<N>::CHUNK_SIZE_BITS;
    let height = grafting::height::<N>();
    let graftable = grafting::graftable_chunks::<F>(*pre_size, height).min(*pre_size / chunk_bits);
    let (required, forced_explicit) = required_chunk_intervals(
        pre_size,
        old_floor,
        floor_scan_end,
        prepared,
        chunk_bits,
        graftable,
    );
    let chunk = |index: u64| {
        usize::try_from(index)
            .map(|index| bitmap.get_chunk(index))
            .map_err(|_| super::BuildError::InvalidState("a bitmap chunk index exceeds usize"))
    };
    let mut units = Vec::new();
    let mut raw_targets = Vec::new();
    let mut grafted_targets = Vec::new();
    let hasher = qmdb::hasher::<H>();
    for (start, end) in required {
        let mut chunk_index = start;
        while chunk_index < end {
            let mut run_end = chunk_index;
            while run_end < end
                && run_end < graftable
                && !forced_explicit.contains(&run_end)
                && chunk(run_end)? == BitMap::<N>::EMPTY_CHUNK
            {
                run_end += 1;
            }
            for (first_chunk, chunk_height) in
                zero_blocks::<F>(chunk_index, run_end, *pre_size, height)
            {
                let leaf_start = Location::new(first_chunk * chunk_bits);
                let target_height = height + chunk_height;
                let position = F::subtree_root_position(leaf_start, target_height);
                let digest = raw_storage
                    .get_node(position)
                    .await?
                    .ok_or(merkle::Error::ElementPruned(position))?;
                // A clear block grafts no bytes, so both trees hold the same digest there.
                if grafted_storage.get_node(position).await? != Some(digest) {
                    return Err(super::BuildError::InvalidState(
                        "a clear bitmap block's grafted digest differs from its raw digest",
                    ));
                }
                let target = SubtreeTarget {
                    leaf_start,
                    height: target_height,
                    digest,
                };
                raw_targets.push(target);
                grafted_targets.push(target);
                units.push(BitmapUnit::zero_subtree(first_chunk, chunk_height, digest));
            }
            if run_end == end {
                break;
            }
            let bytes = chunk(run_end)?;
            let raw_subtree_digest = if run_end < graftable {
                let leaf_start = Location::new(run_end * chunk_bits);
                let position = F::subtree_root_position(leaf_start, height);
                let digest = raw_storage
                    .get_node(position)
                    .await?
                    .ok_or(merkle::Error::ElementPruned(position))?;
                raw_targets.push(SubtreeTarget {
                    leaf_start,
                    height,
                    digest,
                });
                let grafted_digest = if bytes == BitMap::<N>::EMPTY_CHUNK {
                    digest
                } else {
                    hasher.hash(&[bytes.as_slice(), digest.as_ref()])
                };
                grafted_targets.push(SubtreeTarget {
                    leaf_start,
                    height,
                    digest: grafted_digest,
                });
                Some(digest)
            } else {
                None
            };
            units.push(BitmapUnit::ExplicitChunk {
                chunk_index: run_end,
                bytes,
                raw_subtree_digest,
            });
            chunk_index = run_end + 1;
        }
    }
    let (raw_proof, grafted_proof) = if raw_targets.is_empty() {
        (None, None)
    } else {
        (
            Some(
                create_multi_subtree_proof(
                    raw_storage,
                    F::inactive_peaks(pre_size, old_floor),
                    &raw_targets,
                )
                .await?,
            ),
            Some(
                create_multi_subtree_proof(
                    grafted_storage,
                    grafting::chunk_aligned_inactive_peaks::<F>(pre_size, old_floor, height)?,
                    &grafted_targets,
                )
                .await?,
            ),
        )
    };
    let mut grafted_append_frontier = Vec::new();
    for position in F::nodes_to_pin(pre_size) {
        grafted_append_frontier.push(
            grafted_storage
                .get_node(position)
                .await?
                .ok_or(merkle::Error::ElementPruned(position))?,
        );
    }
    Ok(BitmapProof::Proven(ProvenBitmapProof {
        pre_size,
        floor_scan_end,
        units,
        raw_proof,
        grafted_proof,
        grafted_append_frontier,
    }))
}

/// Split the run of chunks `[first, end)` into the fewest aligned blocks whose roots exist in a
/// tree of `leaves` operations, returning each block's first chunk and height in chunk-index
/// space. A block of height `h` holds `2^h` chunks, starts at a multiple of `2^h`, and exists
/// once the tree has formed its node at height `grafting_height + h`, which a Merkle mountain
/// belt does only some time after the block's last leaf. The caller supplies graftable chunks,
/// so a single chunk always exists.
#[cfg(feature = "std")]
pub(super) fn zero_blocks<F: Graftable>(
    mut first: u64,
    end: u64,
    leaves: u64,
    grafting_height: u32,
) -> Vec<(u64, u32)> {
    let mut blocks = Vec::new();
    while first < end {
        let mut chunk_height = first.trailing_zeros().min((end - first).ilog2());
        while chunk_height > 0
            && first >> chunk_height
                >= grafting::graftable_chunks::<F>(leaves, grafting_height + chunk_height)
        {
            chunk_height -= 1;
        }
        blocks.push((first, chunk_height));
        first += 1 << chunk_height;
    }
    blocks
}

/// Witnesses for activity bits read or changed while verifying one batch.
/// Choose `Initial` only for the fixed initial checkpoint. All other states require
/// `Proven`. The scan endpoint is a claim until floor replay checks it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BitmapProof<F: Graftable, D: Digest, const N: usize> {
    /// The synthesized one-bit bitmap of the fixed initial state.
    Initial {
        /// End of `[old_floor, floor_scan_end)`, checked against the floor replay's stopping point.
        floor_scan_end: Location<F>,
    },
    /// Proof material for authenticating a non-initial state's bitmap.
    Proven(ProvenBitmapProof<F, D, N>),
}

/// Coverage, inclusion paths, and append peaks for a non-initial activity bitmap.
///
/// `units` must exactly partition the verifier-required chunk intervals. The raw and
/// grafted proofs authenticate targets derived from those units independently, using
/// operation-space positions. Their digest vectors contain missing frontier nodes in
/// increasing position order, not a separate path per target. Each proof is absent
/// exactly when there are no graftable targets. The append frontier supplies all peaks
/// needed to extend the grafted tree even when coverage proofs touch only old chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenBitmapProof<F: Graftable, D: Digest, const N: usize> {
    /// Claimed pre-state operation count.
    pub pre_size: Location<F>,
    /// End of `[old_floor, floor_scan_end)`, checked against the floor replay's stopping point.
    pub floor_scan_end: Location<F>,
    /// Coverage units in increasing chunk order, partitioning all required intervals.
    pub units: Vec<BitmapUnit<F, D, N>>,
    /// Inclusion proof for the units' subtree digests in the raw operations tree.
    pub raw_proof: Option<merkle::Proof<F, D>>,
    /// Inclusion proof for the units' bitmap-adjusted digests in the grafted tree.
    pub grafted_proof: Option<merkle::Proof<F, D>>,
    /// Grafted tree's append nodes in `F::nodes_to_pin(pre_size)` order, at operation-tree positions.
    pub grafted_append_frontier: Vec<D>,
}

/// One portion of the required bitmap coverage, measured in `8 * N`-bit chunks.
/// Explicit bytes are required for chunks containing invalidated records, trailing
/// chunks, and chunks cut by a scan boundary. Other all-zero regions may use a
/// shared raw/grafted subtree digest instead of their bytes, as the bundled prover
/// does. Both forms are claims until roots are checked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BitmapUnit<F: Graftable, D: Digest, const N: usize> {
    /// Full bytes for one chunk.
    ExplicitChunk {
        /// Zero-based bitmap chunk index.
        chunk_index: u64,
        /// Exact chunk bytes.
        bytes: [u8; N],
        /// Raw subtree digest at height `log2(8 * N)`, present only for a graftable chunk.
        raw_subtree_digest: Option<D>,
    },
    /// Claim that `2^chunk_height` aligned, complete graftable chunks contain no set bits.
    /// The same positioned subtree digest must authenticate under both roots. Raw
    /// inclusion alone says nothing about the activity bitmap.
    ZeroSubtree {
        /// First covered chunk.
        first_chunk: u64,
        /// Height in chunk-index space.
        chunk_height: u32,
        /// Digest authenticated independently in raw and grafted trees.
        common_subtree_digest: D,
        /// Family profile marker.
        _family: PhantomData<F>,
    },
}

impl<F: Graftable, D: Digest, const N: usize> BitmapUnit<F, D, N> {
    /// Package an all-zero interval claim without authenticating it.
    /// The first chunk must be aligned to `2^chunk_height`; verification checks geometry,
    /// required coverage, and inclusion of the shared digest under both pre-state roots.
    pub const fn zero_subtree(
        first_chunk: u64,
        chunk_height: u32,
        common_subtree_digest: D,
    ) -> Self {
        Self::ZeroSubtree {
            first_chunk,
            chunk_height,
            common_subtree_digest,
            _family: PhantomData,
        }
    }
}

/// Checked activity chunks and tree nodes belonging to one authenticated pre-state.
/// Coverage is complete for the requested scan and mutations, not the whole database.
/// `zero_ranges` use half-open chunk intervals. Node maps use operation-tree positions.
/// The scan endpoint is bounded here and justified later by deterministic replay.
pub(crate) struct VerifiedBitmapView<F: Family, D: Digest, const N: usize> {
    pub(crate) floor_scan_end: Location<F>,
    pub(crate) explicit_chunks: BTreeMap<u64, [u8; N]>,
    pub(crate) zero_ranges: Vec<(u64, u64)>,
    pub(crate) raw_nodes: BTreeMap<Position<F>, D>,
    pub(crate) grafted_nodes: BTreeMap<Position<F>, D>,
}

impl<F: Family, D: Digest, const N: usize> VerifiedBitmapView<F, D, N> {
    /// Return an authenticated bit, or `None` when its chunk was not covered.
    /// Floor replay uses this to check that the terminal and superseded records are
    /// active. An explicit chunk supplies its stored bit, and a verified zero interval
    /// supplies `false`. An uncovered chunk supplies no evidence about activity.
    pub(crate) fn bit(&self, loc: Location<F>) -> Option<bool> {
        let chunk_index = *loc / BitMap::<N>::CHUNK_SIZE_BITS;
        if let Some(chunk) = self.explicit_chunks.get(&chunk_index) {
            return Some(BitMap::<N>::get_bit_from_chunk(chunk, *loc));
        }
        let insertion = self
            .zero_ranges
            .partition_point(|(start, _)| *start <= chunk_index);
        let covering = insertion
            .checked_sub(1)
            .and_then(|index| self.zero_ranges.get(index));
        covering.and_then(|(_, end)| (chunk_index < *end).then_some(false))
    }

    /// Iterate set-bit locations in `[start, end)` in increasing order.
    /// Candidate verification compares this sequence with the supplied payload locations
    /// to detect omissions and extras. The caller must first verify complete coverage
    /// of the interval. Verified zero ranges contribute no locations, and this iterator
    /// does not detect uncovered ranges.
    pub(crate) fn set_bits(
        &self,
        start: Location<F>,
        end: Location<F>,
    ) -> impl Iterator<Item = Location<F>> + '_ {
        let bits = BitMap::<N>::CHUNK_SIZE_BITS;
        self.explicit_chunks
            .range(*start / bits..end.as_u64().div_ceil(bits))
            .flat_map(move |(&index, chunk)| {
                let first = index * bits;
                let from = start.as_u64().max(first);
                let to = end.as_u64().min(first.saturating_add(bits));
                (from..to).filter_map(move |loc| {
                    BitMap::<N>::get_bit_from_chunk(chunk, loc).then_some(Location::new(loc))
                })
            })
    }
}

/// Compute the exact chunk intervals the proof must cover, plus chunks needing bytes.
///
/// Floor replay needs complete scan coverage to detect every live candidate. Root
/// reconstruction needs full bytes for changed old chunks to preserve unrelated bits.
/// This helper determines those requirements before inspecting the supplied units.
///
/// Cover the proposed floor scan, every superseded record, and the old terminal.
/// Force scan-boundary chunks to be explicit when the boundary cuts a chunk. Also
/// require old pending and partial bytes because appends may graft or complete them.
/// Return sorted intervals with overlaps merged. Touching intervals remain separate
/// proof boundaries. The caller has bounded the scan within the authenticated size
/// and validated the chunk profile, but has not yet justified the scan's endpoint.
fn required_chunk_intervals<F, K, V>(
    pre_size: Location<F>,
    old_floor: Location<F>,
    floor_scan_end: Location<F>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    chunk_bits: u64,
    graftable_chunks: u64,
) -> RequiredChunks
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
{
    let mut intervals = Vec::new();
    let mut forced_explicit = BTreeSet::new();
    if floor_scan_end > old_floor {
        let first = *old_floor / chunk_bits;
        let last = (*floor_scan_end - 1) / chunk_bits;
        intervals.push((first, last + 1));
        if !old_floor.as_u64().is_multiple_of(chunk_bits) {
            forced_explicit.insert(first);
        }
        if !floor_scan_end.as_u64().is_multiple_of(chunk_bits) {
            forced_explicit.insert(last);
        }
    }
    for location in prepared
        .superseded
        .iter()
        .copied()
        .chain(core::iter::once(pre_size - 1))
    {
        forced_explicit.insert(*location / chunk_bits);
    }
    // Appends may graft the old pending chunk or complete the old partial chunk.
    let complete = *pre_size / chunk_bits;
    if complete > graftable_chunks {
        forced_explicit.insert(graftable_chunks);
    }
    if !pre_size.as_u64().is_multiple_of(chunk_bits) {
        forced_explicit.insert(complete);
    }
    intervals.extend(forced_explicit.iter().map(|index| (*index, *index + 1)));
    intervals.sort_unstable();
    let mut merged: ChunkIntervals = Vec::new();
    for (start, end) in intervals {
        if let Some(last) = merged.last_mut()
            && start < last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    (merged, forced_explicit)
}

/// Combine authenticated node sets, rejecting unequal digests at the same position.
/// Both sets must belong to the same tree and checkpoint. This preserves one coherent
/// source of siblings when an inclusion path overlaps an append frontier.
fn merge_verified_nodes<F: Family, D: Digest>(
    destination: &mut BTreeMap<Position<F>, D>,
    source: BTreeMap<Position<F>, D>,
) -> Result<(), VerifyError> {
    for (position, digest) in source {
        if destination
            .insert(position, digest)
            .is_some_and(|existing| existing != digest)
        {
            return Err(BitmapFault::NodeConflict.into());
        }
    }
    Ok(())
}

/// Set or clear the bit for `location`, an index into the whole bitmap, within its chunk.
pub(crate) fn set_chunk_bit<const N: usize>(
    chunk: &mut [u8; N],
    location: u64,
    value: bool,
) -> Result<(), VerifyError> {
    let offset = usize::try_from(location % BitMap::<N>::CHUNK_SIZE_BITS / 8)
        .map_err(|_| VerifyError::Overflow)?;
    let byte = chunk.get_mut(offset).ok_or(VerifyError::Overflow)?;
    let mask = 1u8 << (location % 8);
    if value {
        *byte |= mask;
    } else {
        *byte &= !mask;
    }
    Ok(())
}

/// Authenticate exact bitmap coverage and sparse-tree siblings for this transition.
///
/// The returned view supplies the candidate locations for floor replay and the old
/// bits and Merkle siblings for root reconstruction. Checking both coverage and
/// contents prevents a bundle from hiding live candidates or changing unrelated bits.
///
/// Require units to partition the requested chunk intervals in order, without gaps,
/// overlaps, or extras. Explicit chunks provide bytes. Zero subtrees must be aligned,
/// complete, graftable, and avoid every chunk whose bytes will be read or changed.
/// For each graftable explicit chunk, authenticate raw digest `r` in the raw tree and
/// its prescribed bitmap transform in the grafted tree. A zero subtree supplies the
/// same digest to both proofs: under the hash binding assumption and a valid Current
/// pre-state, equality establishes that no chunk in that subtree has active bits.
/// Check pending/partial bytes directly against the opening, including zero unused
/// partial bits. Bind both proofs' sizes and inactive peaks to the authenticated floor.
///
/// Authenticate the grafted append frontier by reconstructing its root and merge it
/// with the proof paths, rejecting conflicting nodes. The raw frontier was verified
/// by the opening. This check authenticates coverage and bits for the claimed scan
/// interval. `advance_floor` must still check its endpoint against deterministic replay.
/// For `u` units, `e` of them explicit, and Merkle height `h`, worst-case CPU is
/// O(k * log(k + 2) + (u + 1) * h * log((u + 1) * h + 2) + e * N), and temporary
/// memory is O(k + (u + 1) * h + e * N). Here k bounds superseded locations. Zero
/// subtrees are processed by their endpoints and Merkle paths, not by scanning all
/// covered bits. Explicit chunks must be read and hashed in full.
pub(crate) fn verify_bitmap<F, K, V, H, const N: usize, S: Strategy>(
    pre: &AuthenticatedPreState<F, K, V, H, N>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    proof: &BitmapProof<F, H::Digest, N>,
    strategy: &S,
) -> Result<VerifiedBitmapView<F, H::Digest, N>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    let chunk_bits = BitMap::<N>::CHUNK_SIZE_BITS;
    let grafting_height = grafting::height::<N>();
    let (floor_scan_end, proven) = match proof {
        BitmapProof::Initial { floor_scan_end } if pre.initial => (*floor_scan_end, None),
        BitmapProof::Proven(proof) if !pre.initial => (proof.floor_scan_end, Some(proof)),
        _ => return Err(BitmapFault::Kind.into()),
    };
    if floor_scan_end < pre.old_floor || floor_scan_end > pre.commitment.size() {
        return Err(BitmapFault::ScanBounds.into());
    }
    let complete_chunks = pre.commitment.size().as_u64() / chunk_bits;
    let graftable_chunks =
        grafting::graftable_chunks::<F>(pre.commitment.size().as_u64(), grafting_height)
            .min(complete_chunks);
    let partial_chunk = pre.commitment.size().as_u64() % chunk_bits;
    let (required, forced_explicit) = required_chunk_intervals(
        pre.commitment.size(),
        pre.old_floor,
        floor_scan_end,
        prepared,
        chunk_bits,
        graftable_chunks,
    );
    let Some(proven) = proven else {
        let mut chunk = BitMap::<N>::EMPTY_CHUNK;
        set_chunk_bit(&mut chunk, 0, true)?;
        return Ok(VerifiedBitmapView {
            floor_scan_end,
            explicit_chunks: BTreeMap::from([(0, chunk)]),
            zero_ranges: Vec::new(),
            raw_nodes: pre.raw_frontier.clone(),
            grafted_nodes: pre.raw_frontier.clone(),
        });
    };
    if proven.pre_size != pre.commitment.size() {
        return Err(BitmapFault::PreSize.into());
    }
    let mut explicit_chunks = BTreeMap::new();
    let mut zero_ranges = Vec::new();
    let mut raw_targets = Vec::new();
    let mut grafted_targets = Vec::new();
    let merkle_hasher = qmdb::hasher::<H>();

    let mut interval_index = 0usize;
    let mut cursor = required.first().map_or(0, |interval| interval.0);
    for unit in &proven.units {
        let Some((_, required_end)) = required.get(interval_index).copied() else {
            return Err(BitmapFault::Coverage.into());
        };
        match unit {
            BitmapUnit::ExplicitChunk {
                chunk_index,
                bytes,
                raw_subtree_digest,
            } => {
                if *chunk_index != cursor {
                    return Err(BitmapFault::Coverage.into());
                }
                let expected_raw = *chunk_index < graftable_chunks;
                if raw_subtree_digest.is_some() != expected_raw {
                    return Err(BitmapFault::RawDigest.into());
                }
                if *chunk_index == complete_chunks && partial_chunk != 0 {
                    let used_bytes = usize::try_from(partial_chunk.div_ceil(8))
                        .map_err(|_| VerifyError::Overflow)?;
                    let (used, unused) = bytes
                        .split_at_checked(used_bytes)
                        .ok_or(VerifyError::Overflow)?;
                    let last_used = used.last().copied().unwrap_or_default();
                    if unused.iter().any(|byte| *byte != 0)
                        || (!partial_chunk.is_multiple_of(8)
                            && last_used >> (partial_chunk % 8) != 0)
                    {
                        return Err(BitmapFault::PartialChunk.into());
                    }
                    let Some((next_bit, digest)) = pre.witness.partial_chunk else {
                        return Err(BitmapFault::PartialChunk.into());
                    };
                    if next_bit != partial_chunk || merkle_hasher.digest(bytes) != digest {
                        return Err(BitmapFault::PartialChunk.into());
                    }
                } else if *chunk_index == graftable_chunks && *chunk_index < complete_chunks {
                    if pre.witness.pending_chunk_digest.as_ref()
                        != Some(&merkle_hasher.digest(bytes))
                    {
                        return Err(BitmapFault::PendingChunk.into());
                    }
                } else if *chunk_index >= complete_chunks {
                    return Err(BitmapFault::Coverage.into());
                }
                if let Some(raw_digest) = raw_subtree_digest {
                    let leaf_start = chunk_index * chunk_bits;
                    let grafted_digest = if *bytes == [0u8; N] {
                        *raw_digest
                    } else {
                        merkle_hasher.hash(&[bytes.as_slice(), raw_digest.as_ref()])
                    };
                    raw_targets.push(SubtreeTarget {
                        leaf_start: Location::new(leaf_start),
                        height: grafting_height,
                        digest: *raw_digest,
                    });
                    grafted_targets.push(SubtreeTarget {
                        leaf_start: Location::new(leaf_start),
                        height: grafting_height,
                        digest: grafted_digest,
                    });
                }
                // Exact coverage makes chunk indexes unique.
                explicit_chunks.insert(*chunk_index, *bytes);
                cursor = cursor.checked_add(1).ok_or(VerifyError::Overflow)?;
            }
            BitmapUnit::ZeroSubtree {
                first_chunk,
                chunk_height,
                common_subtree_digest,
                ..
            } => {
                let width = 1u64
                    .checked_shl(*chunk_height)
                    .ok_or(VerifyError::Overflow)?;
                let end = first_chunk
                    .checked_add(width)
                    .ok_or(VerifyError::Overflow)?;
                if *first_chunk != cursor || end > required_end {
                    return Err(BitmapFault::Coverage.into());
                }
                if !first_chunk.is_multiple_of(width) {
                    return Err(BitmapFault::Misaligned.into());
                }
                if end > graftable_chunks {
                    return Err(BitmapFault::Ungraftable.into());
                }
                if forced_explicit.range(*first_chunk..end).next().is_some() {
                    return Err(BitmapFault::NeedsBytes.into());
                }
                // The bounded chunk interval also bounds its height and operation locations.
                let height = grafting_height + chunk_height;
                let leaf_start = first_chunk * chunk_bits;
                let target = SubtreeTarget {
                    leaf_start: Location::new(leaf_start),
                    height,
                    digest: *common_subtree_digest,
                };
                raw_targets.push(target);
                grafted_targets.push(target);
                zero_ranges.push((*first_chunk, end));
                cursor = end;
            }
        }
        if cursor == required_end {
            interval_index += 1;
            if let Some((start, _)) = required.get(interval_index) {
                cursor = *start;
            }
        }
    }
    if interval_index != required.len() {
        return Err(BitmapFault::Coverage.into());
    }

    let raw_inactive = F::inactive_peaks(pre.commitment.size(), pre.old_floor);
    let grafted_inactive = pre.grafted_inactive_peaks;
    let authenticate = |proof: &Option<merkle::Proof<F, H::Digest>>,
                        targets: &[_],
                        root: &_,
                        inactive,
                        mismatch| {
        match proof {
            None if targets.is_empty() => Ok(BTreeMap::new()),
            Some(proof)
                if !targets.is_empty()
                    && proof.leaves == pre.commitment.size()
                    && proof.inactive_peaks == inactive =>
            {
                verify_multi_subtree_inclusion(proof, &merkle_hasher, targets, root, mismatch)
            }
            _ => Err(BitmapFault::ProofShape.into()),
        }
    };
    let raw = || {
        authenticate(
            &proven.raw_proof,
            &raw_targets,
            &pre.ops_root,
            raw_inactive,
            BitmapFault::RawRoot,
        )
    };
    let grafted = || {
        authenticate(
            &proven.grafted_proof,
            &grafted_targets,
            &pre.witness.grafted_root,
            grafted_inactive,
            BitmapFault::GraftedRoot,
        )
    };
    // Small paths cost less to check directly than to dispatch to a worker pool.
    // The same authentication closures run in either case, before merging any nodes.
    let (raw_nodes, grafted_nodes) = if raw_targets.len() + grafted_targets.len() < PARALLEL_TARGETS
    {
        (raw(), grafted())
    } else {
        strategy.join(raw, grafted)
    };
    let mut raw_nodes = raw_nodes?;
    let mut grafted_nodes = grafted_nodes?;

    let frontier_positions = F::nodes_to_pin(pre.commitment.size()).collect::<Vec<_>>();
    if frontier_positions.len() != proven.grafted_append_frontier.len()
        || F::peaks(F::location_to_position(pre.commitment.size()))
            .map(|(position, _)| position)
            .ne(frontier_positions.iter().copied())
    {
        return Err(BitmapFault::Frontier.into());
    }
    let reconstructed = merkle_hasher
        .root(
            pre.commitment.size(),
            grafted_inactive,
            proven.grafted_append_frontier.iter(),
        )
        .map_err(|_| BitmapFault::Frontier)?;
    if reconstructed != pre.witness.grafted_root {
        return Err(BitmapFault::Frontier.into());
    }
    merge_verified_nodes(
        &mut grafted_nodes,
        frontier_positions
            .into_iter()
            .zip(proven.grafted_append_frontier.iter().copied())
            .collect(),
    )?;
    merge_verified_nodes(&mut raw_nodes, pre.raw_frontier.clone())?;

    Ok(VerifiedBitmapView {
        floor_scan_end,
        explicit_chunks,
        zero_ranges,
        raw_nodes,
        grafted_nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{merkle::mmr, qmdb::any::value::FixedEncoding};
    use alloc::vec;
    use commonware_cryptography::Sha256;
    use commonware_utils::sequence::FixedBytes;

    type F = mmr::Family;
    type Key = FixedBytes<8>;

    /// The required intervals and the chunks that need bytes, with 8-bit chunks.
    fn required(
        pre_size: u64,
        old_floor: u64,
        floor_scan_end: u64,
        superseded: &[u64],
        graftable_chunks: u64,
    ) -> (Vec<(u64, u64)>, Vec<u64>) {
        let prepared = PreparedOrderedBatch::<F, Key, FixedEncoding<Key>> {
            application_ops: Vec::new(),
            superseded: superseded.iter().map(|loc| Location::new(*loc)).collect(),
            deleted_keys: BTreeSet::new(),
            post_empty: false,
        };
        let (intervals, explicit) = required_chunk_intervals(
            Location::new(pre_size),
            Location::new(old_floor),
            Location::new(floor_scan_end),
            &prepared,
            8,
            graftable_chunks,
        );
        (intervals, explicit.into_iter().collect())
    }

    #[test]
    fn coverage_spans_the_scan_and_every_changed_chunk() {
        // The scan [10, 30) cuts chunks 1 and 3, so both need bytes. The superseded record at
        // 50 and the terminal at 63 need theirs. Touching intervals stay separate.
        assert_eq!(
            required(64, 10, 30, &[50], 8),
            (vec![(1, 4), (6, 7), (7, 8)], vec![1, 3, 6, 7])
        );
        // An aligned scan needs no boundary bytes; a change inside it merges into it.
        assert_eq!(
            required(64, 8, 24, &[12], 8),
            (vec![(1, 3), (7, 8)], vec![1, 7])
        );
        // Without a scan, the pending chunk 7 and the partial chunk 8 still need bytes,
        // since appends may graft or complete them.
        assert_eq!(
            required(70, 60, 60, &[], 7),
            (vec![(7, 8), (8, 9)], vec![7, 8])
        );
    }

    #[test]
    fn a_view_reads_only_covered_bits() {
        let view = VerifiedBitmapView::<F, <Sha256 as Hasher>::Digest, 1> {
            floor_scan_end: Location::new(32),
            explicit_chunks: BTreeMap::from([(0, [0b0000_0101]), (3, [0b1000_0000])]),
            zero_ranges: vec![(1, 3)],
            raw_nodes: BTreeMap::new(),
            grafted_nodes: BTreeMap::new(),
        };
        let bit = |loc| view.bit(Location::new(loc));
        assert_eq!(
            [0, 1, 2, 8, 23, 24, 31].map(bit),
            [true, false, true, false, false, false, true].map(Some)
        );
        // Chunk 4 is not covered, so its bits are unknown rather than clear.
        assert_eq!(bit(32), None);
        let set = |start, end| {
            view.set_bits(Location::new(start), Location::new(end))
                .map(|loc| *loc)
                .collect::<Vec<_>>()
        };
        assert_eq!(set(0, 32), vec![0, 2, 31]);
        assert_eq!(set(1, 31), vec![2]);
        assert_eq!(set(3, 31), Vec::<u64>::new());
    }

    #[test]
    fn chunk_bits_set_and_clear_in_place() {
        let mut chunk = [0u8; 2];
        set_chunk_bit(&mut chunk, 9, true).unwrap();
        // Locations index the whole bitmap, so 31 is bit 15 of its 16-bit chunk.
        set_chunk_bit(&mut chunk, 16 + 15, true).unwrap();
        assert_eq!(chunk, [0, 0b1000_0010]);
        set_chunk_bit(&mut chunk, 9, false).unwrap();
        assert_eq!(chunk, [0, 0b1000_0000]);
    }

    /// The prover's split of a clear run: aligned blocks that exist, covering the run exactly,
    /// with no two neighbors it could have proven as one.
    #[cfg(feature = "std")]
    fn check_zero_blocks<Fam: Graftable>(leaves: u64, height: u32) {
        let chunks = grafting::graftable_chunks::<Fam>(leaves, height);
        for first in 0..chunks {
            for end in first + 1..=chunks {
                let blocks = zero_blocks::<Fam>(first, end, leaves, height);
                let mut cursor = first;
                for &(start, chunk_height) in &blocks {
                    assert_eq!(start, cursor);
                    assert!(start.is_multiple_of(1 << chunk_height));
                    assert!(
                        start >> chunk_height
                            < grafting::graftable_chunks::<Fam>(leaves, height + chunk_height),
                        "{start}/{chunk_height} does not exist at {leaves}"
                    );
                    cursor += 1 << chunk_height;
                }
                assert_eq!(cursor, end);
                for pair in blocks.windows(2) {
                    let ((left, left_height), (_, right_height)) = (pair[0], pair[1]);
                    let mergeable = left_height == right_height
                        && left.is_multiple_of(2 << left_height)
                        && left >> (left_height + 1)
                            < grafting::graftable_chunks::<Fam>(leaves, height + left_height + 1);
                    assert!(!mergeable, "{pair:?} could be one block at {leaves}");
                }
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn clear_runs_split_into_the_fewest_blocks_that_exist() {
        assert_eq!(
            zero_blocks::<F>(3, 12, 1 << 20, 1),
            vec![(3, 0), (4, 2), (8, 2)]
        );
        for leaves in [64, 100, 127, 128, 200, 255, 256, 300] {
            check_zero_blocks::<mmr::Family>(leaves, 2);
            check_zero_blocks::<crate::merkle::mmb::Family>(leaves, 2);
        }
    }
}
