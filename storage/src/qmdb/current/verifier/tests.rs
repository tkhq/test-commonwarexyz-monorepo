//! End-to-end checks of proof construction and verification against the database writer.

use super::{
    BatchProofBundle, BitmapFault, BitmapProof, BitmapUnit, BuildError, BundleConfig,
    CandidateFault, CurrentCommitment, InsufficientPreState, PreStateKeyClaim, PreStateMemberClaim,
    PreStateOpeningClaim, TrustedMap, UserBatch, UserOp, VerifyError,
    subtree::{SubtreeTarget, create_multi_subtree_proof},
};
use crate::{
    journal::contiguous::fixed::Config as JournalConfig,
    merkle::{Graftable, Location, full::Config as MerkleConfig, hasher::Hasher as _, mmb, mmr},
    qmdb::{
        self,
        any::{operation::Operation, value::FixedEncoding},
        current::{FixedConfig, grafting, ordered::fixed::Db},
        floor::{Bounded, Decision, Entry, Hold, Limits, Policy, Proportional},
    },
    translator::OneCap,
};
use commonware_codec::{Decode as _, Encode as _};
use commonware_cryptography::{Hasher, Sha256};
use commonware_parallel::{Rayon, Sequential, Strategy, mocks::Explorer};
use commonware_runtime::{Runner as _, Strategizer as _, buffer::paged::CacheRef, deterministic};
use commonware_utils::{NZU16, NZU64, NZUsize, sequence::FixedBytes};
use core::{
    fmt::Debug,
    ops::{Bound, RangeBounds as _},
};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

type Key = FixedBytes<8>;
type Commitment<F> = CurrentCommitment<F, Key, FixedEncoding<Key>, Sha256, 32>;
type Bundle<F> = BatchProofBundle<F, Key, FixedEncoding<Key>, Sha256, 32>;
type Digest = <Sha256 as Hasher>::Digest;
type Unit<F> = BitmapUnit<F, Digest, 32>;
type Batch = UserBatch<Key, Key>;
type TestDb<F, S> = Db<F, deterministic::Context, Key, Key, Sha256, OneCap, 32, S>;
/// Applies a forgery to a bundle's bitmap units.
type ChangeUnits<'a, F> = &'a dyn Fn(&mut Vec<Unit<F>>);
/// Forges a bundle's bitmap units and requires one of the given errors.
type ForgeUnits<'a, F> = &'a dyn Fn(&str, &[VerifyError], ChangeUnits<'_, F>);

const CHUNK_BITS: u64 = 256;

std::thread_local! {
    /// The forgeries [`rejects`] has tried on this thread.
    static FORGED: RefCell<BTreeSet<String>> = const { RefCell::new(BTreeSet::new()) };
}

fn key(value: u64) -> Key {
    Key::from(value.to_be_bytes())
}

fn index(key: &Key) -> u64 {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(key.as_ref());
    u64::from_be_bytes(bytes)
}

/// `value` with one bit flipped, so a forgery always changes it.
fn flipped(value: &Key) -> Key {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(value.as_ref());
    bytes[0] ^= 1;
    Key::new(bytes)
}

/// Hand-written edge cases followed by seeded batches of sets, deletes and range deletes.
fn batches() -> Vec<Batch> {
    let mut batches = vec![UserBatch::new()]; // Empty metadata-only transition.
    let mut insert = UserBatch::new();
    for i in 0..600 {
        insert.set(key(i), key(i + 1));
    }
    batches.push(insert); // Cross multiple bitmap chunks.
    let mut remove = UserBatch::new();
    remove.delete_range(key(10)..=key(590)).unwrap();
    remove.set(key(100), key(900)); // Split the range; this key survives.
    remove.set(key(200), key(901));
    remove.delete(key(200)); // Last wins: join this gap again.
    remove.set_metadata(key(42));
    batches.push(remove);
    batches.push(UserBatch::new()); // Advance the floor without user mutations.
    let mut empty = UserBatch::new();
    empty.delete_range(key(0)..=key(1000)).unwrap();
    batches.push(empty);
    batches.push(UserBatch::from([UserOp::set(key(7), key(8))])); // Repopulate an empty state.
    // The only key's successor wraps to itself, so a range below it deletes nothing.
    batches.push(UserBatch::from([UserOp::delete_range(..key(7))]));
    for round in 0..64 {
        let mut batch = UserBatch::new();
        for step in 0..16 {
            let start = (round * 37 + step * 13) % 640;
            match (round + step) % 4 {
                0 => batch.delete_range(key(start)..=key(start + 20)).unwrap(),
                1 => batch.delete(key(start)),
                _ => batch.set(key(start), key(round)),
            }
        }
        batches.push(batch);
    }
    // Ranges open at one end or both: below every key, the least keys, the greatest keys, all
    // but a set, and everything, twice, so the last starts from an empty database.
    batches.push(UserBatch::from([UserOp::delete_range(..key(0))]));
    batches.push(UserBatch::from([UserOp::delete_range(..=key(40))]));
    batches.push(UserBatch::from([UserOp::delete_range((
        Bound::Excluded(key(600)),
        Bound::Unbounded,
    ))]));
    batches.push(UserBatch::from([
        UserOp::delete_range(..),
        UserOp::set(key(300), key(1)),
    ]));
    batches.push(UserBatch::from([UserOp::delete_range(..)]));
    batches.push(UserBatch::from([UserOp::delete_range(..)]));
    batches
}

/// Setup batches that leave a long run of clear chunks above the floor, and batches whose floor
/// scans then cross it.
///
/// Deleting 2,000 of 6,500 keys appends 2,000 inactive deletes in a row. That batch's floor walk
/// runs out of entries before reaching them, about 500 live records short of its old log's end.
/// Updating 600 keys then gives the next walk enough entries to move those records and cross the
/// deletes, about seven clear chunks, which no batch since has touched.
fn clear_run_batches() -> (Vec<Batch>, Vec<Batch>) {
    let mut fill = UserBatch::new();
    for i in 0..6_500 {
        fill.set(key(i), key(i));
    }
    let mut clear = UserBatch::new();
    clear.delete_range(key(2_000)..=key(3_999)).unwrap();
    let mut cross = UserBatch::new();
    for i in 0..600 {
        cross.set(key(i), key(i + 1));
    }
    (vec![fill, clear], vec![cross, UserBatch::new()])
}

/// Apply `batch` to the model of the database's keys and values.
fn apply_model(expected: &mut BTreeMap<Key, Key>, batch: &Batch) {
    for operation in batch.iter() {
        match operation {
            UserOp::Set { key, value } => {
                expected.insert(key.clone(), value.clone());
            }
            UserOp::Delete { key } => {
                expected.remove(key);
            }
            UserOp::DeleteRange { start, end } => {
                expected.retain(|key, _| !(start.as_ref(), end.as_ref()).contains(key));
            }
        }
    }
}

/// One more than the largest key any of `batches` names.
fn keyspace(batches: &[Batch]) -> u64 {
    batches
        .iter()
        .flat_map(|batch| batch.iter())
        .flat_map(|op| match op {
            UserOp::Set { key, .. } | UserOp::Delete { key } => vec![key],
            UserOp::DeleteRange { start, end } => [start, end]
                .into_iter()
                .filter_map(|bound| match bound {
                    Bound::Included(key) | Bound::Excluded(key) => Some(key),
                    Bound::Unbounded => None,
                })
                .collect(),
        })
        .map(index)
        .max()
        .map_or(0, |largest| largest + 1)
}

/// Assert that verifying `bundle`, forged as `forgery` describes, fails sequentially and in
/// parallel with the same error, one of `expected`.
fn rejects<F: Graftable, P: Policy<F, Key, Key> + Clone + Debug>(
    forgery: &str,
    expected: &[VerifyError],
    batch: &Batch,
    pre: Commitment<F>,
    bundle: &Bundle<F>,
    post: Commitment<F>,
    policy: &P,
) {
    FORGED.with(|forged| forged.borrow_mut().insert(forgery.to_owned()));
    let strategy = Explorer::parallel(&Rayon::new(NZUsize!(4)).unwrap());
    let cache = TrustedMap::empty_cache(pre);
    let Err(serial) = batch.verify(pre, cache.clone(), bundle, post, &mut policy.clone()) else {
        panic!("accepted {forgery} under {policy:?}");
    };
    let Err(parallel) =
        batch.verify_with_strategy(pre, cache, bundle, post, &strategy, &mut policy.clone())
    else {
        panic!("accepted {forgery} under {policy:?} in parallel");
    };
    assert_eq!(serial, parallel, "{forgery} under {policy:?}");
    assert!(
        expected.contains(&serial),
        "{forgery} under {policy:?} failed with {serial:?}, not one of {expected:?}"
    );
}

/// Replace the bundle's bitmap units with `change`'s result and require rejection.
#[allow(clippy::too_many_arguments)]
fn forge_units<F: Graftable, P: Policy<F, Key, Key> + Clone + Debug>(
    forgery: &str,
    expected: &[VerifyError],
    batch: &Batch,
    pre: Commitment<F>,
    proof: &Bundle<F>,
    post: Commitment<F>,
    policy: &P,
    change: impl FnOnce(&mut Vec<Unit<F>>),
) {
    let mut forged = proof.clone();
    if let BitmapProof::Proven(bitmap) = &mut forged.bitmap {
        change(&mut bitmap.units);
    }
    rejects(forgery, expected, batch, pre, &forged, post, policy);
}

/// Forge each part of an honest bundle and require rejection for the reason the forgery breaks.
fn reject_forgeries<F: Graftable, P: Policy<F, Key, Key> + Clone + Debug>(
    batch: &Batch,
    pre: Commitment<F>,
    proof: &Bundle<F>,
    post: Commitment<F>,
    policy: &P,
    database_empty: bool,
) {
    use BitmapFault as B;
    use VerifyError::{InvalidBitmapProof as Bitmap, InvalidCandidates as Candidates};

    let mut forged = proof.clone();
    forged.claims.post_floor += 1;
    rejects(
        "a moved post floor",
        &[VerifyError::ClaimedBoundsMismatch],
        batch,
        pre,
        &forged,
        post,
        policy,
    );
    let mut forged = proof.clone();
    match &mut forged.bitmap {
        BitmapProof::Initial { floor_scan_end } => *floor_scan_end += 1,
        BitmapProof::Proven(bitmap) => bitmap.floor_scan_end += 1,
    }
    // Moving the end can leave the log, change which chunks are needed, expose another live
    // record, or simply disagree with where the floor walk stopped.
    rejects(
        "a moved scan end",
        &[
            Bitmap(B::ScanBounds),
            Bitmap(B::Coverage),
            Bitmap(B::NeedsBytes),
            Bitmap(B::ScanEnd),
            Candidates(CandidateFault::Locations),
            Candidates(CandidateFault::EmptiedScan),
        ],
        batch,
        pre,
        &forged,
        post,
        policy,
    );
    // Range expansion runs first, and a range open below needs the record wrapping to the
    // least key before anything is classified.
    let open_below = matches!(batch.first().map(UserOp::start), Some(Bound::Unbounded));
    if !database_empty && !batch.is_empty() {
        let mut forged = proof.clone();
        forged.pre_state_claims.clear();
        let missing = if open_below {
            InsufficientPreState::OrderedEdge
        } else {
            InsufficientPreState::Classification
        };
        rejects(
            "no key claims",
            &[VerifyError::InsufficientPreState(missing)],
            batch,
            pre,
            &forged,
            post,
            policy,
        );
    }
    if !database_empty && open_below {
        use crate::qmdb::current::ordered::proof::constant::ExclusionProof;

        // Drop every claim carrying the greatest key's record, which wraps to the least.
        let mut forged = proof.clone();
        forged.pre_state_claims.retain(|claim| match claim {
            PreStateKeyClaim::Member(member) => member.proof.next_key > member.key,
            PreStateKeyClaim::Exclusion(exclusion) => !matches!(&exclusion.proof,
                ExclusionProof::KeyValue(_, update) if update.next_key <= update.key),
        });
        rejects(
            "a missing wrapping record",
            &[VerifyError::InsufficientPreState(
                InsufficientPreState::OrderedEdge,
            )],
            batch,
            pre,
            &forged,
            post,
            policy,
        );
    }
    if !proof.pre_state_candidates.operations.is_empty() {
        let mut forged = proof.clone();
        forged.pre_state_candidates.operations.remove(0);
        rejects(
            "a missing candidate",
            &[Candidates(CandidateFault::Locations)],
            batch,
            pre,
            &forged,
            post,
            policy,
        );
        let mut forged = proof.clone();
        if let Operation::Update(update) = &mut forged.pre_state_candidates.operations[0].1 {
            update.value = flipped(&update.value);
        }
        rejects(
            "a changed candidate value",
            &[Candidates(CandidateFault::Proof)],
            batch,
            pre,
            &forged,
            post,
            policy,
        );
    }
    if let Some(index) = proof
        .pre_state_claims
        .iter()
        .position(|claim| matches!(claim, PreStateKeyClaim::Member(_)))
    {
        let mut forged = proof.clone();
        if let PreStateKeyClaim::Member(member) = &mut forged.pre_state_claims[index] {
            member.value = flipped(&member.value);
        }
        rejects(
            "a changed member value",
            &[VerifyError::InvalidMemberProof],
            batch,
            pre,
            &forged,
            post,
            policy,
        );
        let mut forged = proof.clone();
        if let PreStateKeyClaim::Member(member) = &mut forged.pre_state_claims[index] {
            member.proof.next_key = flipped(&member.proof.next_key);
        }
        rejects(
            "a changed member successor",
            &[VerifyError::InvalidMemberProof],
            batch,
            pre,
            &forged,
            post,
            policy,
        );
    }
    let BitmapProof::Proven(bitmap) = &proof.bitmap else {
        return;
    };
    let units = &bitmap.units;
    if units.is_empty() {
        return;
    }
    let forge = |forgery: &str, expected: &[VerifyError], change: ChangeUnits<'_, F>| {
        forge_units(forgery, expected, batch, pre, proof, post, policy, change);
    };
    let coverage = [Bitmap(B::Coverage)];
    forge("a missing bitmap unit", &coverage, &|units| {
        units.remove(0);
    });
    forge("a repeated bitmap unit", &coverage, &|units| {
        units.insert(0, units[0].clone())
    });
    forge("an extra bitmap unit", &coverage, &|units| {
        let mut extra = units[units.len() - 1].clone();
        if let BitmapUnit::ExplicitChunk { chunk_index, .. } = &mut extra {
            *chunk_index += 1;
        }
        units.push(extra);
    });
    if units.len() > 1 {
        forge("reordered bitmap units", &coverage, &|units| {
            units.swap(0, 1)
        });
    }
    if let BitmapUnit::ExplicitChunk {
        raw_subtree_digest, ..
    } = &units[0]
    {
        // The grafted tree binds a graftable chunk's bytes; a digest binds a pending or partial
        // chunk's.
        let expected: &[VerifyError] = if raw_subtree_digest.is_some() {
            &[Bitmap(B::GraftedRoot)]
        } else {
            &[Bitmap(B::PendingChunk), Bitmap(B::PartialChunk)]
        };
        forge("a changed bitmap chunk", expected, &|units| {
            if let BitmapUnit::ExplicitChunk { bytes, .. } = &mut units[0] {
                bytes[0] ^= 1;
            }
        });
    }
    if let Some(index) = units.iter().position(|unit| {
        matches!(unit, BitmapUnit::ExplicitChunk { bytes, raw_subtree_digest: Some(_), .. }
            if *bytes != [0; 32])
    }) {
        // A chunk the transition reads is refused outright; any other fails the grafted root.
        forge(
            "a set chunk claimed clear",
            &[Bitmap(B::NeedsBytes), Bitmap(B::GraftedRoot)],
            &|units| {
                if let BitmapUnit::ExplicitChunk {
                    chunk_index,
                    raw_subtree_digest: Some(digest),
                    ..
                } = units[index]
                {
                    units[index] = BitmapUnit::zero_subtree(chunk_index, 0, digest);
                }
            },
        );
    }
    // The old terminal's chunk always needs its bytes.
    let terminal_chunk = (*pre.size() - 1) / CHUNK_BITS;
    if let Some(index) = units.iter().position(|unit| {
        matches!(unit, BitmapUnit::ExplicitChunk { chunk_index, .. } if *chunk_index == terminal_chunk)
    }) && let BitmapUnit::ExplicitChunk {
        raw_subtree_digest, ..
    } = units[index]
    {
        let expected = if raw_subtree_digest.is_some() {
            B::NeedsBytes
        } else {
            B::Ungraftable
        };
        forge(
            "the terminal's chunk claimed clear",
            &[Bitmap(expected)],
            &|units| {
                let digest =
                    raw_subtree_digest.unwrap_or_else(|| Sha256::hash(&[b"clear".as_slice()]));
                units[index] = BitmapUnit::zero_subtree(terminal_chunk, 0, digest);
            },
        );
    }
    reject_zero_subtree_forgeries::<F>(units, &forge);
}

/// Forge the zero subtrees an honest bundle carries.
fn reject_zero_subtree_forgeries<F: Graftable>(units: &[Unit<F>], forge: ForgeUnits<'_, F>) {
    use BitmapFault as B;
    use VerifyError::InvalidBitmapProof as Bitmap;

    let zero = |unit: &Unit<F>| match unit {
        BitmapUnit::ZeroSubtree {
            first_chunk,
            chunk_height,
            common_subtree_digest,
            ..
        } => Some((*first_chunk, *chunk_height, *common_subtree_digest)),
        BitmapUnit::ExplicitChunk { .. } => None,
    };
    let Some(first) = units.iter().position(|unit| zero(unit).is_some()) else {
        return;
    };
    forge("another block's digest", &[Bitmap(B::RawRoot)], &|units| {
        if let BitmapUnit::ZeroSubtree {
            common_subtree_digest,
            ..
        } = &mut units[first]
        {
            common_subtree_digest.0[0] ^= 1;
        }
    });
    // Doubling a block either breaks its alignment, overruns what it may cover, or leaves the
    // next unit out of place.
    forge(
        "a grown zero subtree",
        &[
            Bitmap(B::Coverage),
            Bitmap(B::Misaligned),
            Bitmap(B::Ungraftable),
            Bitmap(B::NeedsBytes),
        ],
        &|units| {
            if let BitmapUnit::ZeroSubtree { chunk_height, .. } = &mut units[first] {
                *chunk_height += 1;
            }
        },
    );
    // A block of four chunks or more holds a misaligned pair at its second chunk. The digests
    // do not matter: alignment is checked before either tree is.
    if let Some((index, (first_chunk, _, digest))) =
        units.iter().enumerate().find_map(|(index, unit)| {
            zero(unit)
                .filter(|(_, height, _)| *height >= 2)
                .map(|zero| (index, zero))
        })
    {
        forge(
            "a misaligned zero subtree",
            &[Bitmap(B::Misaligned)],
            &|units| {
                units.splice(
                    index..=index,
                    [
                        BitmapUnit::zero_subtree(first_chunk, 0, digest),
                        BitmapUnit::zero_subtree(first_chunk + 1, 1, digest),
                    ],
                );
            },
        );
    }
    // The prover splits an aligned pair of equal adjacent blocks only when the tree has not yet
    // formed their parent, as in a Merkle mountain belt.
    for pair in 0..units.len().saturating_sub(1) {
        let (Some((left, height, left_digest)), Some((right, right_height, right_digest))) =
            (zero(&units[pair]), zero(&units[pair + 1]))
        else {
            continue;
        };
        if height == right_height
            && right == left + (1 << height)
            && left.is_multiple_of(2 << height)
        {
            let tree_height = grafting::height::<32>() + height + 1;
            let position = F::subtree_root_position(Location::new(left * CHUNK_BITS), tree_height);
            let parent =
                qmdb::hasher::<Sha256>().node_digest(position, &left_digest, &right_digest);
            forge("an unformed parent", &[Bitmap(B::Geometry)], &|units| {
                units[pair] = BitmapUnit::zero_subtree(left, height + 1, parent);
                units.remove(pair + 1);
            });
        }
    }
}

/// `proof` with every zero subtree replaced by the clear chunks it covers, sent in full.
async fn without_zero_subtrees<F: Graftable, S: Strategy>(
    db: &TestDb<F, S>,
    proof: &Bundle<F>,
) -> Bundle<F> {
    let mut proof = proof.clone();
    let old_floor = proof.claims.pre_floor;
    let BitmapProof::Proven(bitmap) = &mut proof.bitmap else {
        return proof;
    };
    let height = grafting::height::<32>();
    let hasher = qmdb::hasher::<Sha256>();
    let (mut units, mut raw_targets, mut grafted_targets) = (Vec::new(), Vec::new(), Vec::new());
    for unit in &bitmap.units {
        let chunks = match unit {
            BitmapUnit::ExplicitChunk {
                chunk_index,
                bytes,
                raw_subtree_digest,
            } => {
                units.push(unit.clone());
                if let Some(digest) = raw_subtree_digest {
                    let leaf_start = Location::new(chunk_index * CHUNK_BITS);
                    raw_targets.push(SubtreeTarget {
                        leaf_start,
                        height,
                        digest: *digest,
                    });
                    let grafted = if *bytes == [0; 32] {
                        *digest
                    } else {
                        hasher.hash(&[bytes.as_slice(), digest.as_ref()])
                    };
                    grafted_targets.push(SubtreeTarget {
                        leaf_start,
                        height,
                        digest: grafted,
                    });
                }
                continue;
            }
            BitmapUnit::ZeroSubtree {
                first_chunk,
                chunk_height,
                ..
            } => *first_chunk..first_chunk + (1 << chunk_height),
        };
        for chunk_index in chunks {
            let leaf_start = Location::new(chunk_index * CHUNK_BITS);
            let digest = crate::merkle::storage::Storage::get_node(
                &db.any.log.merkle,
                F::subtree_root_position(leaf_start, height),
            )
            .await
            .unwrap()
            .unwrap();
            units.push(BitmapUnit::ExplicitChunk {
                chunk_index,
                bytes: [0; 32],
                raw_subtree_digest: Some(digest),
            });
            let target = SubtreeTarget {
                leaf_start,
                height,
                digest,
            };
            raw_targets.push(target);
            grafted_targets.push(target);
        }
    }
    let pre_size = bitmap.pre_size;
    bitmap.units = units;
    if raw_targets.is_empty() {
        return proof;
    }
    bitmap.raw_proof = Some(
        create_multi_subtree_proof(
            &db.any.log.merkle,
            F::inactive_peaks(pre_size, old_floor),
            &raw_targets,
        )
        .await
        .unwrap(),
    );
    bitmap.grafted_proof = Some(
        create_multi_subtree_proof(
            &db.grafted_storage(),
            grafting::chunk_aligned_inactive_peaks::<F>(pre_size, old_floor, height).unwrap(),
            &grafted_targets,
        )
        .await
        .unwrap(),
    );
    proof
}

/// The most digests any one Merkle proof, pinned-node list, or append frontier in `proof` holds.
fn max_digests<F: Graftable>(proof: &Bundle<F>) -> usize {
    use crate::qmdb::current::ordered::proof::constant::ExclusionProof;

    let mut most = 0;
    if let PreStateOpeningClaim::Proven { terminal, .. } = &proof.pre_state {
        most = terminal
            .raw_proof
            .digests
            .len()
            .max(terminal.pins_before_terminal.len());
    }
    for claim in &proof.pre_state_claims {
        let operation_proof = match claim {
            PreStateKeyClaim::Member(member) => &member.proof.proof,
            PreStateKeyClaim::Exclusion(exclusion) => match &exclusion.proof {
                ExclusionProof::KeyValue(proof, _) | ExclusionProof::Commit(proof, _) => proof,
            },
        };
        most = most.max(operation_proof.range_proof.proof.digests.len());
    }
    if let BitmapProof::Proven(bitmap) = &proof.bitmap {
        for merkle_proof in bitmap.raw_proof.iter().chain(&bitmap.grafted_proof) {
            most = most.max(merkle_proof.digests.len());
        }
        most = most.max(bitmap.grafted_append_frontier.len());
    }
    if let Some(raw) = &proof.pre_state_candidates.raw_multi_proof {
        most = most.max(raw.digests.len());
    }
    most
}

/// Encode `proof`, require that it decodes to the same bytes under limits it exactly meets and
/// that each limit one smaller refuses it, and return the decoded bundle.
fn round_trip<F: Graftable>(proof: &Bundle<F>) -> Bundle<F> {
    let units = match &proof.bitmap {
        BitmapProof::Initial { .. } => 0,
        BitmapProof::Proven(bitmap) => bitmap.units.len(),
    };
    let exact = BundleConfig {
        max_claims: proof.pre_state_claims.len(),
        max_candidates: proof.pre_state_candidates.operations.len(),
        max_bitmap_units: units,
        max_digests: max_digests(proof),
        key: (),
        update: (),
        operation: (),
        value: (),
    };
    let encoded = proof.encode();
    let decoded = Bundle::<F>::decode_cfg(encoded.clone(), &exact).unwrap();
    assert_eq!(decoded.encode(), encoded);
    let limits = [
        exact.max_claims,
        exact.max_candidates,
        exact.max_bitmap_units,
        exact.max_digests,
    ];
    let tighter = [
        BundleConfig {
            max_claims: exact.max_claims.wrapping_sub(1),
            ..exact.clone()
        },
        BundleConfig {
            max_candidates: exact.max_candidates.wrapping_sub(1),
            ..exact.clone()
        },
        BundleConfig {
            max_bitmap_units: exact.max_bitmap_units.wrapping_sub(1),
            ..exact.clone()
        },
        BundleConfig {
            max_digests: exact.max_digests.wrapping_sub(1),
            ..exact
        },
    ];
    for (limit, config) in limits.into_iter().zip(tighter) {
        if limit > 0 {
            assert!(Bundle::<F>::decode_cfg(encoded.clone(), &config).is_err());
        }
    }
    decoded
}

/// Apply each of `setup` through the writer alone, then prove, merkleize, verify and apply each
/// of `batches` under `policy`.
///
/// The writer and the verifier each hold their own copy of the policy and advance it once per
/// batch; the two copies must stay equal. Every honest bundle verifies from an empty cache and
/// from the retained one, sequentially and in parallel, and so does the same bundle with its
/// zero subtrees sent as chunks; every forged one is rejected. The database must match a model
/// of the batches: its keys always, and its values when `check_values` is set, since a policy
/// may rewrite the values it moves. `probe` sees each honest transition. Returns the
/// post-commitments.
async fn run<F, S, P>(
    context: deterministic::Context,
    strategy: S,
    policy: P,
    check_values: bool,
    (setup, batches): (Vec<Batch>, Vec<Batch>),
    mut probe: impl FnMut(&Batch, Commitment<F>, &Bundle<F>, Commitment<F>),
) -> Vec<Commitment<F>>
where
    F: Graftable,
    S: Strategy,
    P: Policy<F, Key, Key> + Clone + Debug + PartialEq + Send,
{
    let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
    let config = FixedConfig {
        merkle_config: MerkleConfig {
            journal_partition: "verifier-tests-nodes".into(),
            replay_buffer: NZUsize!(4096),
            metadata_partition: "verifier-tests-merkle-metadata".into(),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            strategy: strategy.clone(),
            page_cache: page_cache.clone(),
        },
        journal_config: JournalConfig {
            partition: "verifier-tests-operations".into(),
            replay_buffer: NZUsize!(4096),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            page_cache,
        },
        grafted_metadata_partition: "verifier-tests-grafted".into(),
        translator: OneCap,
        init_cache: Some(NZUsize!(1024)),
        init_buffer: NZUsize!(4096),
        init_concurrency: (),
    };
    let mut db = TestDb::<F, S>::init(context, config, None).await.unwrap();
    let keys = keyspace(&setup).max(keyspace(&batches));
    let verifier_strategy = Explorer::parallel(&Rayon::new(NZUsize!(4)).unwrap());
    let mut writer_policy = policy.clone();
    let mut verifier_policy = policy;
    let mut expected = BTreeMap::new();
    for batch in setup {
        let merkleized = batch.merkleize(&db, &mut writer_policy).await.unwrap();
        db = db.apply_batch(merkleized).await.unwrap();
        apply_model(&mut expected, &batch);
    }
    verifier_policy.clone_from(&writer_policy);
    let mut pre = Commitment::<F>::from_parts(db.bounds().end, db.root()).unwrap();
    let mut cache = TrustedMap::empty_cache(pre);
    let mut commitments = Vec::new();
    for batch in batches {
        let proof = batch.build_batch_proof(&db, &writer_policy).await.unwrap();
        let merkleized = batch.merkleize(&db, &mut writer_policy).await.unwrap();
        let post =
            Commitment::<F>::from_parts(merkleized.bounds().tip.size, merkleized.root()).unwrap();
        probe(&batch, pre, &proof, post);
        reject_forgeries(&batch, pre, &proof, post, &verifier_policy, db.is_empty());

        // A recipient with no cache must also be able to verify this complete bundle, sent as
        // bytes, and it need not carry the facts forward.
        let before = verifier_policy.clone();
        let received = round_trip(&proof);
        let fresh = batch
            .verify(
                pre,
                TrustedMap::empty_cache(pre),
                &received,
                post,
                &mut before.clone(),
            )
            .unwrap();
        assert_eq!(fresh.commitment(), &post);
        // The parent facts hold every record the bundle's member claims authenticated.
        for claim in &proof.pre_state_claims {
            if let PreStateKeyClaim::Member(member) = claim {
                use super::OptionTrusted::KnownTrusted;
                let entry = fresh.parent().entry(&member.key).unwrap();
                assert_eq!(entry.location, KnownTrusted(member.proof.proof.loc));
                assert_eq!(entry.value, KnownTrusted(member.value.clone()));
            }
        }
        // Clear chunks sent in full prove the same transition.
        let explicit = without_zero_subtrees(&db, &proof).await;
        let fresh_explicit = batch
            .verify(
                pre,
                TrustedMap::empty_cache(pre),
                &explicit,
                post,
                &mut before.clone(),
            )
            .unwrap();
        assert_eq!(
            format!("{:?}", fresh.into_cache().unwrap()),
            format!("{:?}", fresh_explicit.into_cache().unwrap())
        );
        let serial = batch
            .verify(pre, cache.clone(), &proof, post, &mut before.clone())
            .unwrap()
            .into_cache()
            .unwrap();
        cache = batch
            .verify_with_strategy(
                pre,
                cache,
                &proof,
                post,
                &verifier_strategy,
                &mut verifier_policy,
            )
            .unwrap()
            .into_cache()
            .unwrap();
        assert_eq!(format!("{serial:?}"), format!("{cache:?}"));
        assert_eq!(cache.commitment(), &post);
        assert_eq!(writer_policy, verifier_policy);
        assert_eq!(
            proof.claims.post_floor,
            merkleized.bounds().inactivity_floor
        );

        db = db.apply_batch(merkleized).await.unwrap();
        assert_eq!(db.root(), *post.root());
        apply_model(&mut expected, &batch);
        for i in 0..keys {
            let (actual, wanted) = (db.get(&key(i)).await.unwrap(), expected.get(&key(i)));
            if check_values {
                assert_eq!(actual.as_ref(), wanted);
            } else {
                assert_eq!(actual.is_some(), wanted.is_some());
            }
        }
        assert_eq!(db.is_empty(), expected.is_empty());
        // Proof production must work after obsolete log and bitmap chunks are pruned.
        db = db.sync().await.unwrap();
        let boundary = db.sync_boundary();
        db = db.prune(boundary).await.unwrap();
        commitments.push(post);
        pre = post;
    }
    db.destroy().await.unwrap();
    commitments
}

/// Keeps updates, but stops the walk at every `period`th one it is asked about, counting across
/// batches, so the writer and the verifier must advance its state alike.
#[derive(Clone, Debug, PartialEq)]
struct StopEvery {
    period: usize,
    seen: usize,
}

impl<F: Graftable> Policy<F, Key, Key> for StopEvery {
    fn evicts(&self) -> bool {
        false
    }

    fn limits(&self, made_inactive: usize) -> Limits {
        Limits {
            entries: made_inactive + 1,
            skips: u64::MAX,
        }
    }

    fn decide<'a>(&mut self, entry: Entry<'a, F, Key, Key>) -> Decision<'a, Key> {
        self.seen += 1;
        if self.seen.is_multiple_of(self.period) {
            entry.stop()
        } else {
            entry.keep()
        }
    }
}

/// Moves updates in proportion, writing a fixed value for each.
#[derive(Clone, Debug, PartialEq)]
struct Replace;

impl<F: Graftable> Policy<F, Key, Key> for Replace {
    fn evicts(&self) -> bool {
        false
    }

    fn limits(&self, made_inactive: usize) -> Limits {
        Limits {
            entries: made_inactive + 1,
            skips: u64::MAX,
        }
    }

    fn decide<'a>(&mut self, entry: Entry<'a, F, Key, Key>) -> Decision<'a, Key> {
        entry.replace(key(u64::MAX))
    }
}

/// Evicts every update it reaches, which a verifier cannot replay from the scan alone.
#[derive(Clone)]
struct Evict;

impl<F: Graftable> Policy<F, Key, Key> for Evict {
    fn evicts(&self) -> bool {
        true
    }

    fn limits(&self, made_inactive: usize) -> Limits {
        Limits {
            entries: made_inactive + 1,
            skips: u64::MAX,
        }
    }

    fn decide<'a>(&mut self, entry: Entry<'a, F, Key, Key>) -> Decision<'a, Key> {
        entry.evict().0
    }
}

fn sequential_and_parallel_agree<F: Graftable>() {
    FORGED.with(|forged| forged.borrow_mut().clear());
    let serial = deterministic::Runner::default().start(|context| {
        run::<F, _, _>(
            context,
            Sequential,
            Proportional,
            true,
            (Vec::new(), batches()),
            |_, _, _, _| {},
        )
    });
    // The deterministic runtime supplies its own Rayon pool for async database jobs.
    // Verification separately uses a real four-worker pool and synchronous dispatch.
    let parallel = deterministic::Runner::default().start(|context| {
        let strategy = Explorer::parallel(&context.strategy(NZUsize!(4)));
        run::<F, _, _>(
            context,
            strategy,
            Proportional,
            true,
            (Vec::new(), batches()),
            |_, _, _, _| {},
        )
    });
    assert_eq!(serial, parallel);
    let forged = FORGED.with(|forged| forged.take());
    assert!(
        forged.contains("a missing wrapping record"),
        "never tried a missing wrapping record"
    );
}

#[test]
fn mmr_sequential_and_parallel_agree() {
    sequential_and_parallel_agree::<mmr::Family>();
}

#[test]
fn mmb_sequential_and_parallel_agree() {
    sequential_and_parallel_agree::<mmb::Family>();
}

/// The verifier follows whichever supported policy the writer used.
fn follows_the_writers_policy<F: Graftable>() {
    deterministic::Runner::default().start(|context| {
        run::<F, _, _>(
            context,
            Sequential,
            Hold,
            true,
            (Vec::new(), batches()),
            |_, _, _, _| {},
        )
    });
    deterministic::Runner::default().start(|context| {
        let bounded = Bounded {
            entries: 3,
            skips: u64::MAX,
        };
        run::<F, _, _>(
            context,
            Sequential,
            bounded,
            true,
            (Vec::new(), batches()),
            |_, _, _, _| {},
        )
    });
    deterministic::Runner::default().start(|context| {
        let policy = StopEvery { period: 3, seen: 0 };
        run::<F, _, _>(
            context,
            Sequential,
            policy,
            true,
            (Vec::new(), batches()),
            |_, _, _, _| {},
        )
    });
    deterministic::Runner::default().start(|context| {
        run::<F, _, _>(
            context,
            Sequential,
            Replace,
            false,
            (Vec::new(), batches()),
            |_, _, _, _| {},
        )
    });
}

#[test]
fn mmr_follows_the_writers_policy() {
    follows_the_writers_policy::<mmr::Family>();
}

#[test]
fn mmb_follows_the_writers_policy() {
    follows_the_writers_policy::<mmb::Family>();
}

/// A bundle built under one policy must not verify under another that moves the floor
/// differently, and a policy a verifier cannot replay is refused on every batch.
fn rejects_other_policies<F: Graftable>() {
    let (differs, refused, seen) = (
        Rc::new(Cell::new(0)),
        Rc::new(Cell::new(0)),
        Rc::new(Cell::new(0)),
    );
    let probe = {
        let (differs, refused, seen) = (differs.clone(), refused.clone(), seen.clone());
        move |batch: &Batch, pre: Commitment<F>, proof: &Bundle<F>, post: Commitment<F>| {
            seen.set(seen.get() + 1);
            let held = batch.verify(pre, TrustedMap::empty_cache(pre), proof, post, &mut Hold);
            differs.set(differs.get() + usize::from(held.is_err()));
            let evicting = batch.verify(pre, TrustedMap::empty_cache(pre), proof, post, &mut Evict);
            let unsupported = matches!(evicting, Err(VerifyError::UnsupportedFloorPolicy));
            refused.set(refused.get() + usize::from(unsupported));
        }
    };
    deterministic::Runner::default().start(move |context| {
        run::<F, _, _>(
            context,
            Sequential,
            Proportional,
            true,
            (Vec::new(), batches()),
            probe,
        )
    });
    assert!(
        differs.get() > 0,
        "holding the floor never changed a verdict"
    );
    assert_eq!(refused.get(), seen.get(), "an evicting policy was accepted");
}

#[test]
fn mmr_rejects_other_policies() {
    rejects_other_policies::<mmr::Family>();
}

#[test]
fn mmb_rejects_other_policies() {
    rejects_other_policies::<mmb::Family>();
}

/// A floor scan across a long clear run is proven by a few zero subtrees, the largest spanning
/// several chunks, rather than by every chunk's bytes. `run` checks that the bundle verifies,
/// that the same bundle with those chunks sent in full proves the same transition, and that
/// forged zero subtrees are rejected.
fn proves_clear_runs_with_zero_subtrees<F: Graftable>() {
    FORGED.with(|forged| forged.borrow_mut().clear());
    let tallest = Rc::new(Cell::new(None));
    let probe = {
        let tallest = tallest.clone();
        move |_: &Batch, _: Commitment<F>, proof: &Bundle<F>, _: Commitment<F>| {
            let BitmapProof::Proven(bitmap) = &proof.bitmap else {
                return;
            };
            for unit in &bitmap.units {
                if let BitmapUnit::ZeroSubtree { chunk_height, .. } = unit {
                    tallest.set(tallest.get().max(Some(*chunk_height)));
                }
            }
        }
    };
    deterministic::Runner::default().start(move |context| {
        run::<F, _, _>(
            context,
            Sequential,
            Proportional,
            true,
            clear_run_batches(),
            probe,
        )
    });
    assert!(
        tallest.get().is_some_and(|height| height >= 2),
        "the clear run was proven by blocks no taller than {:?}",
        tallest.get()
    );
    let forged = FORGED.with(|forged| forged.take());
    for forgery in [
        "another block's digest",
        "a grown zero subtree",
        "a misaligned zero subtree",
        "the terminal's chunk claimed clear",
    ] {
        assert!(forged.contains(forgery), "never tried {forgery}");
    }
}

#[test]
fn mmr_proves_clear_runs_with_zero_subtrees() {
    proves_clear_runs_with_zero_subtrees::<mmr::Family>();
}

#[test]
fn mmb_proves_clear_runs_with_zero_subtrees() {
    proves_clear_runs_with_zero_subtrees::<mmb::Family>();
}

/// The prover refuses a policy the verifier would refuse, rather than build a bundle for it.
#[test]
fn prover_refuses_unsupported_policies() {
    deterministic::Runner::default().start(|context| async move {
        let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
        let config = FixedConfig {
            merkle_config: MerkleConfig {
                journal_partition: "verifier-refusal-nodes".into(),
                replay_buffer: NZUsize!(4096),
                metadata_partition: "verifier-refusal-merkle-metadata".into(),
                items_per_blob: NZU64!(1024),
                write_buffer: NZUsize!(4096),
                strategy: Sequential,
                page_cache: page_cache.clone(),
            },
            journal_config: JournalConfig {
                partition: "verifier-refusal-operations".into(),
                replay_buffer: NZUsize!(4096),
                items_per_blob: NZU64!(1024),
                write_buffer: NZUsize!(4096),
                page_cache,
            },
            grafted_metadata_partition: "verifier-refusal-grafted".into(),
            translator: OneCap,
            init_cache: Some(NZUsize!(1024)),
            init_buffer: NZUsize!(4096),
            init_concurrency: (),
        };
        let db = Db::<mmr::Family, _, Key, Key, Sha256, _, 32, _>::init(context, config, None)
            .await
            .unwrap();
        let batch = UserBatch::from([UserOp::set(key(1), key(2))]);
        let skip_limited = Bounded {
            entries: 1,
            skips: 0,
        };
        assert!(matches!(
            batch.build_batch_proof(&db, &Evict).await,
            Err(BuildError::UnsupportedFloorPolicy)
        ));
        assert!(matches!(
            batch.build_batch_proof(&db, &skip_limited).await,
            Err(BuildError::UnsupportedFloorPolicy)
        ));
        db.destroy().await.unwrap();
    });
}

/// A finite interval is proven by its lower endpoint's classification and every member after it.
async fn range_claims<F: Graftable>(context: deterministic::Context) {
    use crate::qmdb::current::ordered::proof::verify_range;

    let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
    let config = FixedConfig {
        merkle_config: MerkleConfig {
            journal_partition: "range-claims-nodes".into(),
            replay_buffer: NZUsize!(4096),
            metadata_partition: "range-claims-merkle-metadata".into(),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            strategy: Sequential,
            page_cache: page_cache.clone(),
        },
        journal_config: JournalConfig {
            partition: "range-claims-operations".into(),
            replay_buffer: NZUsize!(4096),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            page_cache,
        },
        grafted_metadata_partition: "range-claims-grafted".into(),
        translator: OneCap,
        init_cache: Some(NZUsize!(1024)),
        init_buffer: NZUsize!(4096),
        init_concurrency: (),
    };
    let mut db = TestDb::<F, Sequential>::init(context, config, None)
        .await
        .unwrap();
    let mut batch = Batch::new();
    for i in 1..=6 {
        batch.set(key(i * 10), key(i));
    }
    let merkleized = batch.merkleize(&db, &mut Proportional).await.unwrap();
    db = db.apply_batch(merkleized).await.unwrap();
    let commitment = Commitment::<F>::from_parts(db.bounds().end, db.root()).unwrap();

    let member = async |at: u64| {
        PreStateKeyClaim::Member(PreStateMemberClaim {
            key: key(at),
            value: key(at / 10),
            proof: db.key_value_proof(key(at)).await.unwrap(),
        })
    };
    let absent = async |at: u64| {
        PreStateKeyClaim::Exclusion(super::PreStateExclusionClaim {
            queried_key: key(at),
            proof: db.exclusion_proof(&key(at)).await.unwrap(),
        })
    };
    let (k20, k40) = (key(20), key(40));
    let included = (Bound::Included(&k20), Bound::Included(&k40));

    // Members from the lower endpoint on, with nothing missing, cover the interval.
    let exact = [member(20).await, member(30).await, member(40).await];
    assert!(verify_range::<F, _, _, Sha256, 32>(
        included,
        &exact,
        &commitment
    ));
    // Dropping the middle member leaves a gap, which no claim fills.
    let gap = [member(20).await, member(40).await];
    assert!(!verify_range::<F, _, _, Sha256, 32>(
        included,
        &gap,
        &commitment
    ));
    // The last member inside the interval cannot be omitted either.
    let short = [member(20).await, member(30).await];
    assert!(!verify_range::<F, _, _, Sha256, 32>(
        included,
        &short,
        &commitment
    ));
    // A member past the interval is not part of it.
    let long = [
        member(20).await,
        member(30).await,
        member(40).await,
        member(50).await,
    ];
    assert!(!verify_range::<F, _, _, Sha256, 32>(
        included,
        &long,
        &commitment
    ));
    // An excluded lower bound starts at the member's successor.
    let k20 = key(20);
    let excluded = (Bound::Excluded(&k20), Bound::Included(&k40));
    assert!(verify_range::<F, _, _, Sha256, 32>(
        excluded,
        &exact,
        &commitment
    ));
    // A lower endpoint that is no member is classified by the proof of its absence.
    let (k25, k45) = (key(25), key(45));
    let between = (Bound::Included(&k25), Bound::Included(&k45));
    let from_absent = [absent(25).await, member(30).await, member(40).await];
    assert!(verify_range::<F, _, _, Sha256, 32>(
        between,
        &from_absent,
        &commitment
    ));
    // Claims bind to the commitment they are checked against.
    let other = Commitment::<F>::from_parts(
        commitment.size(),
        Sha256::hash(&[b"another root".as_slice()]),
    )
    .unwrap();
    assert!(!verify_range::<F, _, _, Sha256, 32>(
        included, &exact, &other
    ));
    // An interval must be finite and ordered.
    let reversed = (Bound::Included(&k40), Bound::Included(&k20));
    assert!(!verify_range::<F, _, _, Sha256, 32>(
        reversed,
        &exact,
        &commitment
    ));
    let unbounded = (Bound::Unbounded, Bound::Included(&k40));
    assert!(!verify_range::<F, _, _, Sha256, 32>(
        unbounded,
        &exact,
        &commitment
    ));
}

#[test]
fn mmr_range_claims_cover_an_interval() {
    deterministic::Runner::default().start(range_claims::<mmr::Family>);
}

#[test]
fn mmb_range_claims_cover_an_interval() {
    deterministic::Runner::default().start(range_claims::<mmb::Family>);
}

fn assert_send<T: Send>(_: T) {}

/// The variable-size database the `Send` check covers beside the fixed one.
type VariableDb<F, S> = crate::qmdb::current::ordered::variable::Db<
    F,
    deterministic::Context,
    Key,
    Vec<u8>,
    Sha256,
    OneCap,
    32,
    S,
>;

// Callers spawn proof construction, so its futures must be `Send` for every family, strategy
// and policy the prover accepts, including a policy that is not `Sync`, with fixed and
// variable values.
#[allow(dead_code)]
fn assert_prover_futures_are_send<F: Graftable, S: Strategy, P>(
    fixed: &TestDb<F, S>,
    variable: &VariableDb<F, S>,
    batch: &Batch,
    values: &UserBatch<Key, Vec<u8>>,
    policy: &P,
) where
    P: Policy<F, Key, Key> + Policy<F, Key, Vec<u8>> + Clone + Send,
{
    assert_send(batch.build_batch_proof(fixed, policy));
    assert_send(values.build_batch_proof(variable, policy));
}
