//! Replay Current's bounded relocation of live operations below the append tip.
//!
//! Bitmap bits identify every pre-state candidate. Raw inclusion authenticates its
//! payload. The scan examines candidates in location order and copies those still
//! live after the application operations. Clearing their old activity bits allows
//! the floor to advance. Replay determines the required copies and stopping point,
//! which the verifier compares with the bundle's claimed scan endpoint.

use super::{
    BitmapFault, CandidateFault, PreStateCandidateProof, VerifyError, bitmap::VerifiedBitmapView,
    opening::AuthenticatedPreState, ordered::PreparedOrderedBatch,
};
use crate::{
    merkle::{Family, Graftable, Location},
    qmdb::{
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        floor::{End, Limits, Policy, ReplayError, replay},
        operation::Key,
    },
};
use alloc::vec::Vec;
use commonware_codec::Codec;
use commonware_cryptography::Hasher;

/// Locally derived relocation suffix and terminal commit.
/// `copies` follow the application operations. `copy_locations` are their source
/// locations, whose bits must be cleared. Destinations are consecutive append slots.
/// Root reconstruction uses these operations and source locations to append the
/// copies and clear their old activity bits. `floor` is encoded in `terminal`, and
/// `child_size` includes that terminal operation.
pub(crate) struct FloorResult<F: Graftable, K: Key, V: ValueEncoding> {
    pub(crate) floor: Location<F>,
    pub(crate) copies: Vec<Operation<F, update::Ordered<K, V>>>,
    pub(crate) copy_locations: Vec<Location<F>>,
    pub(crate) terminal: Operation<F, update::Ordered<K, V>>,
    pub(crate) child_size: Location<F>,
}

/// An authenticated old update and its location, as floor replay consumes it.
pub(crate) type Candidate<'a, F, K, V> = (Location<F>, &'a update::Ordered<K, V>);

/// Authenticate exactly the live pre-state payloads exposed by the bitmap scan.
///
/// Floor replay needs both completeness (every candidate is present) and authentic
/// contents (each candidate has the committed key, value, and successor). Bitmap
/// coverage establishes the locations, and the bundle's shared record proof, checked by
/// [`super::ordered::authenticate_and_merge_records`], establishes the payloads.
/// This lets replay decide which records the batch supersedes and which it must copy.
///
/// Compare locations in order with every set bit in `[old_floor, floor_scan_end)`,
/// excluding the separately authenticated terminal. Reject missing, extra, duplicate,
/// or reordered candidates and non-update payloads. Bitmap coverage must already have
/// been verified: this function relies on it to prove completeness. CPU is
/// O(e_scan * N + f) for `f` candidates and `e_scan` explicit scan chunks, and memory O(f).
pub(crate) fn verify_candidates<'a, F, K, V, H, const N: usize>(
    pre: &AuthenticatedPreState<F, K, V, H, N>,
    bitmap: &VerifiedBitmapView<F, H::Digest, N>,
    proof: &'a PreStateCandidateProof<F, K, V>,
) -> Result<Vec<Candidate<'a, F, K, V>>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    Operation<F, update::Ordered<K, V>>: Codec,
{
    let expected = bitmap.set_bits(pre.old_floor, bitmap.floor_scan_end);
    if proof
        .operations
        .iter()
        .map(|(location, _)| *location)
        .ne(expected
            .into_iter()
            .filter(|location| *location != pre.terminal_location))
    {
        return Err(CandidateFault::Locations.into());
    }
    Ok(proof
        .operations
        .iter()
        .map(|(location, operation)| match operation {
            Operation::Update(update) => Ok((*location, update)),
            _ => Err(CandidateFault::NotUpdate),
        })
        .collect::<Result<Vec<_>, _>>()?)
}

/// Reject a floor policy whose walk replay cannot check from the scan alone.
///
/// An eviction also rewrites the link of the key before the evicted one, which the candidates do
/// not determine, and a skip limit can end the walk at an inactive location the scan does not
/// otherwise reveal. A policy that decides no entries never walks, so its skips do not matter.
/// `made_inactive` is the count the writer hands [`Policy::limits`].
pub(crate) fn check_policy<F, K, V, P>(policy: &P, made_inactive: usize) -> Result<(), VerifyError>
where
    F: Family,
    P: Policy<F, K, V>,
{
    let Limits { entries, skips } = policy.limits(made_inactive);
    if policy.evicts() || (entries > 0 && skips != u64::MAX) {
        return Err(VerifyError::UnsupportedFloorPolicy);
    }
    Ok(())
}

/// What a floor walk over a non-empty child decided.
pub(crate) struct Walked<F: Graftable, K: Key, V: ValueEncoding> {
    pub(crate) floor: Location<F>,
    pub(crate) copies: Vec<Operation<F, update::Ordered<K, V>>>,
    pub(crate) copy_locations: Vec<Location<F>>,
    /// The end of the old-log coverage the walk consumed: the claimed `floor_scan_end`
    /// must equal it.
    pub(crate) scan_end: Location<F>,
}

/// Replay `policy` over the old active `candidates` and then the derived application
/// operations, for a child that is not empty. `candidates` must hold every active old
/// record the walk can reach, in ascending order, superseded ones included; extra
/// records beyond the walk's end are ignored by the walk but make the coverage claim wrong.
///
/// The caller has accepted `policy` with [`check_policy`]. The policy sees the number of
/// operations the batch made inactive, as the writer's own walk does.
pub(crate) fn walk_floor<F, K, V, P>(
    old_floor: Location<F>,
    base_size: Location<F>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    candidates: &[Candidate<'_, F, K, V>],
    policy: &mut P,
) -> Result<Walked<F, K, V>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    P: Policy<F, K, V::Value>,
{
    let fixed_tip = base_size
        .checked_add(
            u64::try_from(prepared.application_ops.len()).map_err(|_| VerifyError::Overflow)?,
        )
        .ok_or(VerifyError::Overflow)?;
    // Both sources are ordered. Application locations are bounded by fixed_tip.
    let live = candidates
        .iter()
        .copied()
        .filter(|(location, _)| !prepared.superseded.contains(location))
        .chain(
            prepared
                .application_ops
                .iter()
                .zip(0u64..)
                .filter_map(|(operation, index)| match operation {
                    Operation::Update(update) => Some((base_size + index, update)),
                    _ => None,
                }),
        )
        .collect::<Vec<_>>();
    let reached = replay(
        policy,
        old_floor,
        fixed_tip,
        prepared.made_inactive(),
        live.iter()
            .map(|(location, update)| (*location, &update.key, update.value.clone())),
    )
    .map_err(|error| match error {
        ReplayError::Evicts | ReplayError::Evicted => VerifyError::UnsupportedFloorPolicy,
        ReplayError::Unordered => CandidateFault::Unordered.into(),
    })?;
    // Every update the walk reaches is moved, so the moves are a prefix of the live updates.
    let mut copies = Vec::with_capacity(reached.moves.len());
    let mut copy_locations = Vec::with_capacity(reached.moves.len());
    for ((location, update), (moved, value)) in live.iter().zip(reached.moves) {
        debug_assert_eq!(*location, moved);
        let mut copy = (*update).clone();
        copy.value = value;
        copies.push(Operation::Update(copy));
        copy_locations.push(*location);
    }
    let floor = reached.floor;
    let scan_end = match reached.end {
        // Entries ran out on an old record, so the walk read the old log through it.
        End::Entries if floor <= base_size => floor,
        // The policy stopped at an old record, which the scan must expose.
        End::Stopped(location) if location < base_size => location + 1,
        // The walk went on into the application operations, so it consumed the whole old log.
        End::Entries | End::Stopped(_) | End::Exhausted => base_size,
        // `check_policy` rejects every policy that decides entries and limits skips.
        End::Skips => return Err(VerifyError::UnsupportedFloorPolicy),
    };
    Ok(Walked {
        floor,
        copies,
        copy_locations,
        scan_end,
    })
}

/// Derive live-record copies, the new floor, and the terminal commit by replaying the scan.
///
/// Current relocates live records to reclaim the inactive prefix as commits invalidate
/// old records. How far the floor advances is a [`Policy`] chosen by whoever runs the writer
/// and handed to the verifier with the batch, because the commitment does not record it.
/// The policy sees the number of operations the batch made inactive, one per superseded
/// record and two per delete, exactly as the writer's own walk does. Reject a policy replay
/// cannot check, then check that the old terminal and every superseded location have active
/// bits.
///
/// Scan authenticated old candidates, then the derived application operations.
/// `fixed_tip` is the end of those application operations, before any floor copies.
/// Skip superseded old records. Each derived application update is live: ordered
/// derivation emits at most one operation per key, and every replaced old location
/// is superseded. [`replay`] decides each live update the walk reaches, appending a copy
/// at the tip for each. Copies lie outside that prefix and are never scanned again.
///
/// The claimed `floor_scan_end` must equal the coverage the walk consumed: through the
/// last decided update when its entries ran out, through a stopping update, and through
/// the end of the old log when the prefix below `fixed_tip` was exhausted. These checks
/// prevent either omitting an old live record or extending the floor beyond the required
/// scan. For an empty child, require no scan and place the floor at the new terminal.
/// The ordered stage must already have established emptiness.
///
/// All inputs must come from this transition's authenticated or derived stage results.
/// The policy's entry limit bounds the copies, and the walk passes at most the O(k)
/// superseded records besides them. Authenticating all supplied candidates is a separate
/// cost, paid before this replay, including for rejected bundles.
pub(crate) fn advance_floor<F, K, V, H, P, const N: usize>(
    pre: &AuthenticatedPreState<F, K, V, H, N>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    bitmap: &VerifiedBitmapView<F, H::Digest, N>,
    candidates: &[Candidate<'_, F, K, V>],
    commit_metadata: Option<V::Value>,
    policy: &mut P,
) -> Result<FloorResult<F, K, V>, VerifyError>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    P: Policy<F, K, V::Value>,
{
    check_policy(policy, prepared.made_inactive())?;
    if bitmap.bit(pre.terminal_location) != Some(true)
        || prepared
            .superseded
            .iter()
            .any(|location| bitmap.bit(*location) != Some(true))
    {
        return Err(BitmapFault::Inactive.into());
    }
    let base_size = pre.commitment.size();
    let fixed_tip = base_size
        .checked_add(
            u64::try_from(prepared.application_ops.len()).map_err(|_| VerifyError::Overflow)?,
        )
        .ok_or(VerifyError::Overflow)?;
    let floor;
    let mut copies = Vec::new();
    let mut copy_locations = Vec::new();
    if prepared.post_empty {
        if bitmap.floor_scan_end != pre.old_floor || !candidates.is_empty() {
            return Err(CandidateFault::EmptiedScan.into());
        }
        floor = fixed_tip;
    } else {
        let walked = walk_floor(pre.old_floor, base_size, prepared, candidates, policy)?;
        if bitmap.floor_scan_end != walked.scan_end {
            return Err(BitmapFault::ScanEnd.into());
        }
        floor = walked.floor;
        copies = walked.copies;
        copy_locations = walked.copy_locations;
    }
    let terminal_location = fixed_tip
        .checked_add(u64::try_from(copies.len()).map_err(|_| VerifyError::Overflow)?)
        .ok_or(VerifyError::Overflow)?;
    Ok(FloorResult {
        floor,
        copies,
        copy_locations,
        terminal: Operation::CommitFloor(commit_metadata, floor),
        child_size: terminal_location
            .checked_add(1)
            .ok_or(VerifyError::Overflow)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        merkle::mmr,
        qmdb::{
            any::value::FixedEncoding,
            floor::{Bounded, Decision, Entry, Hold, Proportional},
        },
    };
    use alloc::{collections::BTreeSet, vec};
    use commonware_utils::sequence::FixedBytes;

    type F = mmr::Family;
    type Key = FixedBytes<8>;
    type Encoding = FixedEncoding<Key>;
    type Op = Operation<F, update::Ordered<Key, Encoding>>;

    const OLD_FLOOR: u64 = 2;
    const BASE_SIZE: u64 = 20;

    fn key(index: u64) -> Key {
        FixedBytes::new(index.to_be_bytes())
    }

    fn record(index: u64) -> update::Ordered<Key, Encoding> {
        update::Ordered {
            key: key(index),
            value: key(index),
            next_key: key(index + 1),
        }
    }

    fn update(index: u64) -> Op {
        Operation::Update(record(index))
    }

    /// Old live records at 3, 8 and 12 and a superseded one at 5, followed by application
    /// updates at 20 and 21.
    fn walk(
        policy: &mut impl Policy<F, Key, Key>,
    ) -> Result<Walked<F, Key, Encoding>, VerifyError> {
        let records = [3, 5, 8, 12].map(|location| (Location::new(location), record(location)));
        let candidates = records
            .iter()
            .map(|(location, record)| (*location, record))
            .collect::<Vec<_>>();
        let prepared = PreparedOrderedBatch {
            application_ops: vec![update(100), update(5)],
            superseded: BTreeSet::from([Location::new(5)]),
            deleted_keys: BTreeSet::new(),
            post_empty: false,
        };
        walk_floor(
            Location::new(OLD_FLOOR),
            Location::new(BASE_SIZE),
            &prepared,
            &candidates,
            policy,
        )
    }

    /// The floor, the scan end, and the source location of each copy.
    fn summary(walked: &Walked<F, Key, Encoding>) -> (u64, u64, Vec<u64>) {
        (
            *walked.floor,
            *walked.scan_end,
            walked
                .copy_locations
                .iter()
                .map(|location| **location)
                .collect(),
        )
    }

    /// Keeps every update before `0` and stops at it.
    struct StopAt(u64);

    impl Policy<F, Key, Key> for StopAt {
        fn evicts(&self) -> bool {
            false
        }

        fn limits(&self, _: usize) -> Limits {
            Limits {
                entries: usize::MAX,
                skips: u64::MAX,
            }
        }

        fn decide<'a>(&mut self, entry: Entry<'a, F, Key, Key>) -> Decision<'a, Key> {
            if *entry.location() == self.0 {
                entry.stop()
            } else {
                entry.keep()
            }
        }
    }

    /// Writes `0` for each update it decides, within one entry.
    struct Replace(Key);

    impl Policy<F, Key, Key> for Replace {
        fn evicts(&self) -> bool {
            false
        }

        fn limits(&self, _: usize) -> Limits {
            Limits {
                entries: 1,
                skips: u64::MAX,
            }
        }

        fn decide<'a>(&mut self, entry: Entry<'a, F, Key, Key>) -> Decision<'a, Key> {
            entry.replace(self.0.clone())
        }
    }

    /// Evicts every update it decides.
    struct Evict;

    impl Policy<F, Key, Key> for Evict {
        fn evicts(&self) -> bool {
            true
        }

        fn limits(&self, _: usize) -> Limits {
            Limits {
                entries: 1,
                skips: u64::MAX,
            }
        }

        fn decide<'a>(&mut self, entry: Entry<'a, F, Key, Key>) -> Decision<'a, Key> {
            entry.evict().0
        }
    }

    #[test]
    fn hold_scans_nothing() {
        let walked = walk(&mut Hold).unwrap();
        assert_eq!(summary(&walked), (OLD_FLOOR, OLD_FLOOR, vec![]));
    }

    #[test]
    fn entries_ending_on_an_old_record_end_the_scan_after_it() {
        let mut bounded = Bounded {
            entries: 2,
            skips: u64::MAX,
        };
        let walked = walk(&mut bounded).unwrap();
        // The superseded record at 5 is passed, not moved.
        assert_eq!(summary(&walked), (9, 9, vec![3, 8]));
        // Proportional gets one entry per superseded record plus one: the same two.
        let proportional = walk(&mut Proportional).unwrap();
        assert_eq!(summary(&proportional), summary(&walked));
    }

    #[test]
    fn entries_ending_on_an_application_update_scan_the_whole_old_log() {
        let mut bounded = Bounded {
            entries: 4,
            skips: u64::MAX,
        };
        let walked = walk(&mut bounded).unwrap();
        assert_eq!(summary(&walked), (21, BASE_SIZE, vec![3, 8, 12, 20]));
    }

    #[test]
    fn exhausting_the_live_updates_moves_the_floor_to_the_tip() {
        let mut bounded = Bounded {
            entries: 10,
            skips: u64::MAX,
        };
        let walked = walk(&mut bounded).unwrap();
        assert_eq!(summary(&walked), (22, BASE_SIZE, vec![3, 8, 12, 20, 21]));
    }

    #[test]
    fn stopping_at_an_old_record_ends_the_scan_after_it() {
        let walked = walk(&mut StopAt(8)).unwrap();
        assert_eq!(summary(&walked), (8, 9, vec![3]));
    }

    #[test]
    fn stopping_at_an_application_update_scans_the_whole_old_log() {
        let walked = walk(&mut StopAt(21)).unwrap();
        assert_eq!(summary(&walked), (21, BASE_SIZE, vec![3, 8, 12, 20]));
    }

    #[test]
    fn a_replaced_value_keeps_the_record_link() {
        let walked = walk(&mut Replace(key(7))).unwrap();
        assert_eq!(summary(&walked), (4, 4, vec![3]));
        assert_eq!(
            walked.copies,
            vec![Operation::Update(update::Ordered {
                key: key(3),
                value: key(7),
                next_key: key(4),
            })]
        );
    }

    #[test]
    fn made_inactive_counts_each_delete_twice() {
        let prepared = PreparedOrderedBatch::<F, Key, Encoding> {
            application_ops: vec![Operation::Delete(key(7)), update(6)],
            superseded: BTreeSet::from([Location::new(6), Location::new(7)]),
            deleted_keys: BTreeSet::from([key(7)]),
            post_empty: false,
        };
        assert_eq!(prepared.made_inactive(), 3);
    }

    #[test]
    fn policies_replay_cannot_check_are_rejected() {
        fn check(policy: &impl Policy<F, Key, Key>) -> Result<(), VerifyError> {
            check_policy(policy, 0)
        }
        let unsupported = Err(VerifyError::UnsupportedFloorPolicy);
        assert_eq!(check(&Evict), unsupported);
        let skip_limited = Bounded {
            entries: 1,
            skips: 5,
        };
        assert_eq!(check(&skip_limited), unsupported);
        // A walk that would run out of skips is refused even when the caller skipped the check.
        let mut no_skips = Bounded {
            entries: 2,
            skips: 0,
        };
        assert!(matches!(
            walk(&mut no_skips),
            Err(VerifyError::UnsupportedFloorPolicy)
        ));
        let entryless = Bounded {
            entries: 0,
            skips: 5,
        };
        assert_eq!(check(&entryless), Ok(()));
        assert_eq!(check(&Proportional), Ok(()));
        assert_eq!(check(&Hold), Ok(()));
    }
}
