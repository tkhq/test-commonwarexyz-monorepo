//! Attacks with internally consistent bundles: every proof in the bundle is genuine, but the
//! bundle as a whole describes a transition other than the one the batch produces.

use super::{
    BitmapFault, BitmapProof, CandidateFault, CurrentCommitment, PreStateCandidateProof,
    PreStateExclusionClaim, PreStateKeyClaim, PreStateMemberClaim, TrustedMap, UserBatch,
    VerifyError,
    bitmap::build_bitmap_proof,
    floor::check_policy,
    ordered::{derive_ordered_transition, expand_range_deletes, verify_and_merge_claims},
    opening::authenticate_pre_state,
};
use crate::{
    journal::contiguous::{Contiguous as _, fixed::Config as JournalConfig},
    merkle::{Graftable, Location, full::Config as MerkleConfig, mmb, mmr},
    qmdb::{
        any::value::FixedEncoding,
        current::{
            FixedConfig,
            ordered::{fixed::Db, proof::constant::ExclusionProof},
        },
        floor::Proportional,
    },
    translator::OneCap,
};
use commonware_cryptography::Sha256;
use commonware_parallel::Sequential;
use commonware_runtime::{Runner as _, buffer::paged::CacheRef, deterministic};
use commonware_utils::{NZU16, NZU64, NZUsize, bitmap::Readable as _, sequence::FixedBytes};

type Key = FixedBytes<8>;
type Commitment<F> = CurrentCommitment<F, Key, FixedEncoding<Key>, Sha256, 32>;
type Batch = UserBatch<Key, Key>;
type TestDb<F> = Db<F, deterministic::Context, Key, Key, Sha256, OneCap, 32, Sequential>;

fn key(value: u64) -> Key {
    FixedBytes::new(value.to_be_bytes())
}

async fn open<F: Graftable>(context: deterministic::Context) -> TestDb<F> {
    let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
    let config = FixedConfig {
        merkle_config: MerkleConfig {
            journal_partition: "attack-tests-nodes".into(),
            replay_buffer: NZUsize!(4096),
            metadata_partition: "attack-tests-merkle-metadata".into(),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            strategy: Sequential,
            page_cache: page_cache.clone(),
        },
        journal_config: JournalConfig {
            partition: "attack-tests-operations".into(),
            replay_buffer: NZUsize!(4096),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            page_cache,
        },
        grafted_metadata_partition: "attack-tests-grafted".into(),
        translator: OneCap,
        init_cache: Some(NZUsize!(1024)),
        init_buffer: NZUsize!(4096),
        init_concurrency: (),
    };
    TestDb::<F>::init(context, config, None).await.unwrap()
}

async fn apply<F: Graftable>(db: TestDb<F>, batch: &Batch) -> TestDb<F> {
    let merkleized = batch.merkleize(&db, &mut Proportional).await.unwrap();
    db.apply_batch(merkleized).await.unwrap()
}

fn commitment<F: Graftable>(db: &TestDb<F>) -> Commitment<F> {
    Commitment::<F>::from_parts(db.bounds().end, db.root()).unwrap()
}

/// Keys 1..=8 hold their own key, then key 3 is rewritten so its first record is superseded.
/// Returns the database, the location of the superseded record and its successor key.
async fn with_superseded_record<F: Graftable>(
    context: deterministic::Context,
) -> (TestDb<F>, Location<F>, Key) {
    let db = open::<F>(context).await;
    let db = apply(db, &Batch::from_iter((1..=8).map(|i| UserBatchOp::set(i)))).await;
    let (old, old_loc) = db.any.get_with_loc(&key(3)).await.unwrap().unwrap();
    let db = apply(db, &Batch::from([UserBatchOp::set_value(3, 300)])).await;
    (db, old_loc, old.next_key)
}

struct UserBatchOp;

impl UserBatchOp {
    fn set(i: u64) -> super::UserOp<Key, Key> {
        super::UserOp::set(key(i), key(i))
    }

    fn set_value(i: u64, value: u64) -> super::UserOp<Key, Key> {
        super::UserOp::set(key(i), key(value))
    }
}

fn reject<F: Graftable>(
    what: &str,
    batch: &Batch,
    pre: Commitment<F>,
    bundle: &super::BatchProofBundle<F, Key, FixedEncoding<Key>, Sha256, 32>,
    post: Commitment<F>,
    expected: &[VerifyError],
) {
    let result = batch.verify(pre, TrustedMap::empty_cache(pre), bundle, post, &mut Proportional);
    let Err(error) = result else {
        panic!("accepted {what}");
    };
    assert!(
        expected.contains(&error),
        "{what} failed with {error:?}, not one of {expected:?}"
    );
}

/// A superseded record's genuine operation proof shows a clear activity bit, and forging the
/// bit breaks the grafted root; neither re-establishes the old value as a member or an edge.
fn stale_record_is_not_a_member<F: Graftable>() {
    let executor = deterministic::Runner::default();
    executor.start(|context| async move {
        let (db, old_loc, old_next) = with_superseded_record::<F>(context).await;
        let pre = commitment(&db);
        let batch = Batch::from([UserBatchOp::set_value(3, 301)]);
        let honest = batch.build_batch_proof(&db, &Proportional).await.unwrap();
        let merkleized = batch.merkleize(&db, &mut Proportional).await.unwrap();
        let post =
            Commitment::<F>::from_parts(merkleized.bounds().tip.size, merkleized.root()).unwrap();
        batch
            .verify(pre, TrustedMap::empty_cache(pre), &honest, post, &mut Proportional)
            .unwrap();

        // The old record of key 3, proven at its real (now inactive) location.
        let stale_proof = db.operation_proof(old_loc).await.unwrap();
        let stale_member = PreStateKeyClaim::Member(PreStateMemberClaim {
            key: key(3),
            value: key(3),
            proof: super::super::ordered::proof::constant::KeyValueProof {
                proof: stale_proof.clone(),
                next_key: old_next.clone(),
            },
        });
        let mut forged = honest.clone();
        forged.pre_state_claims = vec![stale_member.clone()];
        reject(
            "a superseded record as a member",
            &batch,
            pre,
            &forged,
            post,
            &[VerifyError::InvalidMemberProof],
        );

        // Setting the bit in the proof's own chunk changes the grafted leaf.
        let mut lit = stale_proof.clone();
        let bit = *lit.loc % 256;
        lit.chunk[(bit / 8) as usize] |= 1 << (bit % 8);
        let mut forged = honest.clone();
        forged.pre_state_claims = vec![PreStateKeyClaim::Member(PreStateMemberClaim {
            key: key(3),
            value: key(3),
            proof: super::super::ordered::proof::constant::KeyValueProof {
                proof: lit.clone(),
                next_key: old_next.clone(),
            },
        })];
        reject(
            "a superseded record with its bit set",
            &batch,
            pre,
            &forged,
            post,
            &[VerifyError::InvalidMemberProof],
        );

        // The stale record's edge (3 -> old_next) as an exclusion for a key between them.
        let between = key(3) < old_next;
        assert!(between, "old successor should be above key 3");
        let queried = {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(key(3).as_ref());
            bytes[7] = bytes[7].wrapping_add(1);
            FixedBytes::new(bytes)
        };
        let creates = Batch::from([super::UserOp::set(queried.clone(), key(9))]);
        let honest_create = creates.build_batch_proof(&db, &Proportional).await.unwrap();
        let merkleized = creates.merkleize(&db, &mut Proportional).await.unwrap();
        let post_create =
            Commitment::<F>::from_parts(merkleized.bounds().tip.size, merkleized.root()).unwrap();
        let mut forged = honest_create.clone();
        forged.pre_state_claims = vec![PreStateKeyClaim::Exclusion(PreStateExclusionClaim {
            queried_key: queried.clone(),
            proof: ExclusionProof::KeyValue(
                lit,
                crate::qmdb::any::operation::update::Ordered {
                    key: key(3),
                    value: key(3),
                    next_key: old_next.clone(),
                },
            ),
        })];
        reject(
            "a superseded edge as an exclusion",
            &creates,
            pre,
            &forged,
            post_create,
            &[VerifyError::InvalidExclusionProof],
        );
        db.destroy().await.unwrap();
    });
}

#[test]
fn mmr_stale_record_is_not_a_member() {
    stale_record_is_not_a_member::<mmr::Family>();
}

#[test]
fn mmb_stale_record_is_not_a_member() {
    stale_record_is_not_a_member::<mmb::Family>();
}

/// A bundle whose bitmap coverage and candidates are genuine for a shorter scan than the floor
/// policy requires hides live records from the floor walk; replay notices.
fn a_shortened_scan_is_rejected<F: Graftable>() {
    let executor = deterministic::Runner::default();
    executor.start(|context| async move {
        let (db, _, _) = with_superseded_record::<F>(context).await;
        let pre = commitment(&db);
        let batch = Batch::from([UserBatchOp::set_value(3, 301)]);
        let honest = batch.build_batch_proof(&db, &Proportional).await.unwrap();
        let merkleized = batch.merkleize(&db, &mut Proportional).await.unwrap();
        let post =
            Commitment::<F>::from_parts(merkleized.bounds().tip.size, merkleized.root()).unwrap();
        batch
            .verify(pre, TrustedMap::empty_cache(pre), &honest, post, &mut Proportional)
            .unwrap();
        let BitmapProof::Proven(bitmap) = &honest.bitmap else {
            panic!("expected a proven bitmap");
        };
        assert!(
            honest.pre_state_candidates.operations.len() >= 2,
            "the walk should reach at least two old records"
        );
        assert!(bitmap.floor_scan_end > honest.claims.pre_floor);

        // Rebuild the pre-state exactly as the prover does, then re-cover a shorter scan.
        let pre_state = authenticate_pre_state(pre, &honest.pre_state).unwrap();
        let mut cache = TrustedMap::empty_cache(pre);
        verify_and_merge_claims(&pre_state, &mut cache, &honest.pre_state_claims, &Sequential)
            .unwrap();
        let points = expand_range_deletes(&cache, batch.ops()).unwrap();
        let prepared = derive_ordered_transition(&cache, &points).unwrap();
        check_policy::<F, Key, Key, _>(&Proportional, prepared.made_inactive()).unwrap();
        let old_floor = honest.claims.pre_floor;
        let grafted_storage = db.grafted_storage();
        let recover = |scan_end: Location<F>| {
            build_bitmap_proof::<F, Key, FixedEncoding<Key>, Sha256, _, _, _, 32>(
                pre.size(),
                old_floor,
                scan_end,
                &prepared,
                db.bitmap(),
                &db.any.log.merkle,
                &grafted_storage,
            )
        };

        // No scan at all: a Hold-like bundle for a Proportional walk.
        let mut forged = honest.clone();
        forged.bitmap = recover(old_floor).await.unwrap();
        forged.pre_state_candidates = PreStateCandidateProof {
            operations: Vec::new(),
            raw_multi_proof: None,
        };
        reject(
            "no floor scan",
            &batch,
            pre,
            &forged,
            post,
            &[VerifyError::InvalidBitmapProof(BitmapFault::ScanEnd)],
        );

        // A scan through only the first candidate, with a genuine proof for just that one.
        let first = honest.pre_state_candidates.operations[0].clone();
        let scan_end = first.0 + 1;
        let mut forged = honest.clone();
        forged.bitmap = recover(scan_end).await.unwrap();
        let raw_multi_proof = crate::merkle::verification::multi_proof(
            &db.any.log.merkle,
            F::inactive_peaks(pre.size(), old_floor),
            &[first.0],
        )
        .await
        .unwrap();
        forged.pre_state_candidates = PreStateCandidateProof {
            operations: vec![first],
            raw_multi_proof: Some(raw_multi_proof),
        };
        reject(
            "a scan through one candidate",
            &batch,
            pre,
            &forged,
            post,
            &[
                VerifyError::InvalidBitmapProof(BitmapFault::ScanEnd),
                VerifyError::InvalidCandidates(CandidateFault::Locations),
            ],
        );

        // A scan past the honest end, with every candidate it exposes, moves the floor further
        // than the policy allows.
        let scan_end = pre.size();
        let mut forged = honest.clone();
        forged.bitmap = recover(scan_end).await.unwrap();
        let mut locations = Vec::new();
        let mut operations = Vec::new();
        for raw in db.bitmap().ones_iter_from(old_floor.as_u64()) {
            let location = Location::<F>::new(raw);
            if location >= pre.size() - 1 {
                break;
            }
            locations.push(location);
            operations.push((location, db.any.log.read(raw).await.unwrap()));
        }
        let raw_multi_proof = crate::merkle::verification::multi_proof(
            &db.any.log.merkle,
            F::inactive_peaks(pre.size(), old_floor),
            &locations,
        )
        .await
        .unwrap();
        forged.pre_state_candidates = PreStateCandidateProof {
            operations,
            raw_multi_proof: Some(raw_multi_proof),
        };
        reject(
            "a scan over the whole old log",
            &batch,
            pre,
            &forged,
            post,
            &[VerifyError::InvalidBitmapProof(BitmapFault::ScanEnd)],
        );

        // The grafted append frontier is bound to the grafted root.
        let mut forged = honest.clone();
        if let BitmapProof::Proven(bitmap) = &mut forged.bitmap {
            bitmap.grafted_append_frontier[0].0[0] ^= 1;
        }
        reject(
            "a changed grafted frontier node",
            &batch,
            pre,
            &forged,
            post,
            &[VerifyError::InvalidBitmapProof(BitmapFault::Frontier)],
        );
        let _ = recover;
        drop(grafted_storage);
        db.destroy().await.unwrap();
    });
}

#[test]
fn mmr_a_shortened_scan_is_rejected() {
    a_shortened_scan_is_rejected::<mmr::Family>();
}

#[test]
fn mmb_a_shortened_scan_is_rejected() {
    a_shortened_scan_is_rejected::<mmb::Family>();
}

/// Every fact a retained cache carries across verified transitions is true at the database
/// the writer produced: entries name active records with their location, value and successor,
/// and spans join a key to its actual successor.
fn retained_facts_are_true<F: Graftable>() {
    use super::{OptionTrusted::KnownTrusted, TrustedState};

    let executor = deterministic::Runner::default();
    executor.start(|context| async move {
        let mut db = open::<F>(context).await;
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        const KEYS: u64 = 48;
        let mut pre = commitment(&db);
        let mut cache = TrustedMap::empty_cache(pre);
        let mut policy = Proportional;
        let mut spans_seen = 0usize;
        let mut entries_seen = 0usize;
        for round in 0..60u32 {
            let mut batch = Batch::new();
            let actions = 1 + next(12);
            for _ in 0..actions {
                let k = next(KEYS);
                match next(10) {
                    0..=4 => batch.set(key(k), key(1000 + next(1_000))),
                    5..=7 => batch.delete(key(k)),
                    8 => {
                        let end = (k + next(6)).min(KEYS - 1);
                        batch.delete_range(key(k)..=key(end)).unwrap();
                    }
                    _ => {
                        if round % 7 == 3 {
                            batch.delete_range(..).unwrap();
                        } else {
                            batch.delete_range(key(k)..).unwrap();
                        }
                    }
                }
            }
            if round % 5 == 0 {
                batch.set_metadata(key(round.into()));
            }
            let proof = batch.build_batch_proof(&db, &policy).await.unwrap();
            let merkleized = batch.merkleize(&db, &mut policy.clone()).await.unwrap();
            let post = Commitment::<F>::from_parts(merkleized.bounds().tip.size, merkleized.root())
                .unwrap();
            cache = batch
                .verify(pre, cache, &proof, post, &mut policy)
                .unwrap()
                .into_cache()
                .unwrap();
            db = db.apply_batch(merkleized).await.unwrap();
            assert_eq!(cache.commitment(), &commitment(&db));

            match cache.state() {
                TrustedState::EmptyDatabase => assert!(db.is_empty(), "round {round}"),
                TrustedState::Partial(partial) => {
                    for entry in partial.entries() {
                        entries_seen += 1;
                        let (record, location) = db
                            .any
                            .get_with_loc(&entry.key)
                            .await
                            .unwrap()
                            .unwrap_or_else(|| panic!("round {round}: {:?} is not active", entry.key));
                        if let KnownTrusted(cached) = entry.location {
                            assert_eq!(cached, location, "round {round}: {:?}", entry.key);
                        }
                        if let KnownTrusted(cached) = &entry.value {
                            assert_eq!(cached, &record.value, "round {round}: {:?}", entry.key);
                        }
                        if let KnownTrusted(cached) = &entry.next_key {
                            assert_eq!(cached, &record.next_key, "round {round}: {:?}", entry.key);
                        }
                    }
                    for (predecessor, successor) in partial.absences().iter() {
                        spans_seen += 1;
                        let (record, _) = db
                            .any
                            .get_with_loc(predecessor)
                            .await
                            .unwrap()
                            .unwrap_or_else(|| panic!("round {round}: span start {predecessor:?} is not active"));
                        assert_eq!(&record.next_key, successor, "round {round}: span from {predecessor:?}");
                    }
                    if partial.entries().next().is_some() {
                        assert!(!db.is_empty());
                    }
                }
            }
            pre = post;
            if round % 4 == 3 {
                db = db.sync().await.unwrap();
                let boundary = db.sync_boundary();
                db = db.prune(boundary).await.unwrap();
            }
        }
        assert!(entries_seen > 100 && spans_seen > 20, "{entries_seen} entries, {spans_seen} spans");
        db.destroy().await.unwrap();
    });
}

#[test]
fn mmr_retained_facts_are_true() {
    retained_facts_are_true::<mmr::Family>();
}

#[test]
fn mmb_retained_facts_are_true() {
    retained_facts_are_true::<mmb::Family>();
}
