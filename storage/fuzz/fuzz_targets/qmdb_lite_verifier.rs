//! Check the stateless verifier against the database writer.
//!
//! Each case proves and merkleizes seeded ordered batches, requires the verifier to accept the
//! honest bundle from an empty cache and from a retained one, sends it as bytes, corrupts one
//! part of the bundle or one byte of its encoding and requires rejection for the reason that
//! part breaks, and checks the database against an independent key/value model. Both Merkle
//! families run, with pruning every few batches.
//!
//! Input bytes: a little-endian u64 seed, then one byte each for rounds, actions, fault, mode,
//! floor policy and policy bound, then a little-endian u16 offset and a byte mask for the
//! encoding fault, then a chain depth. Missing bytes default to zero. Mode values congruent to
//! 1 modulo 16 select long cases: 16-64 batches of 80-320 actions over 4,095-4,097 initial keys
//! in an 8,192-key space. Short cases run up to 16 batches of up to 48 actions. When mode bit 5
//! is set, every fourth batch also deletes an eighth of the key space, leaving runs of clear
//! bitmap chunks for later floor scans to cross and the prover to cover with zero subtrees.
//! When mode bit 6 is set, every fourth batch starts with a range open below, above, or at both
//! ends, which the batch's own sets then split.
//!
//! The chain depth, modulo 4, is how many proved batches may wait unapplied. Each later batch
//! is proved on top of the pending one before it, against that batch's post-state, and once its
//! parent is applied, proving it from the database must give the same bundle and batch. The
//! oldest batch is applied whenever more are waiting.
//!
//! Build with debug assertions so arithmetic overflow is a crash:
//!
//! ```sh
//! cd storage
//! cargo +nightly fuzz run --debug-assertions qmdb_lite_verifier -- -max_len=18
//! ```
//!
//! On macOS, if `crc-fast` fails to link coverage symbols, prefix the command with
//! `RUSTFLAGS='-C link-arg=-Wl,-undefined,dynamic_lookup'`.

#![no_main]

#[path = "../../src/qmdb/current/verifier/benches/harness.rs"]
mod harness;

use commonware_codec::{Decode as _, Encode as _};
use commonware_cryptography::{Sha256, sha256::Digest};
use commonware_runtime::{Runner as _, Supervisor as _, deterministic};
use commonware_storage::{
    merkle::{Graftable, mmb, mmr},
    qmdb::{
        current::verifier::{
            BatchProofBundle, BitmapFault, BitmapProof, BitmapUnit, BundleConfig, CandidateFault,
            InsufficientPreState, PreStateKeyClaim, PreStateOpeningClaim, TrustedMap, UserBatch,
            VerifyError,
        },
        floor::{Bounded, Decision, Entry, Hold, Limits, Policy, Proportional},
    },
};
use libfuzzer_sys::fuzz_target;
use std::{
    collections::{BTreeMap, VecDeque},
    ops::Bound,
    sync::Arc,
};

// Short cases keep mutation throughput high; long cases exercise sustained log growth,
// repeated floor moves, and larger batches across many bitmap chunks.
const SHORT_MAX_BATCHES: usize = 16;
const SHORT_MAX_ACTIONS_PER_BATCH: usize = 48;
const LONG_MIN_BATCHES: usize = 16;
const LONG_MAX_BATCHES: usize = 64;
const LONG_MIN_ACTIONS_PER_BATCH: usize = 80;
const LONG_MAX_ACTIONS_PER_BATCH: usize = 320;
type Key = harness::Key;
type Value = harness::Value;

fn flip_value(value: &mut Value) {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(value.as_ref());
    bytes[0] ^= 1;
    *value = Value::new(bytes);
}

fn flip_digest(digest: &mut Digest) {
    digest.0[0] ^= 1;
}
type Encoding = commonware_storage::qmdb::any::value::FixedEncoding<Value>;
type Commitment<F> = commonware_storage::qmdb::current::verifier::CurrentCommitment<
    F,
    Key,
    Encoding,
    Sha256,
    { harness::CHUNK_BYTES },
>;

#[derive(Clone, Copy, Debug)]
struct FuzzInput {
    seed: u64,
    long: bool,
    sweeps: bool,
    open_ranges: bool,
    rounds: u8,
    actions: u8,
    fault: u8,
    policy: u8,
    bound: u8,
    offset: u16,
    mask: u8,
    depth: u8,
}

/// Generous decoding limits for the bundles these workloads produce.
const LIMITS: BundleConfig<(), (), (), ()> = BundleConfig {
    max_claims: 1 << 16,
    max_candidates: 1 << 16,
    max_bitmap_units: 1 << 16,
    max_digests: 1 << 16,
    key: (),
    update: (),
    operation: (),
    value: (),
};

/// The floor policies the verifier replays: held, proportional, a bounded number of moves, or
/// proportional moves that stop the walk at every `period`th update, counting across batches.
#[derive(Clone, Debug, PartialEq)]
enum FuzzPolicy {
    Hold,
    Proportional,
    Bounded(Bounded),
    StopEvery { period: usize, seen: usize },
}

impl FuzzPolicy {
    fn new(input: FuzzInput) -> Self {
        let bound = 1 + usize::from(input.bound) % 16;
        match input.policy % 4 {
            0 => Self::Proportional,
            1 => Self::Hold,
            2 => Self::Bounded(Bounded {
                entries: bound,
                skips: u64::MAX,
            }),
            _ => Self::StopEvery {
                period: bound,
                seen: 0,
            },
        }
    }
}

impl<F: Graftable> Policy<F, Key, Value> for FuzzPolicy {
    fn evicts(&self) -> bool {
        false
    }

    fn limits(&self, made_inactive: usize) -> Limits {
        match self {
            Self::Hold => Policy::<F, Key, Value>::limits(&Hold, made_inactive),
            Self::Proportional | Self::StopEvery { .. } => {
                Policy::<F, Key, Value>::limits(&Proportional, made_inactive)
            }
            Self::Bounded(bounded) => Policy::<F, Key, Value>::limits(bounded, made_inactive),
        }
    }

    fn decide<'a>(&mut self, entry: Entry<'a, F, Key, Value>) -> Decision<'a, Value> {
        match self {
            Self::Hold => Hold.decide(entry),
            Self::Proportional => Proportional.decide(entry),
            Self::Bounded(bounded) => bounded.decide(entry),
            Self::StopEvery { period, seen } => {
                *seen += 1;
                if seen.is_multiple_of(*period) {
                    entry.stop()
                } else {
                    entry.keep()
                }
            }
        }
    }
}

fn parse(data: &[u8]) -> FuzzInput {
    let mut seed_bytes = [0; 8];
    let copied = data.len().min(seed_bytes.len());
    seed_bytes[..copied].copy_from_slice(&data[..copied]);
    let byte = |index: usize| data.get(index).copied().unwrap_or_default();
    FuzzInput {
        seed: u64::from_le_bytes(seed_bytes),
        long: byte(11) % 16 == 1,
        sweeps: byte(11) & 0x20 != 0,
        open_ranges: byte(11) & 0x40 != 0,
        rounds: byte(8),
        actions: byte(9),
        fault: byte(10),
        policy: byte(12),
        bound: byte(13),
        offset: u16::from_le_bytes([byte(14), byte(15)]),
        mask: byte(16).max(1),
        depth: byte(17) % 4,
    }
}

fn fuzz(input: FuzzInput) {
    deterministic::Runner::seeded(input.seed).start(|context| async move {
        run_family::<mmr::Family>(context, input, "mmr").await;
    });
    deterministic::Runner::seeded(input.seed).start(|context| async move {
        run_family::<mmb::Family>(context, input, "mmb").await;
    });
}

async fn run_family<F: Graftable>(context: deterministic::Context, input: FuzzInput, family: &str) {
    let (rounds, actions, keyspace) = if input.long {
        (
            LONG_MIN_BATCHES
                + usize::from(input.rounds) % (LONG_MAX_BATCHES - LONG_MIN_BATCHES + 1),
            LONG_MIN_ACTIONS_PER_BATCH
                + usize::from(input.actions)
                    % (LONG_MAX_ACTIONS_PER_BATCH - LONG_MIN_ACTIONS_PER_BATCH + 1),
            8192,
        )
    } else {
        (
            usize::from(input.rounds) % (SHORT_MAX_BATCHES + 1),
            usize::from(input.actions) % (SHORT_MAX_ACTIONS_PER_BATCH + 1),
            1024,
        )
    };
    let mut generator = harness::SeededWorkload::new(input.seed, keyspace);
    let mut db = harness::open_db::<F>(
        context.child("qmdb_lite_verifier"),
        &format!("qmdb-lite-verifier-fuzz-{family}"),
    )
    .await;
    let initial_count = if input.long {
        4095 + input.seed % 3
    } else {
        match input.seed % 7 {
            0 => 0,
            1 => 1,
            2 => 255,
            3 => 256,
            4 => 257,
            5 => 512,
            _ => 513,
        }
    };
    // The writer and the verifier each advance their own copy of the policy once per batch.
    let mut writer_policy = FuzzPolicy::new(input);
    let mut verifier_policy = writer_policy.clone();
    // The model of the applied state, and of the state after every proved batch.
    let mut model = BTreeMap::new();
    let mut proved_model = BTreeMap::new();
    let mut pending = VecDeque::<Pending<F>>::new();
    let mut pre = Commitment::<F>::from_parts(db.bounds().end, db.root()).unwrap();
    let mut trusted = TrustedMap::empty_cache(pre);
    for round in 0..=rounds {
        let mut logical = if round == 0 {
            generator.initial_batch(initial_count)
        } else {
            generator.next_batch(actions)
        };
        if input.sweeps && round % 4 == 2 {
            // An eighth of the key space, below its middle: few enough deletes that this batch's
            // floor walk runs out of entries on the live records before them, leaving the run of
            // clear chunks for later walks to cross.
            let width = keyspace / 8;
            let start = keyspace / 2 - width * (1 + (round as u64 / 4) % 4);
            let end = start + width - 1;
            let (start, end) = (harness::key(start), harness::key(end));
            logical
                .batch
                .delete_range(start.clone()..=end.clone())
                .unwrap();
            logical
                .actions
                .push(harness::Action::delete_range(start, end));
        }
        if input.open_ranges && round % 4 == 3 {
            // A range open below or above, one below every key, or one open at both ends. It
            // arrives first, so the batch's later actions split it and win where they overlap.
            let (start, end) = match (round / 4) % 4 {
                0 => (
                    Bound::Unbounded,
                    Bound::Included(harness::key(keyspace / 8)),
                ),
                1 => (
                    Bound::Excluded(harness::key(keyspace - keyspace / 8)),
                    Bound::Unbounded,
                ),
                2 => (Bound::Unbounded, Bound::Excluded(harness::key(0))),
                _ => (Bound::Unbounded, Bound::Unbounded),
            };
            let mut opened = UserBatch::new();
            opened.delete_range((start.clone(), end.clone())).unwrap();
            opened.append(&mut logical.batch);
            if let Some(metadata) = logical.batch.metadata() {
                opened.set_metadata(metadata.clone());
            }
            logical.batch = opened;
            logical
                .actions
                .insert(0, harness::Action::DeleteRange { start, end });
        }
        let batch = &logical.batch;
        let open_below = matches!(batch.first().map(|op| op.start()), Some(Bound::Unbounded));
        let pre_empty = proved_model.is_empty();
        let writer_before = writer_policy.clone();
        let (proof, merkleized) = match pending.back() {
            None => {
                let proof = batch
                    .build_batch_proof(&db, &writer_policy)
                    .await
                    .expect("the database builds a complete proof");
                let merkleized = batch
                    .merkleize(&db, &mut writer_policy)
                    .await
                    .expect("the database merkleizes the same batch");
                (proof, merkleized)
            }
            Some(parent) => {
                let proof = batch
                    .build_batch_proof_on(&parent.merkleized, &db, &writer_policy)
                    .await
                    .expect("a pending batch builds a complete proof");
                let (proved, merkleized) = batch
                    .prove_on(&parent.merkleized, &db, &mut writer_policy)
                    .await
                    .expect("a pending batch proves the same batch");
                assert_eq!(proved.encode(), proof.encode());
                (proof, merkleized)
            }
        };
        let on_parent = !pending.is_empty();
        let post = Commitment::<F>::from_parts(merkleized.bounds().tip.size, merkleized.root())
            .expect("post commitment");

        // Every generated proof must work from an empty cache as well as a cache retained
        // from an earlier successful verification.
        let before = verifier_policy.clone();
        let encoded = proof.encode();
        let received =
            Proof::<F>::decode_cfg(encoded.clone(), &LIMITS).expect("the bundle decodes");
        let verified = batch
            .verify(
                pre,
                TrustedMap::empty_cache(pre),
                &received,
                post,
                &mut before.clone(),
            )
            .expect("complete proof verifies from an empty cache");
        assert_eq!(verified.commitment(), &post);
        trusted = batch
            .verify(pre, trusted, &proof, post, &mut verifier_policy)
            .expect("complete proof verifies with retained facts")
            .into_cache()
            .expect("retained facts carry to the child");
        assert_eq!(trusted.commitment(), &post);
        assert_eq!(writer_policy, verifier_policy);
        assert_eq!(
            proof.claims.post_floor,
            merkleized.bounds().inactivity_floor
        );

        // Only corrupt untrusted input. Imported cache facts are caller-trusted;
        // both honest cache paths above must succeed before trying a forgery.
        let mut forged = proof.clone();
        let mut claimed_post = post;
        let fault = input.fault % FAULTS;
        let expected = if fault == ENCODING_FAULT {
            // One changed byte must not panic the decoder or the verifier. It may decode to an
            // equivalent bundle: an exclusion claim's queried key, for one, is only a hint.
            let mut bytes = encoded.to_vec();
            let offset = usize::from(input.offset) % bytes.len();
            bytes[offset] ^= input.mask;
            if let Ok(decoded) = Proof::<F>::decode_cfg(bytes, &LIMITS) {
                let _ = batch.verify(
                    pre,
                    TrustedMap::empty_cache(pre),
                    &decoded,
                    post,
                    &mut before.clone(),
                );
            }
            None
        } else if fault == 8 {
            let mut root = *post.root();
            flip_digest(&mut root);
            claimed_post = Commitment::<F>::from_parts(post.size(), root).unwrap();
            Some(&[VerifyError::PostCommitmentMismatch][..])
        } else if fault == 7 && (pre_empty || batch.is_empty()) {
            None
        } else {
            corrupt(&mut forged, fault, open_below)
        };
        if let Some(expected) = expected {
            let error = batch
                .verify(
                    pre,
                    TrustedMap::empty_cache(pre),
                    &forged,
                    claimed_post,
                    &mut before.clone(),
                )
                .err();
            assert!(
                error.is_some_and(|error| expected.contains(&error)),
                "fault {fault} gave {error:?}, not one of {expected:?}: seed {}, round {round}, \
                 family {family}",
                input.seed
            );
        }

        logical.apply_model(&mut proved_model);
        pending.push_back(Pending {
            round,
            logical,
            reprove: on_parent.then_some(writer_before),
            bundle: proof.encode().to_vec(),
            merkleized,
        });
        while pending.len() > usize::from(input.depth) {
            let oldest = pending.pop_front().unwrap();
            db = apply_pending(db, oldest, &mut model, keyspace).await;
        }
        pre = post;
    }
    while let Some(oldest) = pending.pop_front() {
        db = apply_pending(db, oldest, &mut model, keyspace).await;
    }
    db.destroy().await.expect("destroy fuzz database");
}

/// A merkleized batch the database has not applied.
type Merkleized<F> = Arc<
    commonware_storage::qmdb::current::batch::MerkleizedBatch<
        F,
        Digest,
        commonware_storage::qmdb::any::ordered::fixed::Update<Key, Value>,
        { harness::CHUNK_BYTES },
        commonware_parallel::Sequential,
    >,
>;

/// A proved batch waiting to be applied.
struct Pending<F: Graftable> {
    round: usize,
    logical: harness::LogicalBatch,
    /// The writer's policy before the batch, and whether the batch was proved on a pending
    /// parent, so proving it from the database must give the same bundle.
    reprove: Option<FuzzPolicy>,
    bundle: Vec<u8>,
    merkleized: Merkleized<F>,
}

/// Apply `pending`, the oldest proved batch, to `db`, now at its pre-state, and check the
/// database against `model`.
async fn apply_pending<F: Graftable>(
    db: harness::ScenarioDb<F>,
    pending: Pending<F>,
    model: &mut BTreeMap<Key, Value>,
    keyspace: u64,
) -> harness::ScenarioDb<F> {
    let Pending {
        round,
        logical,
        reprove,
        bundle,
        merkleized,
    } = pending;
    if let Some(mut policy) = reprove {
        let (direct, batch) = logical
            .batch
            .prove(&db, &mut policy)
            .await
            .expect("the database proves the batch once its parent is applied");
        assert_eq!(
            direct.encode().to_vec(),
            bundle,
            "a pending parent changed the bundle"
        );
        assert_eq!(batch.root(), merkleized.root());
    }
    logical.apply_model(model);
    let mut db = db.apply_batch(merkleized).await.expect("apply batch");
    for index in 0..keyspace {
        let key = harness::key(index);
        assert_eq!(
            db.get(&key).await.expect("database get"),
            model.get(&key).cloned()
        );
    }
    assert_eq!(db.is_empty(), model.is_empty());
    if round % 4 == 0 {
        db = db.sync().await.unwrap();
        let boundary = db.sync_boundary();
        db = db.prune(boundary).await.unwrap();
    }
    db
}

const FAULTS: u8 = 31;
const ENCODING_FAULT: u8 = 30;
type Proof<F> = BatchProofBundle<F, Key, Encoding, Sha256, { harness::CHUNK_BYTES }>;

/// Moving the scan end can leave the log, change which chunks are needed, expose another live
/// record, or simply disagree with where the floor walk stopped.
const MOVED_SCAN_END: &[VerifyError] = &[
    VerifyError::InvalidBitmapProof(BitmapFault::ScanBounds),
    VerifyError::InvalidBitmapProof(BitmapFault::Coverage),
    VerifyError::InvalidBitmapProof(BitmapFault::NeedsBytes),
    VerifyError::InvalidBitmapProof(BitmapFault::ScanEnd),
    VerifyError::InvalidCandidates(CandidateFault::Locations),
    VerifyError::InvalidCandidates(CandidateFault::EmptiedScan),
];

/// Doubling a block breaks its alignment, overruns the graftable chunks, overlaps the next unit,
/// or no longer matches the tree proofs, which are checked before coverage.
const GROWN_ZERO_SUBTREE: &[VerifyError] = &[
    VerifyError::InvalidBitmapProof(BitmapFault::Coverage),
    VerifyError::InvalidBitmapProof(BitmapFault::Misaligned),
    VerifyError::InvalidBitmapProof(BitmapFault::Ungraftable),
    VerifyError::InvalidBitmapProof(BitmapFault::ProofShape),
    VerifyError::InvalidBitmapProof(BitmapFault::RawRoot),
];

/// The pre-state facts a batch needs were not established: range expansion runs first, and a
/// range open below needs the record wrapping to the least key before anything is classified.
/// Dropping claims also changes the records the shared proof covers, unless the candidates
/// carry them all.
const fn missing_facts(open_below: bool) -> &'static [VerifyError] {
    if open_below {
        &[
            VerifyError::InsufficientPreState(InsufficientPreState::OrderedEdge),
            VerifyError::InvalidRecordProof,
        ]
    } else {
        &[
            VerifyError::InsufficientPreState(InsufficientPreState::Classification),
            VerifyError::InvalidRecordProof,
        ]
    }
}

/// A changed record no longer matches the shared proof, or a copy of it at the same location.
const CHANGED_RECORD: &[VerifyError] = &[
    VerifyError::InvalidRecordProof,
    VerifyError::ConflictingPreState,
];

/// Apply one invalid mutation and return the errors it may cause, or skip it when its required
/// proof field is absent. Every mutation changes authenticated data or violates a required
/// structural check. `open_below` says whether the batch's first range has no lower bound.
fn corrupt<F: Graftable>(
    proof: &mut Proof<F>,
    fault: u8,
    open_below: bool,
) -> Option<&'static [VerifyError]> {
    let expected: &'static [VerifyError] = match fault {
        0 => {
            proof.claims.post_floor += 1;
            &[VerifyError::ClaimedBoundsMismatch]
        }
        1 => {
            proof.pre_state_candidates.operations.pop()?;
            &[
                VerifyError::InvalidCandidates(CandidateFault::Locations),
                VerifyError::InvalidRecordProof,
            ]
        }
        2 => {
            let (_, commonware_storage::qmdb::any::operation::Operation::Update(update)) =
                proof.pre_state_candidates.operations.first_mut()?
            else {
                return None;
            };
            flip_value(&mut update.value);
            CHANGED_RECORD
        }
        3 | 4 => {
            let member = proof
                .pre_state_claims
                .iter_mut()
                .find_map(|claim| match claim {
                    PreStateKeyClaim::Member(member) => Some(member),
                    _ => None,
                })?;
            if fault == 3 {
                flip_value(&mut member.value);
            } else {
                flip_value(&mut member.next_key);
            }
            CHANGED_RECORD
        }
        5 => {
            match &mut proof.bitmap {
                BitmapProof::Initial { floor_scan_end } => *floor_scan_end += 1,
                BitmapProof::Proven(bitmap) => bitmap.floor_scan_end += 1,
            }
            MOVED_SCAN_END
        }
        6 => {
            let BitmapProof::Proven(bitmap) = &mut proof.bitmap else {
                return None;
            };
            bitmap.units.pop()?;
            // Units are authenticated before coverage is checked, so the dropped unit can first
            // break the tree proofs' shape, or leave a claimed record's bit unknown.
            &[
                VerifyError::InvalidBitmapProof(BitmapFault::Coverage),
                VerifyError::InvalidBitmapProof(BitmapFault::ProofShape),
                VerifyError::InvalidBitmapProof(BitmapFault::RawRoot),
                VerifyError::InactiveClaim,
            ]
        }
        7 => {
            proof.pre_state_claims.first()?;
            proof.pre_state_claims.clear();
            missing_facts(open_below)
        }
        9 => {
            proof.claims.pre_size += 1;
            &[VerifyError::ClaimedBoundsMismatch]
        }
        10 => {
            let exclusion = proof
                .pre_state_claims
                .iter_mut()
                .find_map(|claim| match claim {
                    PreStateKeyClaim::Exclusion(exclusion) => Some(exclusion),
                    _ => None,
                })?;
            flip_value(&mut exclusion.owner.next_key);
            // The changed link may stop spanning the queried key before the record is checked.
            &[
                VerifyError::InvalidExclusionProof,
                VerifyError::InvalidRecordProof,
                VerifyError::ConflictingPreState,
            ]
        }
        11 => {
            proof.claims.pre_floor += 1;
            &[VerifyError::ClaimedBoundsMismatch]
        }
        12..=16 => {
            let PreStateOpeningClaim::Proven {
                ops_root, terminal, ..
            } = &mut proof.pre_state
            else {
                return None;
            };
            match fault {
                12 => flip_digest(ops_root),
                13 => terminal.floor += 1,
                14 => flip_metadata(&mut terminal.metadata),
                15 => flip_digest(terminal.raw_proof.digests.first_mut()?),
                _ => flip_digest(terminal.pins_before_terminal.first_mut()?),
            }
            &[VerifyError::InvalidPreStateOpening]
        }
        17 | 18 | 20 | 21 => {
            let BitmapProof::Proven(bitmap) = &mut proof.bitmap else {
                return None;
            };
            match fault {
                17 => {
                    let BitmapUnit::ExplicitChunk {
                        bytes,
                        raw_subtree_digest,
                        ..
                    } = bitmap.units.first_mut()?
                    else {
                        return None;
                    };
                    bytes[0] ^= 1;
                    // The grafted tree binds a graftable chunk; a digest binds the others.
                    if raw_subtree_digest.is_some() {
                        &[VerifyError::InvalidBitmapProof(BitmapFault::GraftedRoot)]
                    } else {
                        &[
                            VerifyError::InvalidBitmapProof(BitmapFault::PendingChunk),
                            VerifyError::InvalidBitmapProof(BitmapFault::PartialChunk),
                        ]
                    }
                }
                18 => {
                    flip_digest(bitmap.raw_proof.as_mut()?.digests.first_mut()?);
                    &[VerifyError::InvalidBitmapProof(BitmapFault::RawRoot)]
                }
                20 => {
                    flip_digest(bitmap.grafted_proof.as_mut()?.digests.first_mut()?);
                    &[VerifyError::InvalidBitmapProof(BitmapFault::GraftedRoot)]
                }
                _ => {
                    flip_digest(bitmap.grafted_append_frontier.first_mut()?);
                    &[VerifyError::InvalidBitmapProof(BitmapFault::Frontier)]
                }
            }
        }
        19 => {
            proof.pre_state_candidates.operations.first_mut()?.0 += 1;
            &[
                VerifyError::InvalidCandidates(CandidateFault::Locations),
                VerifyError::InvalidRecordProof,
                VerifyError::ConflictingPreState,
            ]
        }
        22 => {
            proof.records_proof.take()?;
            &[VerifyError::InvalidRecordProof]
        }
        23 => {
            proof
                .pre_state_claims
                .push(proof.pre_state_claims.first()?.clone());
            &[VerifyError::ConflictingPreState]
        }
        24 => {
            proof
                .pre_state_candidates
                .operations
                .push(proof.pre_state_candidates.operations.first()?.clone());
            &[VerifyError::InvalidCandidates(CandidateFault::Locations)]
        }
        25 => {
            let BitmapProof::Proven(bitmap) = &mut proof.bitmap else {
                return None;
            };
            bitmap.units.push(bitmap.units.first()?.clone());
            &[VerifyError::InvalidBitmapProof(BitmapFault::Coverage)]
        }
        26..=28 => {
            let BitmapProof::Proven(bitmap) = &mut proof.bitmap else {
                return None;
            };
            let index = bitmap.units.iter().position(|unit| match unit {
                BitmapUnit::ZeroSubtree { chunk_height, .. } => fault != 28 || *chunk_height >= 1,
                BitmapUnit::ExplicitChunk { .. } => false,
            })?;
            let BitmapUnit::ZeroSubtree {
                first_chunk,
                chunk_height,
                common_subtree_digest,
                ..
            } = &mut bitmap.units[index]
            else {
                return None;
            };
            match fault {
                26 => {
                    flip_digest(common_subtree_digest);
                    &[VerifyError::InvalidBitmapProof(BitmapFault::RawRoot)]
                }
                27 => {
                    *chunk_height += 1;
                    GROWN_ZERO_SUBTREE
                }
                _ => {
                    // A pair starting at the block's second chunk is misaligned, which the
                    // verifier finds before it checks coverage or the tree proofs.
                    let (first, digest) = (*first_chunk, *common_subtree_digest);
                    bitmap.units.splice(
                        index..=index,
                        [
                            BitmapUnit::zero_subtree(first, 0, digest),
                            BitmapUnit::zero_subtree(first + 1, 1, digest),
                        ],
                    );
                    &[VerifyError::InvalidBitmapProof(BitmapFault::Misaligned)]
                }
            }
        }
        29 if open_below => {
            // Drop every claim carrying the greatest key's record, which wraps to the least.
            let claims = proof.pre_state_claims.len();
            proof.pre_state_claims.retain(|claim| match claim {
                PreStateKeyClaim::Member(member) => member.next_key > member.key,
                PreStateKeyClaim::Exclusion(exclusion) => {
                    exclusion.owner.next_key > exclusion.owner.key
                }
            });
            if proof.pre_state_claims.len() == claims {
                return None;
            }
            missing_facts(true)
        }
        _ => return None,
    };
    Some(expected)
}

fn flip_metadata(metadata: &mut Option<Value>) {
    if let Some(value) = metadata {
        flip_value(value);
    } else {
        *metadata = Some(harness::value(0));
    }
}

fuzz_target!(|data: &[u8]| fuzz(parse(data)));
