//! Proof production from a committed storage-backed ordered Current database.

use super::{
    BatchProofBundle, BuildError, ClaimedBatchBounds, CurrentCommitment, PreStateCandidateProof,
    PreStateExclusionClaim, PreStateKeyClaim, PreStateMemberClaim, PreStateOpeningClaim,
    PreStateTerminalClaim, TrustedMap, UserBatch, UserOp,
    bitmap::{build_bitmap_proof, verify_bitmap},
    commitment::validate_profile,
    floor::{advance_floor, check_policy, verify_candidates, walk_floor},
    opening::authenticate_pre_state,
    ordered::{derive_ordered_transition, expand_range_deletes, verify_and_merge_claims},
};
use crate::{
    Context,
    index::Ordered as OrderedIndex,
    journal::contiguous::{Contiguous as _, Mutable},
    merkle::{self, Graftable, Location},
    qmdb::{
        self,
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        current::ordered,
        floor::Policy,
        operation::Key,
    },
};
use commonware_codec::Codec;
use commonware_cryptography::Hasher;
use commonware_parallel::Strategy;
use commonware_utils::{NZU64, bitmap::Readable as _, futures::try_join_all};
use futures::{StreamExt as _, pin_mut};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Bound,
};

type OrderedUpdate<K, V> = update::Ordered<K, V>;
type ExactRecord<F, K, V> = (Location<F>, OrderedUpdate<K, V>);

impl<K: Key, A: Clone + Send + Sync> UserBatch<K, A> {
    /// Construct a complete proof from this database's current committed checkpoint.
    ///
    /// The bundle works with an empty trusted cache. Ordered membership and exclusion
    /// proofs cover the requested keys, every member of a delete range, and affected
    /// predecessors. A range with no lower bound adds the greatest key's record, whose
    /// successor link proves where the range begins. Bitmap and raw-log proofs cover
    /// the floor scan. No database writes occur. Keep this checkpoint until the
    /// corresponding batch is merkleized.
    /// The recipient must still call `verify` with its trusted pre-commitment and the
    /// proposed post-commitment; proof construction does not authorize a transition.
    ///
    /// `policy` must be the floor policy [`merkleize`](Self::merkleize) will use, in the same
    /// state. Proof construction walks a copy of it, so the caller's policy does not change.
    /// Policies a verifier cannot replay fail with [`BuildError::UnsupportedFloorPolicy`].
    ///
    /// # Complexity
    ///
    /// Use `n`, `m`, `k`, and `h` from the [module complexity discussion](super#complexity).
    /// Let `e` be the bitmap units produced, explicit chunks and zero subtrees, `w` the
    /// number of old bitmap bits examined, `l` the largest translated-key bucket read, and
    /// `i` the cost of one index lookup. With fixed-size items and unit-cost storage reads, a
    /// conservative CPU bound is
    /// `O(k * (i + l * log(l + 2)) + w + t * log(t + 2))`,
    /// where `t = (k + e + 1) * h`. This includes local authentication of generated
    /// proofs. Peak additional memory, including the output proof, is O(t + l).
    /// The already resident database/index and backend read buffers are excluded.
    /// There are O(k * l + (k + e + 1) * h) record/node reads; cache misses and
    /// decompression determine their actual CPU and I/O cost. Bucket sorting occurs
    /// in predecessor lookup and range streaming, even for a short requested range.
    ///
    /// For well-distributed keys with bounded bucket occupancy, ordinary built-in
    /// index lookup costs O(log(m + 2)) on average. If the floor crosses O(k) chunks,
    /// the typical bound is O(k * log(m + 2) + k * h * log(k * h + 2)) CPU and
    /// O(k * h) memory. This assumes fixed `N` and is not a distribution-free average.
    /// With only `B` possible translated keys, uniform keys have expected bucket
    /// occupancy proportional to `1 + m / B`; uniformity alone does not keep buckets
    /// bounded as `m` grows. The collision terms above must then be retained.
    /// Worst-case collisions allow `l = m` and index work O(m); a long inactive scan
    /// allows `w = O(n)`, but zero subtrees keep `e = O(k * log(n + 2))`: each clear run
    /// between the O(k) chunks sent in full takes O(log(n + 2)) blocks. Substituting those
    /// values gives O(k * m * log(m + 2) + n + t * log(t + 2)) CPU and O(t + m) memory.
    /// Custom `I`, `C`, and `S` implementations can add costs beyond this model.
    /// Uses the database's strategy for independent proof authentication and hashing.
    /// The key-proof and floor-candidate storage requests all run concurrently. Member proofs
    /// reuse the location and successor found during request discovery. The caller polls these
    /// I/O futures on its storage runtime; Rayon workers perform synchronous CPU work. No Tokio
    /// API is used here. Total work retains the bounds above. Running the P requests together
    /// adds up to O(P * l) memory for maximum bucket size l; output order is deterministic.
    pub async fn build_batch_proof<F, E, C, V, I, H, P, const N: usize, S>(
        &self,
        db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
        policy: &P,
    ) -> Result<BatchProofBundle<F, K, V, H, N>, BuildError<F>>
    where
        F: Graftable,
        E: Context,
        C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
        V: ValueEncoding<Value = A>,
        I: OrderedIndex<Value = Location<F>>,
        H: Hasher,
        S: Strategy,
        P: Policy<F, K, A> + Clone,
        Operation<F, OrderedUpdate<K, V>>: Codec,
    {
        validate_profile::<H, N>();
        let commitment = CurrentCommitment::from_parts(db.bounds().end, db.root())?;
        let (pre_state, old_floor) = build_opening(db, commitment).await?;
        let pre = authenticate_pre_state(commitment, &pre_state)?;
        let expanded = self.expand_from_db(db).await?;
        // Lower endpoints establish where each authenticated range walk must begin,
        // including an excluded endpoint whose member must survive.
        let keys = expanded
            .keys()
            .cloned()
            .chain(self.iter().filter_map(|op| match op.start() {
                Bound::Included(key) | Bound::Excluded(key) => Some(key.clone()),
                Bound::Unbounded => None,
            }))
            .collect::<BTreeSet<_>>();
        // A range with no lower bound begins at the least key. Its predecessor is the
        // greatest key, whose successor link wraps to it.
        let first = if matches!(self.first().map(UserOp::start), Some(Bound::Unbounded)) {
            let stream = db.keys(..);
            pin_mut!(stream);
            stream.next().await.transpose()?
        } else {
            None
        };
        let deleted = expanded
            .iter()
            .filter(|(_, value)| value.is_none())
            .map(|(key, _)| key)
            .collect::<BTreeSet<_>>();
        // Discover and deduplicate requests before constructing their Merkle paths.
        // A deleted key's predecessor may already be another requested member.
        let mut requests = BTreeMap::new();
        for key in keys {
            if let Some((update, location)) = db.any.get_with_loc(&key).await? {
                requests.insert(key.clone(), Some((location, update)));
                if deleted.contains(&key) {
                    let (location, predecessor) = predecessor_record(db, &key).await?;
                    requests.insert(predecessor.key.clone(), Some((location, predecessor)));
                }
            } else {
                requests.insert(key, None);
            }
        }
        if let Some(first) = first {
            let (location, last) = predecessor_record(db, &first).await?;
            requests.insert(last.key.clone(), Some((location, last)));
        }
        let pre_state_claims = try_join_all(requests.into_iter().map(|(key, value)| async move {
                Ok::<_, BuildError<F>>(match value {
                    Some((location, update)) => PreStateKeyClaim::Member(PreStateMemberClaim {
                        proof: ordered::proof::constant::KeyValueProof {
                            proof: db.operation_proof(location).await?,
                            next_key: update.next_key,
                        },
                        key,
                        value: update.value,
                    }),
                    None => PreStateKeyClaim::Exclusion(PreStateExclusionClaim {
                        proof: db.exclusion_proof(&key).await?,
                        queried_key: key,
                    }),
                })
            }))
            .await?;
        let mut cache = TrustedMap::empty_cache(commitment);
        if pre.old_floor == pre.terminal_location {
            cache.set_state(super::TrustedState::EmptyDatabase);
        }
        verify_and_merge_claims(&pre, &mut cache, &pre_state_claims, db.strategy())?;
        let points = expand_range_deletes(&cache, self.ops())?;
        let prepared = derive_ordered_transition(&cache, &points)?;
        check_policy(policy, prepared.made_inactive())
            .map_err(|_| BuildError::UnsupportedFloorPolicy)?;
        let (scan_end, candidates) =
            build_candidate_proof(db, &prepared, commitment.size(), old_floor, policy.clone())
                .await?;
        let bitmap = build_bitmap_proof::<F, K, V, H, _, _, _, N>(
            commitment.size(),
            old_floor,
            scan_end,
            &prepared,
            db.bitmap(),
            &db.any.log.merkle,
            &db.grafted_storage(),
        )
        .await?;
        let view = verify_bitmap(&pre, &prepared, &bitmap, db.strategy())?;
        let verified = verify_candidates(&pre, &view, &candidates)?;
        let floor = advance_floor(
            &pre,
            &prepared,
            &view,
            &verified,
            self.metadata().cloned(),
            &mut policy.clone(),
        )?;
        Ok(BatchProofBundle::new(
            ClaimedBatchBounds {
                pre_size: commitment.size(),
                pre_floor: old_floor,
                post_size: floor.child_size,
                post_floor: floor.floor,
            },
            pre_state,
            pre_state_claims,
            bitmap,
            candidates,
        ))
    }

    /// Merkleize the same normalized mutations through the ordinary database writer.
    /// Range deletes enumerate the database's active keys; surviving sets have already
    /// split those ranges. This returns an unapplied batch, so its root can be sent to
    /// a verifier before the caller chooses to apply it. Use the same checkpoint as
    /// `build_batch_proof`; a proof from another checkpoint cannot authenticate it.
    ///
    /// The writer's walk decides with `policy` as [`build_batch_proof`](Self::build_batch_proof)
    /// replayed it, advancing a policy that keeps state. As with the writer, that state is
    /// provisional until the batch is applied.
    ///
    /// # Complexity
    ///
    /// This wrapper adds range expansion and O(k * log(k + 2)) ordered-set/batch
    /// insertion work, with O(k + l) temporary memory, before delegating to the
    /// ordinary Current writer. Here `k = b + r + 1` and `l` is the largest bucket
    /// read. Expansion takes O(k * (i + l * log(l + 2))) CPU and O(k * l) record
    /// reads in the model used by `build_batch_proof`: typically O(k * log(m + 2)),
    /// but up to O(k * m * log(m + 2)) under worst-case collisions. Add the writer's
    /// Merkle hashing, floor scan, output batch, backend, and strategy costs to these
    /// wrapper bounds; this method does not build or verify a proof bundle.
    #[allow(clippy::type_complexity)]
    pub async fn merkleize<F, E, C, V, I, H, P, const N: usize, S>(
        &self,
        db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
        policy: &mut P,
    ) -> Result<
        std::sync::Arc<
            crate::qmdb::current::batch::MerkleizedBatch<F, H::Digest, OrderedUpdate<K, V>, N, S>,
        >,
        qmdb::Error<F>,
    >
    where
        F: Graftable,
        E: Context,
        C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
        V: ValueEncoding<Value = A>,
        I: OrderedIndex<Value = Location<F>> + 'static,
        H: Hasher,
        S: Strategy,
        P: Policy<F, K, A> + Send,
        Operation<F, OrderedUpdate<K, V>>: Codec,
    {
        let mut batch = db.new_batch();
        for (key, value) in self.expand_from_db(db).await? {
            batch = batch.write(key, value.cloned());
        }
        batch.merkleize(db, self.metadata().cloned(), policy).await
    }

    /// Enumerate range members for proof requests and the ordinary writer, returning each
    /// mutated key with the value a set writes or `None` for a delete. This is a storage
    /// lookup, not a proof: the verifier independently checks completeness by following
    /// authenticated successor links from each range's lower endpoint.
    async fn expand_from_db<F, E, C, V, I, H, const N: usize, S>(
        &self,
        db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
    ) -> Result<BTreeMap<K, Option<&A>>, qmdb::Error<F>>
    where
        F: Graftable,
        E: Context,
        C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
        V: ValueEncoding<Value = A>,
        I: OrderedIndex<Value = Location<F>>,
        H: Hasher,
        S: Strategy,
        Operation<F, OrderedUpdate<K, V>>: Codec,
    {
        let mut expanded = BTreeMap::new();
        for op in self.iter() {
            match op {
                UserOp::Set { key, value } => {
                    expanded.insert(key.clone(), Some(value));
                }
                UserOp::Delete { key } => {
                    expanded.insert(key.clone(), None);
                }
                UserOp::DeleteRange { start, end } => {
                    let stream = db.stream_range((start.clone(), end.clone()));
                    pin_mut!(stream);
                    while let Some(record) = stream.next().await {
                        let (key, _) = record?;
                        expanded.insert(key, None);
                    }
                }
            }
        }
        Ok(expanded)
    }
}

/// Locate the immediate cyclic predecessor needed when deleting a member.
/// The index groups keys by translated prefixes, so inspect both the member's bucket
/// and the previous bucket (wrapping at the minimum), then match the exact successor.
/// Its ordinary membership proof will authenticate this link for the recipient.
/// For a bucket of at most `l` locations, O(i + l * log(l + 2)) CPU, O(l)
/// temporary locations, and O(l) record reads; `i` is index-lookup work. Bounded
/// occupancy gives O(i) typical work, while all `m` active keys can share a bucket.
async fn predecessor_record<F, E, C, K, V, I, H, const N: usize, S>(
    db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
    key: &K,
) -> Result<ExactRecord<F, K, V>, BuildError<F>>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
    let mut locations = db.any.snapshot.get(key).copied().collect::<Vec<_>>();
    if let Some((iter, _)) = db.any.snapshot.prev_translated_key(key) {
        locations.extend(iter.copied());
    }
    locations.sort();
    locations.dedup();
    for location in locations {
        let operation = db
            .any
            .log
            .read(*location)
            .await
            .map_err(qmdb::Error::Journal)?;
        let Operation::Update(update) = operation else {
            return Err(BuildError::InvalidState(
                "the active ordered index referenced a non-update operation",
            ));
        };
        if update.next_key == *key {
            return Ok((location, update));
        }
    }
    Err(BuildError::InvalidState(
        "the active ordered cycle has no predecessor for a member",
    ))
}

/// Obtain the terminal's raw inclusion proof and the frontier immediately before it.
/// The root witness binds the raw log to Current's bitmap-aware commitment. Together
/// these let the verifier authenticate the old floor and reconstruct the append tip.
async fn build_opening<F, E, C, K, V, I, H, const N: usize, S>(
    db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
    commitment: CurrentCommitment<F, K, V, H, N>,
) -> Result<(PreStateOpeningClaim<F, V, H::Digest, N>, Location<F>), BuildError<F>>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
    if commitment.size() == Location::new(1) {
        return Ok((PreStateOpeningClaim::Initial, Location::new(0)));
    }
    let terminal_location = commitment.size() - 1;
    let (raw_proof, mut operations) = db
        .ops_historical_proof(commitment.size(), terminal_location, NZU64!(1))
        .await?;
    let Some(Operation::CommitFloor(metadata, floor)) = operations.pop() else {
        return Err(BuildError::InvalidState(
            "the committed operation log does not end in CommitFloor",
        ));
    };
    let witness = db.ops_root_witness().await?;
    let pins_before_terminal = db.pinned_nodes_at(terminal_location).await?;
    Ok((
        PreStateOpeningClaim::Proven {
            ops_root: db.ops_root(),
            witness,
            terminal: PreStateTerminalClaim {
                metadata,
                floor,
                raw_proof,
                pins_before_terminal,
            },
        },
        floor,
    ))
}

/// Read old active records through the floor scan's required stopping point.
/// Include superseded records too: their pre-state bits remain set until replay.
/// Read as many live records as the floor policy's entry limit allows, then keep those
/// the policy's walk consumes: stop where it ends, or cover the entire old log if
/// copies must continue into the application suffix. A raw multi-proof authenticates
/// these payloads; the bitmap proof separately establishes their completeness.
async fn build_candidate_proof<F, E, C, K, V, I, H, P, const N: usize, S>(
    db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
    prepared: &super::ordered::PreparedOrderedBatch<F, K, V>,
    pre_size: Location<F>,
    old_floor: Location<F>,
    mut policy: P,
) -> Result<(Location<F>, PreStateCandidateProof<F, K, V, H::Digest>), BuildError<F>>
where
    F: Graftable,
    E: Context,
    C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
    K: Key,
    V: ValueEncoding,
    I: OrderedIndex<Value = Location<F>>,
    H: Hasher,
    S: Strategy,
    P: Policy<F, K, V::Value>,
    Operation<F, OrderedUpdate<K, V>>: Codec,
{
    if prepared.post_empty {
        return Ok((
            old_floor,
            PreStateCandidateProof {
                operations: Vec::new(),
                raw_multi_proof: None,
            },
        ));
    }

    // Read ahead as many live records as the policy's entries allow, then let the walk itself
    // say where it ends: a policy may stop early, so the read-ahead can exceed the scan.
    let terminal_location = pre_size - 1;
    let entries =
        u64::try_from(policy.limits(prepared.made_inactive()).entries).unwrap_or(u64::MAX);
    let mut locations = Vec::new();
    let mut live = 0u64;
    if entries > 0 {
        for raw_location in db
            .bitmap()
            .ones_iter_from(old_floor.as_u64())
            .take_while(|location| *location < pre_size.as_u64())
        {
            let location = Location::new(raw_location);
            if location == terminal_location {
                continue;
            }
            locations.push(location);
            if prepared.superseded.contains(&location) {
                continue;
            }
            live += 1;
            if live == entries {
                break;
            }
        }
    }

    // Read that ordered location list concurrently; every surviving bit must still resolve to
    // an update. Verification later authenticates all payloads.
    let mut operations = try_join_all(locations.iter().copied().map(|location| async move {
            let operation = db
                .any
                .log
                .read(location.as_u64())
                .await
                .map_err(qmdb::Error::Journal)?;
            if !prepared.superseded.contains(&location)
                && !matches!(operation, Operation::Update(_))
            {
                return Err(BuildError::InvalidState(
                    "a non-terminal active bitmap bit references a non-update operation",
                ));
            }
            Ok((location, operation))
        }))
        .await?;

    let updates = operations
        .iter()
        .filter_map(|(location, operation)| match operation {
            Operation::Update(update) => Some((*location, update)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let walked = walk_floor(old_floor, pre_size, prepared, &updates, &mut policy)?;
    let floor_scan_end = walked.scan_end;
    operations.retain(|(location, _)| *location < floor_scan_end);
    let locations = operations
        .iter()
        .map(|(location, _)| *location)
        .collect::<Vec<_>>();

    let raw_multi_proof = if operations.is_empty() {
        None
    } else {
        Some(
            merkle::verification::multi_proof(
                &db.any.log.merkle,
                F::inactive_peaks(pre_size, old_floor),
                &locations,
            )
            .await?,
        )
    };
    Ok((
        floor_scan_end,
        PreStateCandidateProof {
            operations,
            raw_multi_proof,
        },
    ))
}
