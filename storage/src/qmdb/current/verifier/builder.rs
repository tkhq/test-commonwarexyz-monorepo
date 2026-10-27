//! Proof production from a committed storage-backed ordered Current database.

use super::{
    BatchProofBundle, BuildError, ClaimedBatchBounds, CurrentCommitment, PreStateCandidateProof,
    PreStateExclusionClaim, PreStateKeyClaim, PreStateMemberClaim, PreStateOpeningClaim,
    PreStateTerminalClaim, UserBatch, UserOp, VerifyError,
    bitmap::build_bitmap_proof,
    commitment::validate_profile,
    floor::{check_policy, walk_floor},
    ordered::PreparedOrderedBatch,
    trusted::strict_span_contains,
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
            batch::DiffEntry,
            operation::{Operation, update},
        },
        current::{batch::MerkleizedBatch, ordered},
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
    sync::Arc,
};

type OrderedUpdate<K, V> = update::Ordered<K, V>;
type ExactRecord<F, K, V> = (Location<F>, OrderedUpdate<K, V>);

impl<K: Key, A: Clone + Send + Sync> UserBatch<K, A> {
    /// Merkleize this batch through the ordinary database writer and build its proof bundle
    /// from the merkleized batch, against this database's current committed checkpoint.
    ///
    /// The bundle works with an empty trusted cache. Member and exclusion claims cover the
    /// requested keys, every member of a delete range, and affected predecessors. A range
    /// with no lower bound adds the greatest key's record, whose successor link proves where
    /// the range begins. One raw multi-proof covers every claimed record and floor
    /// candidate, and the bitmap proof covers the floor scan and each claimed record's
    /// chunk. No database writes occur, and the returned batch is unapplied.
    /// The recipient must still call `verify` with its trusted pre-commitment and the
    /// batch's post-commitment; proof construction does not authorize a transition.
    /// With the `prover-self-check` feature the bundle is also verified before it is returned;
    /// otherwise the prover trusts its own database and the recipient's check is the only one.
    ///
    /// The writer resolves every key once. The bundle reads each pre-state record the batch
    /// superseded or moved at the location the batch recorded for it, and finds the claims'
    /// records, the predecessors of deleted keys and the span of each created key among
    /// them. Only a range endpoint the batch leaves alone, or an absent key whose span no
    /// such record covers, needs an index lookup.
    ///
    /// `policy` advances as [`merkleize`](Self::merkleize) advances it; the proof replays the
    /// walk with a copy taken before. Policies a verifier cannot replay fail with
    /// [`BuildError::UnsupportedFloorPolicy`], an evicting one before the batch is merkleized.
    ///
    /// # Complexity
    ///
    /// Use `n`, `m`, `k`, and `h` from the [module complexity discussion](super#complexity),
    /// and let `d` be the number of keys in the batch's diff: mutated keys, rewritten
    /// predecessors, and records the floor walk moved. On top of `merkleize`, this takes O(d)
    /// record reads, one raw multi-proof over the claims and candidates, the floor replay over
    /// the scanned candidates, the bitmap proof, and O(d * log(d + 2)) ordered-set work, with
    /// O(d + t) additional memory for `t = (k + e + 1) * h` proof digests and `e` bitmap
    /// units. An
    /// index lookup, O(i + l * log(l + 2)) for index work `i` and largest bucket `l`, happens
    /// only for the keys named above. The record reads run concurrently.
    #[allow(clippy::type_complexity)]
    pub async fn prove<F, E, C, V, I, H, P, const N: usize, S>(
        &self,
        db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
        policy: &mut P,
    ) -> Result<
        (
            BatchProofBundle<F, K, V, H, N>,
            Arc<MerkleizedBatch<F, H::Digest, OrderedUpdate<K, V>, N, S>>,
        ),
        BuildError<F>,
    >
    where
        F: Graftable,
        E: Context,
        C: Mutable<Item = Operation<F, OrderedUpdate<K, V>>>,
        V: ValueEncoding<Value = A>,
        I: OrderedIndex<Value = Location<F>> + 'static,
        H: Hasher,
        S: Strategy,
        P: Policy<F, K, A> + Clone + Send,
        Operation<F, OrderedUpdate<K, V>>: Codec,
    {
        validate_profile::<H, N>();
        let commitment = CurrentCommitment::from_parts(db.bounds().end, db.root())?;
        if policy.evicts() {
            return Err(BuildError::UnsupportedFloorPolicy);
        }
        let replay = policy.clone();
        let expanded = self.expand_from_db(db).await?;
        let mut writer = db.new_batch();
        for (key, value) in &expanded {
            writer = writer.write(key.clone(), value.cloned());
        }
        let batch = writer
            .merkleize(db, self.metadata().cloned(), policy)
            .await?;

        let (pre_state, old_floor) = build_opening(db, commitment).await?;
        let old = read_old_records(db, &batch.inner.diff).await?;
        let prepared = prepare_from_batch(&expanded, &batch, &old)?;
        check_policy(&replay, prepared.made_inactive())
            .map_err(|_| BuildError::UnsupportedFloorPolicy)?;
        let pre_state_claims = self.claims(db, &expanded, &prepared, &old).await?;

        let known = old
            .iter()
            .map(|(location, update)| (*location, update))
            .collect::<BTreeMap<_, _>>();
        #[cfg(feature = "prover-self-check")]
        let mut check = replay.clone();
        let walked = build_candidate_proof(
            db,
            &prepared,
            &known,
            commitment.size(),
            old_floor,
            replay,
        )
        .await?;
        // One raw multi-proof covers every record the claims and candidates carry.
        let claimed = pre_state_claims
            .iter()
            .map(|claim| match claim {
                PreStateKeyClaim::Member(claim) => claim.location,
                PreStateKeyClaim::Exclusion(claim) => claim.location,
            })
            .collect::<BTreeSet<_>>();
        let mut records = claimed.clone();
        records.extend(walked.proof.operations.iter().map(|(location, _)| *location));
        let records = records.into_iter().collect::<Vec<_>>();
        let records_proof = if records.is_empty() {
            None
        } else {
            Some(
                merkle::verification::multi_proof(
                    &db.any.log.merkle,
                    F::inactive_peaks(commitment.size(), old_floor),
                    &records,
                )
                .await?,
            )
        };
        let bounds = batch.bounds();
        if child_size(commitment.size(), &prepared, walked.copies)? != bounds.tip.size
            || walked.floor != bounds.inactivity_floor
        {
            return Err(BuildError::InvalidState(
                "the prover's floor replay differs from the batch's walk",
            ));
        }
        let bitmap = build_bitmap_proof::<F, K, V, H, _, _, _, N>(
            commitment.size(),
            old_floor,
            walked.scan_end,
            &prepared,
            &claimed,
            db.bitmap(),
            &db.any.log.merkle,
            &db.grafted_storage(),
        )
        .await?;
        let bundle = BatchProofBundle::new(
            ClaimedBatchBounds {
                pre_size: commitment.size(),
                pre_floor: old_floor,
                post_size: bounds.tip.size,
                post_floor: bounds.inactivity_floor,
            },
            pre_state,
            pre_state_claims,
            bitmap,
            walked.proof,
            records_proof,
        );
        #[cfg(feature = "prover-self-check")]
        {
            let post = CurrentCommitment::from_parts(bounds.tip.size, batch.root())?;
            self.verify_with_strategy(
                commitment,
                super::TrustedMap::empty_cache(commitment),
                &bundle,
                post,
                db.strategy(),
                &mut check,
            )?;
        }
        Ok((bundle, batch))
    }

    /// Construct a complete proof from this database's current committed checkpoint.
    ///
    /// This is [`prove`](Self::prove) with a copy of `policy`, discarding the merkleized
    /// batch: the caller's policy does not change. Use `prove` to keep the batch rather than
    /// merkleize it again. Keep this checkpoint until the corresponding batch is merkleized.
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
        I: OrderedIndex<Value = Location<F>> + 'static,
        H: Hasher,
        S: Strategy,
        P: Policy<F, K, A> + Clone + Send,
        Operation<F, OrderedUpdate<K, V>>: Codec,
    {
        self.prove(db, &mut policy.clone())
            .await
            .map(|(bundle, _)| bundle)
    }

    /// Build the pre-state key claims, in key order, from the batch's old records.
    ///
    /// Claims cover the mutated keys, each range's lower endpoint, the predecessor of every
    /// deleted key, and the greatest key when a range has no lower bound. A key with an old
    /// record is a member. An absent key's exclusion uses the old record whose successor link
    /// spans it: such a record is the key's pre-state span owner, since no pre-state key lies
    /// inside a successor link. Keys the batch did not reveal fall back to the database index.
    /// An empty pre-state needs no claims: its opening establishes that every key is absent.
    async fn claims<F, E, C, V, I, H, const N: usize, S>(
        &self,
        db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
        expanded: &BTreeMap<K, Option<&A>>,
        prepared: &PreparedOrderedBatch<F, K, V>,
        old: &[ExactRecord<F, K, V>],
    ) -> Result<Vec<PreStateKeyClaim<F, K, V>>, BuildError<F>>
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
        let member = |key: &K| {
            old.binary_search_by(|(_, update)| update.key.cmp(key))
                .ok()
                .and_then(|index| old.get(index))
        };
        let owner = |key: &K| {
            let below = old.partition_point(|(_, update)| update.key < *key);
            let owner = below
                .checked_sub(1)
                .and_then(|index| old.get(index))
                .or_else(|| old.last())?;
            strict_span_contains(&owner.1.key, &owner.1.next_key, key).then_some(owner)
        };
        // Only a mutated key the batch's diff lacks can be absent without an old record.
        let request = |key: &K, mutated: bool| match (member(key), owner(key)) {
            (Some(record), _) => Request::Member(record.clone()),
            (None, Some(owner)) => Request::Excluded(owner.clone()),
            (None, None) if mutated => Request::Absent,
            (None, None) => Request::Unknown,
        };
        let mut requests = BTreeMap::new();
        for key in expanded.keys() {
            requests.insert(key.clone(), request(key, true));
        }
        // Lower endpoints establish where each authenticated range walk must begin,
        // including an excluded endpoint whose member must survive.
        for op in self.iter() {
            if let Bound::Included(key) | Bound::Excluded(key) = op.start()
                && !requests.contains_key(key)
            {
                requests.insert(key.clone(), request(key, false));
            }
        }
        // A deleted key's predecessor rewrites or supersedes its record, so it is in the diff.
        let by_next = old
            .iter()
            .map(|record| (&record.1.next_key, record))
            .collect::<BTreeMap<_, _>>();
        let mut predecessors = Vec::new();
        for key in &prepared.deleted_keys {
            predecessors.push(match by_next.get(key) {
                Some(&record) => record.clone(),
                None => predecessor_record(db, key).await?,
            });
        }
        // A range with no lower bound begins at the least key. Its predecessor is the
        // greatest key, whose successor link wraps to it.
        if matches!(self.first().map(UserOp::start), Some(Bound::Unbounded)) {
            let stream = db.keys(..);
            pin_mut!(stream);
            if let Some(first) = stream.next().await.transpose()? {
                predecessors.push(match by_next.get(&first) {
                    Some(&record) => record.clone(),
                    None => predecessor_record(db, &first).await?,
                });
            }
        }
        for record in predecessors {
            requests.insert(record.1.key.clone(), Request::Member(record));
        }

        // Settle the keys the batch did not reveal. An absent key's span owner comes from the
        // index; an empty pre-state has none and needs no claim.
        let claims = try_join_all(requests.into_iter().map(|(key, request)| async move {
                let request = match request {
                    Request::Unknown => match db.any.get_with_loc(&key).await? {
                        Some((update, location)) => Request::Member((location, update)),
                        None => Request::Absent,
                    },
                    request => request,
                };
                let request = match request {
                    Request::Absent => match db.any.get_span(&key).await? {
                        Some(owner) => Request::Excluded(owner),
                        None => return Ok(None),
                    },
                    request => request,
                };
                Ok::<_, BuildError<F>>(Some(match request {
                    Request::Member((location, update)) => {
                        PreStateKeyClaim::Member(PreStateMemberClaim {
                            location,
                            key,
                            value: update.value,
                            next_key: update.next_key,
                        })
                    }
                    Request::Excluded((location, owner)) => {
                        PreStateKeyClaim::Exclusion(PreStateExclusionClaim {
                            queried_key: key,
                            location,
                            owner,
                        })
                    }
                    Request::Absent | Request::Unknown => {
                        return Err(BuildError::InvalidState("an absent key was not resolved"));
                    }
                }))
            }))
            .await?;
        Ok(claims.into_iter().flatten().collect())
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

/// How a claim's key is established in the pre-state.
enum Request<F: Graftable, K: Key, V: ValueEncoding> {
    /// A member, with its location and record.
    Member(ExactRecord<F, K, V>),
    /// An absent key, with the record whose successor link spans it.
    Excluded(ExactRecord<F, K, V>),
    /// An absent key whose span owner is not among the batch's old records.
    Absent,
    /// A key the batch did not touch, whose membership the index decides.
    Unknown,
}

/// Read the pre-state record at every location `diff` records for a key, in the diff's key
/// order: the records the batch supersedes and those its floor walk moves.
async fn read_old_records<F, E, C, K, V, I, H, const N: usize, S>(
    db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
    diff: &[(K, DiffEntry<F, V::Value>)],
) -> Result<Vec<ExactRecord<F, K, V>>, BuildError<F>>
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
    try_join_all(
        diff.iter()
            .filter_map(|(key, entry)| Some((key, entry.base_old_loc()?)))
            .map(|(key, location)| async move {
        match db
            .any
            .log
            .read(*location)
            .await
            .map_err(qmdb::Error::Journal)?
        {
            Operation::Update(update) if update.key == *key => Ok((location, update)),
            _ => Err(BuildError::InvalidState(
                "a batch's base location does not hold its key's update",
            )),
        }
    }),
    )
    .await
}

/// Recover from a merkleized batch the transition a verifier derives for it.
///
/// The writer emits the application operations first: deletes, sets, then the predecessor
/// rewrites that keep successor links intact. Floor copies and the commit follow. A key the
/// caller did not mutate is in the diff either as a rewritten predecessor, whose final
/// successor differs from its old one, or as a record the floor walk moved unchanged. `old`
/// holds the record at each entry's base location, in the diff's key order.
fn prepare_from_batch<F, K, V, D, const N: usize, S>(
    expanded: &BTreeMap<K, Option<&V::Value>>,
    batch: &MerkleizedBatch<F, D, OrderedUpdate<K, V>, N, S>,
    old: &[ExactRecord<F, K, V>],
) -> Result<PreparedOrderedBatch<F, K, V>, BuildError<F>>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: commonware_cryptography::Digest,
    S: Strategy,
{
    const UNEXPLAINED: &str = "the batch changed a key its mutations do not explain";
    let (base, operations) = batch.operations();
    let mut old = old.iter();
    let mut superseded = BTreeSet::new();
    let mut deleted_keys = BTreeSet::new();
    let mut count = 0usize;
    for (key, entry) in batch.inner.diff.iter() {
        let record = entry.base_old_loc().and_then(|_| old.next());
        match (expanded.get(key), entry, record) {
            (Some(None), DiffEntry::Deleted { .. }, Some((location, _))) => {
                deleted_keys.insert(key.clone());
                superseded.insert(*location);
            }
            (Some(Some(_)), DiffEntry::Active { .. }, record) => {
                superseded.extend(record.map(|(location, _)| *location));
            }
            (None, DiffEntry::Active { loc, .. }, Some((location, update))) => {
                let written = loc
                    .checked_sub(*base)
                    .and_then(|offset| usize::try_from(*offset).ok())
                    .and_then(|offset| operations.get(offset));
                let Some(Operation::Update(written)) = written else {
                    return Err(BuildError::InvalidState(UNEXPLAINED));
                };
                if written.next_key == update.next_key {
                    continue;
                }
                superseded.insert(*location);
            }
            _ => return Err(BuildError::InvalidState(UNEXPLAINED)),
        }
        count += 1;
    }
    let application_ops = operations
        .get(..count)
        .ok_or(BuildError::InvalidState(UNEXPLAINED))?
        .to_vec();
    Ok(PreparedOrderedBatch {
        application_ops,
        superseded,
        deleted_keys,
        post_empty: batch.inner.total_active_keys == 0,
    })
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

/// The child's size: the application operations, the floor walk's `copies`, and the terminal
/// commit appended to the pre-state's `base_size` operations.
fn child_size<F: Graftable, K: Key, V: ValueEncoding>(
    base_size: Location<F>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    copies: usize,
) -> Result<Location<F>, BuildError<F>> {
    u64::try_from(prepared.application_ops.len())
        .ok()
        .and_then(|operations| operations.checked_add(u64::try_from(copies).ok()?))
        .and_then(|appended| base_size.checked_add(appended)?.checked_add(1))
        .ok_or(BuildError::Inconsistent(VerifyError::Overflow))
}

/// Floor candidates read for one batch, and what the floor walk over them decided.
struct CandidateEvidence<F: Graftable, K: Key, V: ValueEncoding> {
    /// The end of the old-log coverage the walk consumed.
    scan_end: Location<F>,
    /// The child's inactivity floor.
    floor: Location<F>,
    /// The number of live records the walk copies to the tip.
    copies: usize,
    /// The candidates' payloads.
    proof: PreStateCandidateProof<F, K, V>,
}

/// Read old active records through the floor scan's required stopping point.
/// Include superseded records too: their pre-state bits remain set until replay.
/// Records in `known`, read already at their locations, are not read again.
/// Read as many live records as the floor policy's entry limit allows, then keep those
/// the policy's walk consumes: stop where it ends, or cover the entire old log if
/// copies must continue into the application suffix. The bundle's shared raw multi-proof
/// authenticates these payloads; the bitmap proof separately establishes their completeness.
async fn build_candidate_proof<F, E, C, K, V, I, H, P, const N: usize, S>(
    db: &ordered::db::Db<F, E, C, K, V, I, H, N, S>,
    prepared: &PreparedOrderedBatch<F, K, V>,
    known: &BTreeMap<Location<F>, &OrderedUpdate<K, V>>,
    pre_size: Location<F>,
    old_floor: Location<F>,
    mut policy: P,
) -> Result<CandidateEvidence<F, K, V>, BuildError<F>>
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
        // The child is empty: its floor is the terminal, right after the application operations.
        return Ok(CandidateEvidence {
            scan_end: old_floor,
            floor: child_size(pre_size, prepared, 0)? - 1,
            copies: 0,
            proof: PreStateCandidateProof {
                operations: Vec::new(),
            },
        });
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
            let operation = match known.get(&location) {
                Some(&update) => Operation::Update(update.clone()),
                None => db
                    .any
                    .log
                    .read(location.as_u64())
                    .await
                    .map_err(qmdb::Error::Journal)?,
            };
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
    Ok(CandidateEvidence {
        scan_end: floor_scan_end,
        floor: walked.floor,
        copies: walked.copies.len(),
        proof: PreStateCandidateProof { operations },
    })
}
