//! Caller-authorized mutations, kept normalized as the batch is edited.

use super::{
    BatchProofBundle, CurrentCommitment, TrustedMap, Verified, VerifyError,
    transition::verify_batch,
};
use crate::{
    merkle::Graftable,
    qmdb::{
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        floor::Policy,
        operation::Key,
    },
};
use alloc::{
    collections::{BTreeSet, btree_set},
    vec,
    vec::Vec,
};
use commonware_codec::Codec;
use commonware_cryptography::Hasher;
use core::{
    cmp::Ordering,
    iter::FusedIterator,
    ops::{Bound, RangeBounds},
};

/// Return a bound's key and whether the range includes it, or `None` for an unbounded side.
const fn finite<K>(bound: &Bound<K>) -> Option<(&K, bool)> {
    match bound {
        Bound::Included(key) => Some((key, true)),
        Bound::Excluded(key) => Some((key, false)),
        Bound::Unbounded => None,
    }
}

/// Order lower bounds by the first key each admits: unbounded first, then by key, with an
/// included key before the range that starts just after it.
fn cmp_start<K: Ord>(a: &Bound<K>, b: &Bound<K>) -> Ordering {
    match (finite(a), finite(b)) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some((a, a_included)), Some((b, b_included))) => {
            a.cmp(b).then_with(|| b_included.cmp(&a_included))
        }
    }
}

/// Order upper bounds by the last key each admits: by key, with a range that stops short of
/// a key before the one that includes it, and unbounded last.
fn cmp_end<K: Ord>(a: &Bound<K>, b: &Bound<K>) -> Ordering {
    match (finite(a), finite(b)) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some((a, a_included)), Some((b, b_included))) => {
            a.cmp(b).then_with(|| a_included.cmp(&b_included))
        }
    }
}

/// Return whether a range could contain a key: an unbounded side always could, since the key
/// type need not have a least or greatest value.
fn nonempty<K: Ord>(start: &Bound<K>, end: &Bound<K>) -> bool {
    match (finite(start), finite(end)) {
        (Some((start, start_included)), Some((end, end_included))) => {
            start < end || (start == end && start_included && end_included)
        }
        _ => true,
    }
}

/// Return whether a range that ends at `end` overlaps or adjoins one that starts at `start`
/// no earlier than it does. At a shared key they join if either includes it.
fn joins<K: Ord>(end: &Bound<K>, start: &Bound<K>) -> bool {
    match (finite(end), finite(start)) {
        (Some((end, end_included)), Some((start, start_included))) => {
            start < end || (start == end && (start_included || end_included))
        }
        _ => true,
    }
}

/// Return the bound that also includes its own key.
const fn closed<K>(bound: &Bound<K>) -> Bound<&K> {
    match bound {
        Bound::Included(key) | Bound::Excluded(key) => Bound::Included(key),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// Return whether the range from `start` to `end` contains `key`.
pub(super) fn range_contains<K: Ord>(start: &Bound<K>, end: &Bound<K>, key: &K) -> bool {
    (start.as_ref(), end.as_ref()).contains(key)
}

/// A point mutation or a non-wrapping delete interval.
///
/// Batch insertion applies operations in arrival order: the last operation covering
/// a key wins. Range endpoints may be excluded so a later set can split a range
/// without requiring arithmetic on keys. A range that contains no key changes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UserOp<K, V> {
    /// Set or create a key with the supplied value.
    Set { key: K, value: V },
    /// Delete a key if it exists.
    Delete { key: K },
    /// Delete every member between these bounds. Excluded endpoints survive; an unbounded
    /// side reaches the first or last key.
    DeleteRange { start: Bound<K>, end: Bound<K> },
}

impl<K, V> UserOp<K, V> {
    /// Construct a set mutation.
    pub const fn set(key: K, value: V) -> Self {
        Self::Set { key, value }
    }

    /// Construct a point delete.
    pub const fn delete(key: K) -> Self {
        Self::Delete { key }
    }

    /// Construct a range delete, such as `start..=end`, `start..`, or `..`. A range that
    /// contains no key, including a reversed one, deletes nothing.
    pub fn delete_range(range: impl RangeBounds<K>) -> Self
    where
        K: Clone,
    {
        Self::DeleteRange {
            start: range.start_bound().cloned(),
            end: range.end_bound().cloned(),
        }
    }

    /// Return where the operation starts: a point operation's key, or a range's lower bound.
    pub const fn start(&self) -> Bound<&K> {
        match self {
            Self::Set { key, .. }
            | Self::Delete { key }
            | Self::DeleteRange {
                start: Bound::Included(key),
                ..
            } => Bound::Included(key),
            Self::DeleteRange {
                start: Bound::Excluded(key),
                ..
            } => Bound::Excluded(key),
            Self::DeleteRange {
                start: Bound::Unbounded,
                ..
            } => Bound::Unbounded,
        }
    }

    /// Distinguish an interval starting after a key from a point operation at that key.
    const fn excludes_start(&self) -> bool {
        matches!(
            self,
            Self::DeleteRange {
                start: Bound::Excluded(_),
                ..
            }
        )
    }
}

/// An operation ordered by where it starts: a range with no lower bound first, then by key,
/// then whether a range excludes that key.
///
/// A batch holds at most one operation per start, so inserting an operation replaces the one
/// that starts where it does, whatever its kind or value.
#[derive(Clone, Debug)]
pub(super) struct ByStart<K, V>(pub(super) UserOp<K, V>);

impl<K: Clone, V> ByStart<K, V> {
    /// The probe for the operation starting at `key`: a point mutation or an included range.
    const fn at(key: K) -> Self {
        Self(UserOp::delete(key))
    }

    /// The probe for the range starting just after `key`.
    fn after(key: K) -> Self {
        Self(UserOp::DeleteRange {
            start: Bound::Excluded(key.clone()),
            end: Bound::Included(key),
        })
    }
}

impl<K: Ord, V> PartialEq for ByStart<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl<K: Ord, V> Eq for ByStart<K, V> {}
impl<K: Ord, V> PartialOrd for ByStart<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord, V> Ord for ByStart<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_start(&self.0.start(), &other.0.start())
    }
}

/// Caller-authorized mutations and metadata for one database commit.
///
/// Every edit preserves a normalized representation: operations are ordered by where
/// they start, delete intervals are disjoint, and point mutations lie outside them.
/// A later set splits a covering range immediately. A later delete removes the set
/// and joins adjoining delete intervals. Single-key intervals become point deletes.
/// Iteration includes ranges and can reconstruct the batch without losing mutations.
/// Removing an operation removes its final effect; earlier requests are not restored.
///
/// # Editing complexity
///
/// With `b` stored operations and fixed-size keys, `set` takes O(log(b + 2)) time
/// and O(1) additional memory, including splitting one range. A point `delete` has
/// the same cost unless it touches a range; that case, and every `delete_range`,
/// rebuilds delete coverage in O(b * log(b + 2)) time and O(b) temporary memory.
/// Stored memory is O(b), plus value payloads. These are deterministic bounds.
/// Point-only workloads take the logarithmic path; no average bound independent of
/// the frequency of range edits is implied. Extending or appending `a` operations
/// applies these costs per edit: O(a * log(a + b + 2)) for point-only batches, up to
/// O(a * (a + b) * log(a + b + 2)) with ranges, and O(a + b) peak operation storage.
///
/// Verification authenticates every member covered by each range before deriving the
/// physical operations, predecessor rewrites, and floor copies. Even an empty batch
/// derives a terminal commit and may advance the floor. The caller must authorize
/// the mutations and metadata before verification.
///
/// # Examples
///
/// A set splits a range immediately; deleting that key joins it again:
///
/// ```
/// use commonware_storage::qmdb::current::verifier::{UserBatch, UserOp};
///
/// let mut batch = UserBatch::from([
///     UserOp::delete_range(1u8..=9),
///     UserOp::set(5, 42u8),
/// ]);
/// assert_eq!(batch.len(), 3); // [1, 5), Set(5, 42), (5, 9]
/// batch.delete(5);
/// assert_eq!(batch.len(), 1); // [1, 9]
/// batch.set(0, 7);
/// batch.delete_range(3..).unwrap();
/// assert_eq!(batch.len(), 2); // Set(0, 7), [1, ..)
/// ```
#[derive(Clone, Debug)]
pub struct UserBatch<K, V> {
    user_ops: BTreeSet<ByStart<K, V>>,
    metadata: Option<V>,
}

impl<K: Ord + Clone, V> UserBatch<K, V> {
    /// Construct an empty batch with no metadata.
    pub const fn new() -> Self {
        Self {
            user_ops: BTreeSet::new(),
            metadata: None,
        }
    }

    /// Set a key, replacing its previous mutation and splitting any covering range.
    pub fn set(&mut self, key: K, value: V) {
        let covering = self
            .user_ops
            .range(..=ByStart::at(key.clone()))
            .next_back()
            .and_then(|op| match &op.0 {
                UserOp::DeleteRange { start, end } if range_contains(start, end, &key) => {
                    Some((start.clone(), end.clone()))
                }
                _ => None,
            });
        if let Some((start, end)) = covering {
            self.user_ops.remove(&ByStart(UserOp::DeleteRange {
                start: start.clone(),
                end: end.clone(),
            }));
            let before = Bound::Excluded(key.clone());
            if nonempty(&start, &before) {
                self.user_ops
                    .insert(ByStart(UserOp::DeleteRange { start, end: before }));
            }
            let after = Bound::Excluded(key.clone());
            if nonempty(&after, &end) {
                self.user_ops
                    .insert(ByStart(UserOp::DeleteRange { start: after, end }));
            }
        }
        self.user_ops.replace(ByStart(UserOp::set(key, value)));
    }

    /// Delete a key, replacing a set and joining any delete intervals touching it.
    pub fn delete(&mut self, key: K) {
        let probe = ByStart::at(key.clone());
        // Only the adjacent ranges can contain the key or meet it at an endpoint.
        // Include the entry before a point mutation that may occupy the gap.
        let touches_range = self
            .user_ops
            .range(..=&probe)
            .rev()
            .take(2)
            .chain(
                self.user_ops
                    .range((Bound::Excluded(&probe), Bound::Unbounded))
                    .take(1),
            )
            .any(|op| {
                matches!(&op.0, UserOp::DeleteRange { start, end }
                if (closed(start), closed(end)).contains(&key))
            });
        if !touches_range {
            self.user_ops.replace(probe);
            return;
        }
        self.insert_range(Bound::Included(key.clone()), Bound::Included(key));
    }

    /// Delete every member of `range`, such as `start..=end`, `start..`, `..end`, or `..`,
    /// and merge overlapping deletes. Reversed bounds return `InvalidBounds` without
    /// changing the batch; any other range that contains no key changes nothing.
    /// A later `set(m, value)` splits a covering range into `[start, m)` and `(m, end]`.
    /// A later delete of `m` joins those intervals again. No database access occurs;
    /// verification must authenticate complete membership within the stored bounds.
    pub fn delete_range(&mut self, range: impl RangeBounds<K>) -> Result<(), VerifyError> {
        let (start, end) = (range.start_bound().cloned(), range.end_bound().cloned());
        if let (Some((start, _)), Some((end, _))) = (finite(&start), finite(&end))
            && start > end
        {
            return Err(VerifyError::InvalidBounds);
        }
        self.insert_range(start, end);
        Ok(())
    }

    /// Apply an interval, remove covered sets, and merge delete coverage. An interval that
    /// contains no key changes nothing. Treat point deletes as singleton intervals during
    /// the merge. At equal endpoints, intervals join if either includes the shared key; two
    /// excluded endpoints leave a gap. This establishes last-write-wins semantics before
    /// verification runs.
    fn insert_range(&mut self, start: Bound<K>, end: Bound<K>) {
        if !nonempty(&start, &end) {
            return;
        }
        let mut ranges = vec![(start.clone(), end.clone())];
        self.user_ops.retain(|op| match &op.0 {
            UserOp::Set { key, .. } => !range_contains(&start, &end, key),
            UserOp::Delete { key } => {
                ranges.push((Bound::Included(key.clone()), Bound::Included(key.clone())));
                false
            }
            UserOp::DeleteRange { start, end } => {
                ranges.push((start.clone(), end.clone()));
                false
            }
        });
        ranges.sort_unstable_by(|a, b| cmp_start(&a.0, &b.0));
        let mut merged: Vec<(Bound<K>, Bound<K>)> = Vec::new();
        for (start, end) in ranges {
            if let Some(previous) = merged.last_mut()
                && joins(&previous.1, &start)
            {
                if cmp_end(&end, &previous.1).is_gt() {
                    previous.1 = end;
                }
            } else {
                merged.push((start, end));
            }
        }
        for (start, end) in merged {
            let op = match (start, end) {
                (Bound::Included(start), Bound::Included(end)) if start == end => {
                    UserOp::delete(start)
                }
                (start, end) => UserOp::DeleteRange { start, end },
            };
            self.user_ops.insert(ByStart(op));
        }
    }

    /// Set the metadata encoded in the derived terminal commit.
    pub fn set_metadata(&mut self, metadata: V) {
        self.metadata = Some(metadata);
    }
    /// Clear the metadata encoded in the derived terminal commit.
    pub fn clear_metadata(&mut self) {
        self.metadata = None;
    }
    /// Return the optional commit metadata.
    pub const fn metadata(&self) -> Option<&V> {
        self.metadata.as_ref()
    }
    /// Return the number of normalized operations; a range counts as one operation.
    pub fn len(&self) -> usize {
        self.user_ops.len()
    }
    /// Return whether there are no mutations. Metadata does not affect this result.
    pub fn is_empty(&self) -> bool {
        self.user_ops.is_empty()
    }
    /// Iterate over all normalized operations in starting-key order.
    pub fn iter(&self) -> UserBatchIter<'_, K, V> {
        UserBatchIter(self.user_ops.iter())
    }
    /// Return the first operation starting at `key`, including an excluded range bound.
    /// This does not search for a range covering the key. A range with no lower bound
    /// starts at no key; [`Self::first`] returns it.
    pub fn get(&self, key: &K) -> Option<&UserOp<K, V>> {
        self.user_ops
            .range(ByStart::at(key.clone())..)
            .next()
            .map(|op| &op.0)
            .filter(|op| finite(&op.start()).is_some_and(|(start, _)| *start == key))
    }
    /// Return whether an operation starts at `key`; does not query range coverage.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
    /// Return the first operation in starting-key order.
    pub fn first(&self) -> Option<&UserOp<K, V>> {
        self.user_ops.first().map(|op| &op.0)
    }
    /// Return the last operation in starting-key order.
    pub fn last(&self) -> Option<&UserOp<K, V>> {
        self.user_ops.last().map(|op| &op.0)
    }
    /// Remove the first operation, leaving all remaining operations unchanged.
    pub fn pop_first(&mut self) -> Option<UserOp<K, V>> {
        self.user_ops.pop_first().map(|op| op.0)
    }
    /// Remove the last operation, leaving all remaining operations unchanged.
    pub fn pop_last(&mut self) -> Option<UserOp<K, V>> {
        self.user_ops.pop_last().map(|op| op.0)
    }
    /// Remove the first operation starting at `key`, without restoring earlier requests.
    pub fn remove(&mut self, key: &K) -> Option<UserOp<K, V>> {
        let probe = if self.get(key)?.excludes_start() {
            ByStart::after(key.clone())
        } else {
            ByStart::at(key.clone())
        };
        self.user_ops.take(&probe).map(|op| op.0)
    }
    /// Clear all mutations without changing metadata.
    pub fn clear(&mut self) {
        self.user_ops.clear();
    }
    /// Move the other batch's operations into this batch, with the other batch winning.
    /// Metadata in both batches remains unchanged.
    pub fn append(&mut self, other: &mut Self) {
        self.extend(
            core::mem::take(&mut other.user_ops)
                .into_iter()
                .map(|op| op.0),
        );
    }
    /// Borrow the normalized operations for verification and proof construction.
    pub(super) const fn ops(&self) -> &BTreeSet<ByStart<K, V>> {
        &self.user_ops
    }
}

impl<K: Ord + Clone, V> Default for UserBatch<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
impl<K: Ord + Clone, V, const M: usize> From<[UserOp<K, V>; M]> for UserBatch<K, V> {
    fn from(operations: [UserOp<K, V>; M]) -> Self {
        operations.into_iter().collect()
    }
}
impl<K: Ord + Clone, V: Clone> From<&[UserOp<K, V>]> for UserBatch<K, V> {
    fn from(operations: &[UserOp<K, V>]) -> Self {
        operations.iter().cloned().collect()
    }
}
impl<K: Ord + Clone, V: Clone, const M: usize> From<&[UserOp<K, V>; M]> for UserBatch<K, V> {
    fn from(operations: &[UserOp<K, V>; M]) -> Self {
        operations.iter().cloned().collect()
    }
}
impl<K: Ord + Clone, V> FromIterator<UserOp<K, V>> for UserBatch<K, V> {
    fn from_iter<T: IntoIterator<Item = UserOp<K, V>>>(operations: T) -> Self {
        let mut batch = Self::new();
        batch.extend(operations);
        batch
    }
}
impl<K: Ord + Clone, V> Extend<UserOp<K, V>> for UserBatch<K, V> {
    /// Apply operations in arrival order, preserving normalization after each operation.
    /// A range that contains no key changes nothing.
    fn extend<T: IntoIterator<Item = UserOp<K, V>>>(&mut self, operations: T) {
        for operation in operations {
            match operation {
                UserOp::Set { key, value } => self.set(key, value),
                UserOp::Delete { key } => self.delete(key),
                UserOp::DeleteRange { start, end } => self.insert_range(start, end),
            }
        }
    }
}
impl<K, V> IntoIterator for UserBatch<K, V> {
    type Item = UserOp<K, V>;
    type IntoIter = UserBatchIntoIter<K, V>;
    /// Consume the batch into all normalized operations; metadata is discarded.
    fn into_iter(self) -> Self::IntoIter {
        UserBatchIntoIter(self.user_ops.into_iter())
    }
}
impl<'a, K, V> IntoIterator for &'a UserBatch<K, V> {
    type Item = &'a UserOp<K, V>;
    type IntoIter = UserBatchIter<'a, K, V>;
    /// Borrow all normalized operations without yielding metadata.
    fn into_iter(self) -> Self::IntoIter {
        UserBatchIter(self.user_ops.iter())
    }
}

/// Borrowing iterator over a [`UserBatch`]'s normalized operations in starting-key order.
#[derive(Clone, Debug)]
pub struct UserBatchIter<'a, K, V>(btree_set::Iter<'a, ByStart<K, V>>);

impl<'a, K, V> Iterator for UserBatchIter<'a, K, V> {
    type Item = &'a UserOp<K, V>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|op| &op.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K, V> DoubleEndedIterator for UserBatchIter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|op| &op.0)
    }
}
impl<K, V> ExactSizeIterator for UserBatchIter<'_, K, V> {}
impl<K, V> FusedIterator for UserBatchIter<'_, K, V> {}

/// Owning iterator over a [`UserBatch`]'s normalized operations in starting-key order.
#[derive(Debug)]
pub struct UserBatchIntoIter<K, V>(btree_set::IntoIter<ByStart<K, V>>);

impl<K, V> Iterator for UserBatchIntoIter<K, V> {
    type Item = UserOp<K, V>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|op| op.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K, V> DoubleEndedIterator for UserBatchIntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|op| op.0)
    }
}
impl<K, V> ExactSizeIterator for UserBatchIntoIter<K, V> {}
impl<K, V> FusedIterator for UserBatchIntoIter<K, V> {}

impl<K: Key, A: Clone> UserBatch<K, A> {
    /// Verify that this batch produces `claimed_post` from the trusted `pre_commitment`.
    ///
    /// Use `TrustedMap::empty_cache(pre_commitment)` when starting without cached facts.
    /// Otherwise supply facts previously authenticated at exactly that commitment.
    /// The proof bundle supplies old records and Merkle proofs. After authenticating
    /// those records, the verifier enumerates range deletes through authenticated
    /// immediate successors, rejecting missing members or links. The batch already
    /// stores normalized ranges. It combines the resulting point deletes with the cache and
    /// this batch's mutations to construct the new operations, including predecessor rewrites
    /// and floor copies. It then computes the activity bitmap, floor, and root.
    /// Success returns the [`Verified`] transition: [`Verified::into_cache`] carries the facts
    /// to `claimed_post` for the next transition, and a caller that keeps no cache can drop
    /// it without paying for that. Failure returns no working cache. Clone the input cache
    /// first if it must be retained.
    ///
    /// `policy` must be the floor [`Policy`] the writer used for this batch, in the same state:
    /// the commitment does not record how far the floor moved. Verification asks it for limits
    /// and decisions exactly as the writer's walk did, so a policy that keeps state advances as
    /// the writer's copy did. A failed verification may already have advanced it. Policies that
    /// may evict, or that limit skips while deciding entries, fail with
    /// [`VerifyError::UnsupportedFloorPolicy`].
    ///
    /// # Complexity
    ///
    /// Uses the parameters and fixed-size model in the [module complexity discussion](super#complexity).
    /// Worst-case CPU is
    /// `O((p + k + f) * log(c + p + k + f + 2) + t * log(t + 2))`;
    /// peak memory is `O(c + k + p + f + u + t)`, excluding the borrowed proof/batch.
    /// For honest proofs with O(k) claims, candidates, and bitmap units, these become
    /// `O(k * log(c + k + 2) + k * h * log(k * h + 2))` and `O(c + k * h)`.
    /// These typical-workload bounds are not probabilistic averages. A long inactive
    /// floor scan or surplus proof claims can dominate a small batch. No database
    /// reads occur, and the owned cache is never cloned.
    ///
    /// # Examples
    ///
    /// Start without cached facts, then keep the facts for the next batch:
    ///
    /// ```no_run
    /// use commonware_codec::Codec;
    /// use commonware_cryptography::Sha256;
    /// use commonware_storage::{merkle::mmr, qmdb::{
    ///     any::{operation::Ordered, value::{FixedEncoding, FixedValue}},
    ///     current::verifier::{BatchProofBundle, CurrentCommitment, TrustedMap, UserBatch, VerifyError},
    ///     floor::Proportional,
    ///     operation::Key,
    /// }};
    ///
    /// fn verify<K: Key, V: FixedValue>(
    ///     batch: UserBatch<K, V>,
    ///     pre: CurrentCommitment<mmr::Family, K, FixedEncoding<V>, Sha256, 32>,
    ///     proof: &BatchProofBundle<mmr::Family, K, FixedEncoding<V>, Sha256, 32>,
    ///     post: CurrentCommitment<mmr::Family, K, FixedEncoding<V>, Sha256, 32>,
    /// ) -> Result<(), VerifyError>
    /// where Ordered<mmr::Family, K, FixedEncoding<V>>: Codec,
    /// {
    ///     let verified = batch.verify(pre, TrustedMap::empty_cache(pre), proof, post, &mut Proportional)?;
    ///     assert_eq!(verified.commitment(), &post);
    ///     let cache = verified.into_cache()?;
    ///     assert_eq!(cache.commitment(), &post);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Trust requirements
    ///
    /// The pre-commitment must represent a valid ordered Current database, this batch and
    /// metadata must be authorized by the caller, and any supplied cache facts must be
    /// true at that checkpoint. Neither identity nor authorization is proved here. Proof
    /// material and the claimed post-commitment may be adversarial. See the module's
    /// verification argument for the checks performed by each stage. The chunk profile
    /// must use a power-of-two `N`, divisible by the digest size, with `8 * N` fitting in
    /// `usize`; other profiles do not compile.
    ///
    /// # Errors
    ///
    /// Rejects malformed or conflicting proofs, missing facts needed for deterministic
    /// execution, inconsistent bundle bounds, and a derived size or root different from
    /// `claimed_post`. Missing cached knowledge is never interpreted as key absence.
    pub fn verify<F, V, H, P, const N: usize>(
        &self,
        pre_commitment: CurrentCommitment<F, K, V, H, N>,
        cache: TrustedMap<F, K, V, H, N>,
        proof_bundle: &BatchProofBundle<F, K, V, H, N>,
        claimed_post: CurrentCommitment<F, K, V, H, N>,
        policy: &mut P,
    ) -> Result<Verified<F, K, V, H, N>, VerifyError>
    where
        F: Graftable,
        V: ValueEncoding<Value = A>,
        H: Hasher,
        P: Policy<F, K, A>,
        Operation<F, update::Ordered<K, V>>: Codec,
    {
        self.verify_with_strategy(
            pre_commitment,
            cache,
            proof_bundle,
            claimed_post,
            &commonware_parallel::Sequential,
            policy,
        )
    }

    /// Verify with a caller-selected CPU strategy, including Rayon without an async executor.
    /// Independent claims, raw/grafted bitmap proofs, and leaf hashes can run in parallel.
    /// Cache merges and state derivation retain their canonical order. Every proof and
    /// floor check is identical to `verify`; errors return no partially updated cache.
    /// Total work has the same bounds as `verify`, with O(p + k) collected results and
    /// per-worker encoding buffers. Parallel elapsed time depends on pool size and work
    /// distribution; ordered execution and [`Verified::into_cache`] remain serial.
    pub fn verify_with_strategy<F, V, H, P, const N: usize, S: commonware_parallel::Strategy>(
        &self,
        pre_commitment: CurrentCommitment<F, K, V, H, N>,
        cache: TrustedMap<F, K, V, H, N>,
        proof_bundle: &BatchProofBundle<F, K, V, H, N>,
        claimed_post: CurrentCommitment<F, K, V, H, N>,
        strategy: &S,
        policy: &mut P,
    ) -> Result<Verified<F, K, V, H, N>, VerifyError>
    where
        F: Graftable,
        V: ValueEncoding<Value = A>,
        H: Hasher,
        P: Policy<F, K, A>,
        Operation<F, update::Ordered<K, V>>: Codec,
    {
        verify_batch(
            pre_commitment,
            cache,
            self,
            proof_bundle,
            claimed_post,
            strategy,
            policy,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Batch = UserBatch<u8, u8>;

    /// What the last request covering a key does to it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Effect {
        Untouched,
        Set(u8),
        Deleted,
    }

    /// One edit; a set writes the step at which it is applied, so every set is distinct.
    #[derive(Clone, Copy, Debug)]
    enum Edit {
        Set(u8),
        Delete(u8),
        DeleteRange(Bound<u8>, Bound<u8>),
        Remove(u8),
    }

    const KEYS: u8 = 5;

    fn effect(op: &UserOp<u8, u8>, key: u8) -> Option<Effect> {
        match op {
            UserOp::Set { key: set, value } => (*set == key).then_some(Effect::Set(*value)),
            UserOp::Delete { key: deleted } => (*deleted == key).then_some(Effect::Deleted),
            UserOp::DeleteRange { start, end } => {
                range_contains(start, end, &key).then_some(Effect::Deleted)
            }
        }
    }

    fn effects(batch: &Batch, keys: u8) -> Vec<Effect> {
        (0..keys)
            .map(|key| {
                batch
                    .iter()
                    .find_map(|op| effect(op, key))
                    .unwrap_or(Effect::Untouched)
            })
            .collect()
    }

    /// Apply `edit` to the batch and to a model in which the last request covering a key wins.
    fn apply(batch: &mut Batch, model: &mut [Effect], edit: Edit, step: u8) {
        match edit {
            Edit::Set(key) => {
                batch.set(key, step);
                model[usize::from(key)] = Effect::Set(step);
            }
            Edit::Delete(key) => {
                batch.delete(key);
                model[usize::from(key)] = Effect::Deleted;
            }
            Edit::DeleteRange(start, end) => {
                let reversed = matches!((finite(&start), finite(&end)),
                    (Some((start, _)), Some((end, _))) if start > end);
                let before = batch.iter().cloned().collect::<Vec<_>>();
                if batch.delete_range((start, end)).is_err() {
                    assert!(reversed, "{edit:?} was refused");
                    assert!(batch.iter().eq(before.iter()), "{edit:?} changed the batch");
                    return;
                }
                assert!(!reversed, "{edit:?} was accepted");
                for (key, slot) in (0..).zip(model.iter_mut()) {
                    if (start, end).contains(&key) {
                        *slot = Effect::Deleted;
                    }
                }
            }
            Edit::Remove(key) => {
                // Removing an operation forgets its effect without restoring earlier requests.
                let removed = batch.get(&key).cloned();
                assert_eq!(batch.remove(&key), removed);
                if let Some(op) = removed {
                    for (covered, slot) in model.iter_mut().enumerate() {
                        if effect(&op, covered as u8).is_some() {
                            *slot = Effect::Untouched;
                        }
                    }
                }
            }
        }
    }

    /// Operations are ordered and disjoint, every finite range spans two keys or more, and
    /// delete coverage is joined wherever it can be: two deletes meet only at a key both
    /// exclude. Only the first operation may start unbounded, and only the last may end so.
    fn assert_normalized(batch: &Batch) {
        let bounds = |op: &UserOp<u8, u8>| match op {
            UserOp::Set { key, .. } | UserOp::Delete { key } => {
                (Bound::Included(*key), Bound::Included(*key))
            }
            UserOp::DeleteRange { start, end } => (*start, *end),
        };
        for op in batch {
            if let UserOp::DeleteRange { start, end } = op
                && let (Some((start, _)), Some((end, _))) = (finite(start), finite(end))
            {
                assert!(start < end, "{op:?} is not a range");
            }
        }
        for (before, after) in batch.iter().zip(batch.iter().skip(1)) {
            let (end, start) = (bounds(before).1, bounds(after).0);
            let (Some((end, end_included)), Some((start, start_included))) =
                (finite(&end), finite(&start))
            else {
                panic!("{before:?} and {after:?} overlap");
            };
            let deletes = ![before, after]
                .iter()
                .any(|op| matches!(op, UserOp::Set { .. }));
            let apart = match end.cmp(start) {
                Ordering::Less => true,
                Ordering::Equal if deletes => !end_included && !start_included,
                Ordering::Equal => !end_included || !start_included,
                Ordering::Greater => false,
            };
            assert!(apart, "{before:?} and {after:?} overlap or were not joined");
        }
    }

    /// The batch matches the model, is normalized, and rebuilds from its own operations.
    fn check(batch: &Batch, model: &[Effect], edits: &[Edit]) {
        assert_eq!(effects(batch, KEYS), model, "after {edits:?}");
        assert_normalized(batch);
        let rebuilt = batch.iter().cloned().collect::<Batch>();
        assert!(rebuilt.iter().eq(batch.iter()), "after {edits:?}");
        assert_eq!(batch.len(), batch.iter().count());
    }

    /// Every point edit, every inclusive range, and every range open at one end or both.
    /// Sets split ranges into the excluded bounds, so those are reached too.
    fn all_edits() -> Vec<Edit> {
        let mut edits = vec![Edit::DeleteRange(Bound::Unbounded, Bound::Unbounded)];
        for key in 0..KEYS {
            edits.extend([
                Edit::Set(key),
                Edit::Delete(key),
                Edit::Remove(key),
                Edit::DeleteRange(Bound::Unbounded, Bound::Included(key)),
                Edit::DeleteRange(Bound::Included(key), Bound::Unbounded),
            ]);
            edits.extend(
                (key..KEYS)
                    .map(|end| Edit::DeleteRange(Bound::Included(key), Bound::Included(end))),
            );
        }
        edits
    }

    #[test]
    fn every_three_edits_match_a_last_write_wins_model() {
        let edits = all_edits();
        for &first in &edits {
            for &second in &edits {
                for &third in &edits {
                    let sequence = [first, second, third];
                    let mut batch = Batch::new();
                    let mut model = [Effect::Untouched; KEYS as usize];
                    for (step, edit) in sequence.into_iter().enumerate() {
                        apply(&mut batch, &mut model, edit, step as u8);
                        check(&batch, &model, &sequence[..=step]);
                    }
                }
            }
        }
    }

    #[test]
    fn an_appended_batch_wins() {
        let edits = all_edits();
        for &first in &edits {
            for &second in &edits {
                for &other in &edits {
                    let mut batch = Batch::new();
                    let mut model = [Effect::Untouched; KEYS as usize];
                    apply(&mut batch, &mut model, first, 0);
                    apply(&mut batch, &mut model, second, 1);
                    let mut appended = Batch::new();
                    apply(
                        &mut appended,
                        &mut [Effect::Untouched; KEYS as usize],
                        other,
                        2,
                    );
                    for (slot, theirs) in model.iter_mut().zip(effects(&appended, KEYS)) {
                        if theirs != Effect::Untouched {
                            *slot = theirs;
                        }
                    }
                    batch.append(&mut appended);
                    assert!(appended.is_empty());
                    check(&batch, &model, &[first, second, other]);
                }
            }
        }
    }

    #[test]
    fn long_edit_sequences_match_the_model() {
        const WIDE: u8 = 64;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |bound: u8| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(bound)) as u8
        };
        // Any pair of bounds up to a dozen keys apart, so reversed and empty ranges arrive too.
        // One side in eight is open.
        let bound = |choice: u8, key: u8| match choice {
            0 => Bound::Unbounded,
            1..=3 => Bound::Excluded(key),
            _ => Bound::Included(key),
        };
        let mut batch = Batch::new();
        let mut model = [Effect::Untouched; WIDE as usize];
        for step in 0..2_000u32 {
            let key = next(WIDE);
            let edit = match next(8) {
                0..=2 => Edit::Set(key),
                3 | 4 => Edit::Delete(key),
                5 | 6 => {
                    let end = key.saturating_add(next(14)).saturating_sub(1).min(WIDE - 1);
                    Edit::DeleteRange(bound(next(8), key), bound(next(8), end))
                }
                _ => Edit::Remove(key),
            };
            apply(&mut batch, &mut model, edit, step as u8);
            assert_eq!(effects(&batch, WIDE), model, "at step {step}");
            assert_normalized(&batch);
        }
    }

    #[test]
    fn reversed_ranges_leave_the_batch_unchanged() {
        let mut batch = Batch::from([UserOp::set(3, 1)]);
        assert_eq!(
            batch.delete_range((Bound::Included(5), Bound::Included(1))),
            Err(VerifyError::InvalidBounds)
        );
        assert!(batch.iter().eq([UserOp::set(3, 1)].iter()));
    }

    /// A range that contains no key, empty or reversed, changes nothing, wherever it arrives.
    #[test]
    fn ranges_without_keys_change_nothing() {
        let batch = Batch::from([
            UserOp::set(3, 1),
            UserOp::DeleteRange {
                start: Bound::Excluded(3),
                end: Bound::Excluded(3),
            },
            UserOp::delete_range(4..4),
            UserOp::delete_range((Bound::Included(9), Bound::Included(5))),
        ]);
        assert!(batch.iter().eq([UserOp::set(3, 1)].iter()));
    }

    /// A set splits a range open at both ends into two open at one end each, and deleting the
    /// key joins them again. Looking a key up finds only an operation starting there, not a
    /// range covering it; a range open below is the first operation.
    #[test]
    fn open_ranges_split_and_join() {
        let mut batch = Batch::from([UserOp::delete_range(..), UserOp::set(3, 1)]);
        assert!(
            batch.iter().eq([
                UserOp::DeleteRange {
                    start: Bound::Unbounded,
                    end: Bound::Excluded(3),
                },
                UserOp::set(3, 1),
                UserOp::DeleteRange {
                    start: Bound::Excluded(3),
                    end: Bound::Unbounded,
                },
            ]
            .iter())
        );
        assert_eq!(batch.get(&0), None);
        batch.delete(3);
        assert!(batch.iter().eq([UserOp::delete_range(..)].iter()));
        assert_eq!(batch.first().map(UserOp::start), Some(Bound::Unbounded));
    }
}
