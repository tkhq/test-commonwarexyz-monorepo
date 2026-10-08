//! Cache facts whose truth the caller has already established at one commitment.
//!
//! An entry asserts membership even when its optional fields are unknown. A cyclic
//! edge asserts that both endpoints are active and no active key lies strictly between them.
//! `(k, k)` therefore describes a database with exactly one key. Missing entries do
//! not establish absence. The indexes reject contradictions between retained facts,
//! but do not authenticate those facts against a root.

use super::{CurrentCommitment, OptionTrusted::KnownTrusted, VerifyError};
use crate::{
    merkle::{Family, Graftable, Location},
    qmdb::{any::ValueEncoding, operation::Key},
};
use alloc::collections::BTreeMap;
use commonware_cryptography::Hasher;
use core::ops::Bound::{Excluded, Unbounded};

/// A value which may be retained as an exact trusted fact or omitted from the cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OptionTrusted<A> {
    /// The contained value is trusted at the map's commitment.
    KnownTrusted(A),
    /// The cache makes no claim about this field.
    Unknown,
}

/// Assert membership of `key`, optionally retaining its location, value, and successor.
/// Unknown fields mean omitted knowledge, not absent data. Callers constructing
/// entries must already trust every assertion. Equality and ordering compare only
/// keys. They do not compare record contents or establish consistency.
#[derive(Clone, Debug)]
pub struct TrustedEntry<F: Family, K: Key, V> {
    /// Active key.
    pub key: K,
    /// Active operation location, if retained.
    pub location: OptionTrusted<Location<F>>,
    /// Active value, if retained.
    pub value: OptionTrusted<V>,
    /// Immediate cyclic successor, if retained.
    pub next_key: OptionTrusted<K>,
}

impl<F: Family, K: Key, V> PartialEq for TrustedEntry<F, K, V> {
    /// Compare entry keys only; this does not compare retained facts.
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<F: Family, K: Key, V> Eq for TrustedEntry<F, K, V> {}

impl<F: Family, K: Key, V> PartialOrd for TrustedEntry<F, K, V> {
    /// Use key-only ordering, consistent with entry equality.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Family, K: Key, V> Ord for TrustedEntry<F, K, V> {
    /// Order entries by their active keys, ignoring optional retained fields.
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

/// Test the half-open cyclic interval `[start, end)` in key order.
/// This supplies the wraparound rule used to interpret ordered successor links.
/// Equal endpoints denote the whole circle. Wraparound includes either side of the
/// key-space boundary. The strict variant also excludes `start`.
fn span_contains<K: Ord>(start: &K, end: &K, key: &K) -> bool {
    if start >= end {
        key >= start || key < end
    } else {
        key >= start && key < end
    }
}

/// Test whether `key` lies strictly inside the cyclic interval `(start, end)`.
/// With an authenticated successor edge, containment establishes key absence. During
/// operation derivation it also selects a new key that precedes the old successor.
/// The predicate itself checks only key order. It excludes both endpoints. Equal
/// endpoints include every other key, as required for a database with one active key.
pub(super) fn strict_span_contains<K: Ord>(start: &K, end: &K, key: &K) -> bool {
    key != start && span_contains(start, end, key)
}

/// Check whether any retained map key lies strictly inside a cyclic interval.
/// Edge validation uses this to reject a proposed absence interval containing a
/// known member or another edge's predecessor.
/// Use one ordered lookup for a non-wrapping interval and two for a wrapping interval.
/// Equal endpoints check all keys except the endpoint, detecting singleton conflicts.
fn span_has_key<K: Ord, V>(map: &BTreeMap<K, V>, start: &K, end: &K) -> bool {
    if start < end {
        map.range((Excluded(start), Excluded(end))).next().is_some()
    } else {
        map.range((Excluded(start), Unbounded)).next().is_some()
            || map.range((Unbounded, Excluded(end))).next().is_some()
    }
}

/// Find a known edge whose open cyclic interval contains `key`.
/// A retained edge can establish absence even when the cache has no record for the
/// queried key. Classification uses its predecessor as the insertion anchor, and
/// cache rebasing uses it to find intervals that a new key invalidates.
///
/// Edges must already have disjoint interiors and consistent cyclic order. The closest
/// preceding edge and the final wrapping edge are then the only possible matches.
pub(super) fn containing_edge<'a, K: Ord, V>(
    edges: &'a BTreeMap<K, V>,
    key: &K,
    next_key: impl Fn(&V) -> &K,
) -> Option<(&'a K, &'a V)> {
    let first = edges.range(..=key).next_back();
    if let Some((start, end)) = first
        && strict_span_contains(start, next_key(end), key)
    {
        return Some((start, end));
    }
    let last = edges.last_key_value()?;
    if first.is_none_or(|candidate| candidate.0 != last.0)
        && strict_span_contains(last.0, next_key(last.1), key)
    {
        return Some(last);
    }
    None
}

/// Retained cyclic adjacency edges `(predecessor, successor)` proving open intervals empty.
/// Both endpoints are active keys. The interval excludes those endpoints.
/// An edge that wraps covers keys above the predecessor or below the successor.
/// Equal endpoints cover every other key. Its private storage is updated through
/// the trusted-state APIs, which check consistency but assume caller-authenticated facts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrustedAbsenceMap<K: Key> {
    spans: BTreeMap<K, K>,
}

impl<K: Key> TrustedAbsenceMap<K> {
    /// Create an empty absence cache.
    pub const fn new() -> Self {
        Self {
            spans: BTreeMap::new(),
        }
    }

    /// Return whether no absence span is cached.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Return the cached edge whose strict cyclic span contains `key`.
    pub fn get(&self, key: &K) -> Option<(&K, &K)> {
        containing_edge(&self.spans, key, |next| next)
    }

    /// Iterate over cached predecessor-successor edges.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &K)> {
        self.spans.iter()
    }

    /// Return the successor for an explicitly retained absence edge starting here.
    /// Unlike `get`, this is an endpoint lookup, not a test for absence of the argument.
    pub(crate) fn successor(&self, predecessor: &K) -> Option<&K> {
        self.spans.get(predecessor)
    }
}

/// Trusted knowledge about active ordered state.
#[derive(Clone, Debug)]
pub enum TrustedState<F: Family, K: Key, V> {
    /// A partial cache. Missing records are unknown, not absent.
    Partial(PartialTrustedState<F, K, V>),
    /// The database is authenticated to contain no active user key.
    EmptyDatabase,
}

/// A consistent subset of membership and cyclic adjacency facts.
/// Location indexes prohibit two keys at one active location. Edge indexes enforce
/// unique predecessors and successors and forbid members inside an absence interval.
/// Edges may be retained through an entry's known successor or an explicit span.
/// Removing one source preserves knowledge retained by the other. Consistency is
/// checked locally. The caller remains responsible for the truth of imported facts.
#[derive(Clone, Debug)]
pub struct PartialTrustedState<F: Family, K: Key, V> {
    entries: BTreeMap<K, TrustedEntry<F, K, V>>,
    absences: TrustedAbsenceMap<K>,
    known_locations: BTreeMap<Location<F>, K>,
    known_edges: BTreeMap<K, K>,
    predecessors_by_successor: BTreeMap<K, K>,
}

impl<F: Family, K: Key, V> Default for PartialTrustedState<F, K, V> {
    /// Create a partial cache that asserts no membership or absence facts.
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            absences: TrustedAbsenceMap::new(),
            known_locations: BTreeMap::new(),
            known_edges: BTreeMap::new(),
            predecessors_by_successor: BTreeMap::new(),
        }
    }
}

impl<F: Family, K: Key, V> PartialTrustedState<F, K, V> {
    /// Build indexes for facts the caller has already authenticated.
    /// Reject duplicate entries, shared active locations, incompatible successors, and
    /// absence intervals containing known members or conflicting edges. This does not
    /// check a root or location bounds. `TrustedMap::from_state` checks the upper bound,
    /// and transition verification checks the authenticated floor. Unknown fields remain
    /// unknown. The caller must establish the truth of every supplied fact before this call.
    pub fn new(
        entries: impl IntoIterator<Item = TrustedEntry<F, K, V>>,
        spans: impl IntoIterator<Item = (K, K)>,
    ) -> Result<Self, VerifyError> {
        let mut state = Self::default();
        for entry in entries {
            state.insert_entry(entry)?;
        }
        for (predecessor, successor) in spans {
            state.insert_span(predecessor, successor)?;
        }
        Ok(state)
    }

    /// Iterate over retained active entries.
    pub fn entries(&self) -> impl Iterator<Item = &TrustedEntry<F, K, V>> {
        self.entries.values()
    }

    /// Return the retained entry for `key`.
    pub fn entry(&self, key: &K) -> Option<&TrustedEntry<F, K, V>> {
        self.entries.get(key)
    }

    /// Return the absence-span cache.
    pub const fn absences(&self) -> &TrustedAbsenceMap<K> {
        &self.absences
    }

    /// Return the authenticated immediate predecessor of this endpoint, if retained.
    /// Operation derivation uses it to find the record that needs rewriting after a
    /// deletion. The lookup follows a known successor link in reverse. A neighboring
    /// cached key alone cannot establish that adjacency because the cache may omit keys.
    pub(crate) fn predecessor_of(&self, key: &K) -> Option<&K> {
        self.predecessors_by_successor.get(key)
    }

    /// Return the immediate successor known from an entry or an explicit absence span.
    /// Operation derivation follows these links to bypass deleted keys and select final
    /// successors. The two sources share one index that rejects conflicting successors.
    pub(crate) fn known_edge(&self, key: &K) -> Option<&K> {
        self.known_edges.get(key)
    }

    /// Return the least active key, if the edge wrapping from the greatest key to it is
    /// retained. Range expansion starts a delete with no lower bound here.
    ///
    /// Only the greatest key's successor is no greater than its predecessor, and no retained
    /// edge starts above the greatest key, so the last retained edge wraps exactly when the
    /// wrapping edge is retained. A database with one key wraps from that key to itself.
    pub(crate) fn first_key(&self) -> Option<&K> {
        self.known_edges
            .last_key_value()
            .filter(|(last, first)| first <= last)
            .map(|(_, first)| first)
    }

    /// Find adjacency establishing that `key` is absent, including entry-retained edges.
    /// Return `None` when no retained interval covers it; this does not establish membership.
    pub(crate) fn containing_edge(&self, key: &K) -> Option<(&K, &K)> {
        containing_edge(&self.known_edges, key, |next| next)
    }

    /// Report membership known from an entry or either endpoint of a retained edge.
    /// This distinguishes a mutation of an existing key from a creation. A location or
    /// value may still be unknown, so execution must require those fields when needed.
    pub(crate) fn is_claimed_member(&self, key: &K) -> bool {
        self.entries.contains_key(key)
            || self.known_edges.contains_key(key)
            || self.predecessors_by_successor.contains_key(key)
    }

    /// Check that a proposed adjacency can coexist with all retained membership facts.
    /// Both records and explicit absence spans feed the same edge index. Checking
    /// their consistency prevents the cache from classifying a key as both present
    /// and absent, or supplying different successors for the same predecessor.
    ///
    /// Reject a different successor, a second predecessor for the same successor, any
    /// known member inside the interval, or an endpoint inside another interval. These
    /// checks also exclude overlapping wrapping edges and conflicting singleton circles.
    /// The check establishes structural compatibility, not authentication.
    fn validate_edge(&self, predecessor: &K, successor: &K) -> Result<(), VerifyError> {
        if let Some(existing) = self.known_edges.get(predecessor) {
            return if existing == successor {
                Ok(())
            } else {
                Err(VerifyError::InvalidTrustedState)
            };
        }
        if self
            .predecessors_by_successor
            .get(successor)
            .is_some_and(|existing| existing != predecessor)
        {
            return Err(VerifyError::InvalidTrustedState);
        }
        // An open cyclic interval also covers the singleton-circle case.
        // Disjointness rules out multiple wrapping edges without a separate index.
        if span_has_key(&self.entries, predecessor, successor)
            || span_has_key(&self.known_edges, predecessor, successor)
            || self.containing_edge(predecessor).is_some()
            || self
                .containing_edge(successor)
                .is_some_and(|(source, _)| source != predecessor)
        {
            return Err(VerifyError::InvalidTrustedState);
        }
        Ok(())
    }

    /// Index an already validated edge in both directions, leaving identical entries alone.
    /// Call only after conflict checks; this helper does not validate supplied facts.
    fn add_edge_index(&mut self, predecessor: K, successor: K) {
        if self.known_edges.get(&predecessor) == Some(&successor) {
            return;
        }
        self.predecessors_by_successor
            .insert(successor.clone(), predecessor.clone());
        self.known_edges.insert(predecessor, successor);
    }

    /// Forget an adjacency after its callers have checked that no retained fact claims it.
    /// Keep forward and reverse indexes synchronized when dropping the last source.
    fn remove_edge_index(&mut self, predecessor: &K, successor: &K) {
        self.known_edges.remove(predecessor);
        self.predecessors_by_successor.remove(successor);
    }

    /// Insert a new trusted member after checking key, location, and adjacency conflicts.
    /// Perform every fallible check before changing indexes. The caller authenticates the
    /// record and checks commitment-specific location bounds.
    pub(crate) fn insert_entry(&mut self, entry: TrustedEntry<F, K, V>) -> Result<(), VerifyError> {
        if self.entries.contains_key(&entry.key) {
            return Err(VerifyError::InvalidTrustedState);
        }
        if let KnownTrusted(location) = entry.location
            && self.known_locations.contains_key(&location)
        {
            return Err(VerifyError::InvalidTrustedState);
        }
        if let KnownTrusted(successor) = &entry.next_key {
            self.validate_edge(&entry.key, successor)?;
        } else if containing_edge(&self.known_edges, &entry.key, |next| next).is_some() {
            // Without an entry successor, an interior edge would be the only
            // retained fact that could contradict this member.
            return Err(VerifyError::InvalidTrustedState);
        }

        if let KnownTrusted(location) = entry.location {
            self.known_locations.insert(location, entry.key.clone());
        }
        if let KnownTrusted(successor) = &entry.next_key {
            self.add_edge_index(entry.key.clone(), successor.clone());
        }
        self.entries.insert(entry.key.clone(), entry);
        Ok(())
    }

    /// Drop one retained record and its location index, returning the forgotten fields.
    /// Preserve its adjacency if an explicit span still retains it. Removal forgets
    /// knowledge; it does not assert that the database key is absent.
    pub(crate) fn remove_entry(&mut self, key: &K) -> Option<TrustedEntry<F, K, V>> {
        let entry = self.entries.remove(key)?;
        if let KnownTrusted(location) = entry.location {
            self.known_locations.remove(&location);
        }
        if let KnownTrusted(successor) = &entry.next_key {
            // The entry was removed above, so only an explicit span can retain this edge.
            if self.absences.spans.get(key) != Some(successor) {
                self.remove_edge_index(key, successor);
            }
        }
        Some(entry)
    }

    /// Retain a caller-authenticated adjacency after checking interval consistency.
    /// An identical span is accepted. Insert its endpoints into the shared edge indexes,
    /// which can establish membership even when neither endpoint has a retained record.
    pub(crate) fn insert_span(&mut self, predecessor: K, successor: K) -> Result<(), VerifyError> {
        if let Some(existing) = self.absences.spans.get(&predecessor) {
            return if existing == &successor {
                Ok(())
            } else {
                Err(VerifyError::InvalidTrustedState)
            };
        }
        self.validate_edge(&predecessor, &successor)?;
        self.absences
            .spans
            .insert(predecessor.clone(), successor.clone());
        self.add_edge_index(predecessor, successor);
        Ok(())
    }

    /// Drop one explicit absence span and return its successor.
    /// Keep the shared adjacency if a retained record still supplies the same successor.
    pub(crate) fn remove_span(&mut self, predecessor: &K) -> Option<K> {
        let successor = self.absences.spans.remove(predecessor)?;
        let entry_claims = self
            .entry(predecessor)
            .is_some_and(|entry| matches!(&entry.next_key, KnownTrusted(key) if key == &successor));
        if !entry_claims {
            // The span was removed above, so only a retained entry can keep this edge.
            self.remove_edge_index(predecessor, &successor);
        }
        Some(successor)
    }

    /// Check that every retained active location lies in `[start, end)`.
    /// Commitment binding and transition verification use this to reject cached records
    /// outside the live user-operation range. Checking the first and last locations is
    /// sufficient because the index is ordered. An empty location index passes this
    /// bounds check without establishing that the database itself is empty.
    pub(crate) fn locations_within(&self, start: Location<F>, end: Location<F>) -> bool {
        self.known_locations
            .first_key_value()
            .is_none_or(|(loc, _)| *loc >= start)
            && self
                .known_locations
                .last_key_value()
                .is_none_or(|(loc, _)| *loc < end)
    }

    /// Enumerate retained keys with known locations below `end`.
    /// Cache rebasing uses this to require deletion or relocation below the new floor.
    pub(crate) fn keys_below_location(&self, end: Location<F>) -> impl Iterator<Item = &K> {
        self.known_locations.range(..end).map(|(_, key)| key)
    }
}

/// Caller-trusted facts associated with one exact Current checkpoint.
///
/// Prefer `empty_cache` followed by verification to build a cache of authenticated facts.
/// `from_state`, `replace_entry`, and `install_span` accept facts the caller already
/// trusts. They check local consistency without verifying Merkle proofs, so callers
/// must authenticate peer-supplied facts before passing them to these methods.
/// A successful batch verification preserves or derives every returned fact at the
/// new commitment, provided the input facts were true.
#[derive(Debug)]
pub struct TrustedMap<F, K, V, H, const N: usize>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    commitment: CurrentCommitment<F, K, V, H, N>,
    state: TrustedState<F, K, V::Value>,
}

impl<F, K, V, H, const N: usize> Clone for TrustedMap<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    /// Clone the checkpoint and its trusted facts without authenticating them again.
    fn clone(&self) -> Self {
        Self {
            commitment: self.commitment,
            state: self.state.clone(),
        }
    }
}

impl<F, K, V, H, const N: usize> TrustedMap<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
{
    /// Start with no knowledge of keys at this trusted commitment.
    /// The proof bundle must establish the pre-state facts needed by the batch,
    /// including whether mutated keys are present or absent. An empty cache makes
    /// no claim about whether the database contains keys.
    pub fn empty_cache(commitment: CurrentCommitment<F, K, V, H, N>) -> Self {
        Self {
            commitment,
            state: TrustedState::Partial(PartialTrustedState::default()),
        }
    }

    /// Bind already trusted state to a checkpoint after checking known location bounds.
    /// All facts must be true at exactly `commitment`. No Merkle authentication occurs.
    /// Known user locations must precede the terminal. The next verification also checks
    /// them against the floor authenticated by that terminal. Prefer `empty_cache` when
    /// only the checkpoint, rather than its constituent records, is trusted.
    pub fn from_state(
        commitment: CurrentCommitment<F, K, V, H, N>,
        state: TrustedState<F, K, V::Value>,
    ) -> Result<Self, VerifyError> {
        if let TrustedState::Partial(partial) = &state
            && !partial.locations_within(Location::new(0), commitment.size() - 1)
        {
            return Err(VerifyError::InvalidTrustedState);
        }
        Ok(Self { commitment, state })
    }

    /// Return the checkpoint to which all cached facts are bound.
    pub const fn commitment(&self) -> &CurrentCommitment<F, K, V, H, N> {
        &self.commitment
    }

    /// Return the trusted-state variant.
    pub const fn state(&self) -> &TrustedState<F, K, V::Value> {
        &self.state
    }

    /// Return the retained active entry for `key`, if any.
    pub fn entry(&self, key: &K) -> Option<&TrustedEntry<F, K, V::Value>> {
        self.partial().and_then(|state| state.entry(key))
    }

    /// Return a retained adjacency proving `key` absent, excluding the endpoints.
    /// Includes edges retained in entries as well as explicit spans. `None` means there
    /// is no covering cached edge. Inspect `state` for the separate empty-database fact.
    pub fn absence_span(&self, key: &K) -> Option<(&K, &K)> {
        self.partial().and_then(|state| state.containing_edge(key))
    }

    /// Import or replace a caller-trusted active record at this map's commitment.
    /// Require a partial cache, a location before the terminal if known, and consistency
    /// with other retained records and spans. Restore the old entry if insertion fails.
    /// This is not proof verification: the caller must already know the new fields are
    /// true. An existing independently retained span may prevent changing a successor.
    pub fn replace_entry(
        &mut self,
        entry: TrustedEntry<F, K, V::Value>,
    ) -> Result<(), VerifyError> {
        if matches!(
            entry.location,
            KnownTrusted(location)
                if location
                    >= self.commitment.size() - 1
        ) {
            return Err(VerifyError::InvalidTrustedState);
        }
        let TrustedState::Partial(state) = &mut self.state else {
            return Err(VerifyError::InvalidTrustedState);
        };
        let old = state.remove_entry(&entry.key);
        if let Err(error) = state.insert_entry(entry) {
            if let Some(old) = old {
                state
                    .insert_entry(old)
                    .map_err(|_| VerifyError::InvalidTrustedState)?;
            }
            return Err(error);
        }
        Ok(())
    }

    /// Forget a retained record and return it, without changing the committed database.
    /// An explicit span may still establish this key's membership or adjacency. Removing
    /// a record is not a request to delete the key from the next batch.
    pub fn remove_entry(&mut self, key: &K) -> Option<TrustedEntry<F, K, V::Value>> {
        match &mut self.state {
            TrustedState::Partial(state) => state.remove_entry(key),
            TrustedState::EmptyDatabase => None,
        }
    }

    /// Import a caller-trusted immediate predecessor-successor adjacency.
    /// Both endpoints must be active at this map's commitment and the open cyclic interval
    /// must contain no active key. Local consistency checks cannot authenticate those
    /// assertions. Reject an empty-database cache or conflicts with retained knowledge.
    pub fn install_span(&mut self, predecessor: K, successor: K) -> Result<(), VerifyError> {
        let TrustedState::Partial(state) = &mut self.state else {
            return Err(VerifyError::InvalidTrustedState);
        };
        state.insert_span(predecessor, successor)
    }

    /// Forget an explicit span and return its successor, without changing the database.
    /// An entry with a known successor may continue to establish the same absence interval.
    pub fn remove_span(&mut self, predecessor: &K) -> Option<K> {
        match &mut self.state {
            TrustedState::Partial(state) => state.remove_span(predecessor),
            TrustedState::EmptyDatabase => None,
        }
    }

    /// Borrow partial knowledge, or return `None` for the known-empty database variant.
    pub(crate) const fn partial(&self) -> Option<&PartialTrustedState<F, K, V::Value>> {
        match &self.state {
            TrustedState::Partial(state) => Some(state),
            TrustedState::EmptyDatabase => None,
        }
    }

    /// Replace the working cache state with an authenticated or locally derived state.
    /// The caller must preserve its truth at the working commitment. This helper performs
    /// no proof checks and is only used within transition verification.
    pub(crate) fn set_state(&mut self, state: TrustedState<F, K, V::Value>) {
        self.state = state;
    }

    /// Change the working checkpoint before rebasing facts through a verified transition.
    /// Old facts are not automatically valid at the new root. The caller must complete
    /// rebasing before returning the cache, and discard it if rebasing fails.
    pub(crate) const fn rebind(&mut self, commitment: CurrentCommitment<F, K, V, H, N>) {
        self.commitment = commitment;
    }
}

#[cfg(test)]
mod tests {
    use super::{OptionTrusted::Unknown, *};
    use crate::{merkle::mmr, qmdb::any::value::FixedEncoding};
    use alloc::vec;
    use commonware_cryptography::Sha256;
    use commonware_utils::sequence::FixedBytes;

    type F = mmr::Family;
    type Key = FixedBytes<8>;
    type State = PartialTrustedState<F, Key, Key>;
    type Map = TrustedMap<F, Key, FixedEncoding<Key>, Sha256, 32>;

    fn key(index: u64) -> Key {
        FixedBytes::new(index.to_be_bytes())
    }

    /// A member at `location`, holding its own key as value, with successor `next` if known.
    fn entry(index: u64, location: u64, next: Option<u64>) -> TrustedEntry<F, Key, Key> {
        TrustedEntry {
            key: key(index),
            location: KnownTrusted(Location::new(location)),
            value: KnownTrusted(key(index)),
            next_key: next.map_or(Unknown, |next| KnownTrusted(key(next))),
        }
    }

    fn span(predecessor: u64, successor: u64) -> (Key, Key) {
        (key(predecessor), key(successor))
    }

    /// A ten-operation checkpoint; its terminal commit is at location 9.
    fn commitment() -> CurrentCommitment<F, Key, FixedEncoding<Key>, Sha256, 32> {
        CurrentCommitment::from_parts(Location::new(10), Sha256::hash(&[b"root".as_slice()]))
            .unwrap()
    }

    #[test]
    fn spans_are_open_cyclic_intervals() {
        let contains = |start: u64, end: u64, key: u64| strict_span_contains(&start, &end, &key);
        assert!(contains(3, 7, 5));
        assert!(!contains(3, 7, 3) && !contains(3, 7, 7) && !contains(3, 7, 9));
        // A wrapping span covers keys above its predecessor and below its successor.
        assert!(contains(7, 3, 9) && contains(7, 3, 0));
        assert!(!contains(7, 3, 5) && !contains(7, 3, 3) && !contains(7, 3, 7));
        // The edge of a one-key database covers every other key.
        assert!(contains(4, 4, 0) && contains(4, 4, 9) && !contains(4, 4, 4));

        let edges = BTreeMap::from([(2u64, 5u64), (5, 9), (9, 2)]);
        let find = |key: u64| containing_edge(&edges, &key, |next| next).map(|(start, _)| *start);
        assert_eq!(find(3), Some(2));
        assert_eq!(find(7), Some(5));
        assert_eq!(find(10), Some(9));
        assert_eq!(find(0), Some(9));
        assert_eq!(find(5), None);
    }

    #[test]
    fn contradictory_facts_are_rejected() {
        let check = |entries: Vec<_>, spans: Vec<_>| State::new(entries, spans).map(|_| ());
        let invalid = Err(VerifyError::InvalidTrustedState);
        // An explicit span may repeat an entry's edge, and the same span may arrive twice.
        assert_eq!(
            check(
                vec![entry(2, 1, Some(5)), entry(5, 2, Some(9))],
                vec![span(2, 5), span(2, 5)]
            ),
            Ok(())
        );
        // One key twice, or two keys at one location.
        assert_eq!(
            check(vec![entry(2, 1, None), entry(2, 3, None)], vec![]),
            invalid
        );
        assert_eq!(
            check(vec![entry(2, 1, None), entry(5, 1, None)], vec![]),
            invalid
        );
        // Two successors for one key, or two predecessors for one key.
        assert_eq!(check(vec![entry(2, 1, Some(5))], vec![span(2, 7)]), invalid);
        assert_eq!(
            check(vec![entry(2, 1, Some(9)), entry(5, 2, Some(9))], vec![]),
            invalid
        );
        // A member inside a span, whichever is learned first.
        assert_eq!(check(vec![entry(5, 2, None)], vec![span(2, 9)]), invalid);
        assert_eq!(
            check(vec![entry(2, 1, Some(9)), entry(5, 2, None)], vec![]),
            invalid
        );
        // Overlapping spans, wrapping or not, and a one-key circle beside another key.
        assert_eq!(check(vec![], vec![span(3, 8), span(6, 9)]), invalid);
        assert_eq!(check(vec![], vec![span(8, 2), span(9, 3)]), invalid);
        assert_eq!(check(vec![], vec![span(8, 2), span(1, 4)]), invalid);
        assert_eq!(
            check(vec![entry(4, 1, Some(4)), entry(6, 2, None)], vec![]),
            invalid
        );
    }

    #[test]
    fn an_edge_survives_while_either_source_retains_it() {
        let mut state = State::new([entry(2, 1, Some(5))], [span(2, 5)]).unwrap();
        state.remove_entry(&key(2)).unwrap();
        assert_eq!(state.known_edge(&key(2)), Some(&key(5)));
        assert!(state.is_claimed_member(&key(2)) && state.is_claimed_member(&key(5)));
        assert_eq!(state.remove_span(&key(2)), Some(key(5)));
        assert_eq!(state.known_edge(&key(2)), None);
        assert_eq!(state.predecessor_of(&key(5)), None);
        assert!(!state.is_claimed_member(&key(2)));

        let mut state = State::new([entry(2, 1, Some(5))], [span(2, 5)]).unwrap();
        assert_eq!(state.remove_span(&key(2)), Some(key(5)));
        assert_eq!(state.containing_edge(&key(3)), Some((&key(2), &key(5))));
        state.remove_entry(&key(2)).unwrap();
        assert_eq!(state.containing_edge(&key(3)), None);
        // Forgetting the entry also frees its location.
        state.insert_entry(entry(7, 1, None)).unwrap();
    }

    #[test]
    fn the_map_bounds_locations_and_restores_a_refused_replacement() {
        let bind = |entries: Vec<_>| {
            Map::from_state(
                commitment(),
                TrustedState::Partial(State::new(entries, []).unwrap()),
            )
        };
        // The terminal commit holds location 9, so no member can.
        assert!(matches!(
            bind(vec![entry(2, 9, None)]),
            Err(VerifyError::InvalidTrustedState)
        ));
        let mut map = bind(vec![entry(2, 8, Some(5)), entry(5, 3, None)]).unwrap();
        map.install_span(key(5), key(2)).unwrap();
        assert_eq!(
            map.replace_entry(entry(2, 9, Some(5))),
            Err(VerifyError::InvalidTrustedState)
        );
        // A successor past the member 5 is refused, and the old entry stays.
        assert_eq!(
            map.replace_entry(entry(2, 8, Some(7))),
            Err(VerifyError::InvalidTrustedState)
        );
        assert_eq!(map.entry(&key(2)).unwrap().next_key, KnownTrusted(key(5)));
        assert_eq!(map.absence_span(&key(3)), Some((&key(2), &key(5))));
        map.replace_entry(entry(2, 7, Some(5))).unwrap();
        assert_eq!(
            map.entry(&key(2)).unwrap().location,
            KnownTrusted(Location::new(7))
        );

        // A database known to be empty holds no members or spans.
        let mut empty = Map::from_state(commitment(), TrustedState::EmptyDatabase).unwrap();
        assert_eq!(
            empty.install_span(key(2), key(5)),
            Err(VerifyError::InvalidTrustedState)
        );
        assert_eq!(
            empty.replace_entry(entry(2, 1, None)),
            Err(VerifyError::InvalidTrustedState)
        );
        assert!(empty.remove_entry(&key(2)).is_none());
        assert_eq!(empty.absence_span(&key(2)), None);
    }
}
