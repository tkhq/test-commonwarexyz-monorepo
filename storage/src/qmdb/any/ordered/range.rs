//! Shared non-cyclic traversal of canonical ordered successor edges.

use core::ops::{Bound, RangeBounds};

/// Traversal only: callers authenticate the lower-bound classification and every successor edge
/// before supplying them. Missing records are never evidence of completion.
pub(crate) struct RangeWalk<'a, K> {
    bounds: (Bound<&'a K>, Bound<&'a K>),
    next: Option<K>,
}

impl<'a, K: Ord + Clone> RangeWalk<'a, K> {
    pub(crate) fn new(
        bounds: (Bound<&'a K>, Bound<&'a K>),
        exact: bool,
        successor: Option<&K>,
    ) -> Result<Self, ()> {
        let (
            Bound::Included(start) | Bound::Excluded(start),
            Bound::Included(end) | Bound::Excluded(end),
        ) = bounds
        else {
            return Err(());
        };
        if start > end
            || (start == end
                && (!matches!(bounds.0, Bound::Included(_))
                    || !matches!(bounds.1, Bound::Included(_))))
        {
            return Err(());
        }
        let next = if exact && matches!(bounds.0, Bound::Included(_)) {
            Some(start.clone())
        } else {
            successor.filter(|next| *next > start).cloned()
        };
        Ok(Self { bounds, next })
    }

    pub(crate) fn next(&self) -> Option<&K> {
        self.next.as_ref().filter(|key| self.bounds.contains(key))
    }

    pub(crate) fn advance(&mut self, key: &K, successor: &K) -> bool {
        if self.next() != Some(key) {
            return false;
        }
        self.next = (successor > key).then(|| successor.clone());
        true
    }
}
