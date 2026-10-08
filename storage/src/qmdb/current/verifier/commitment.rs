//! Typed Current commitments.

use crate::{
    merkle::{self, Graftable, Location},
    qmdb::{
        any::{ValueEncoding, initial_root, operation::update},
        current::{grafting, roots::combine_roots},
        operation::Key,
    },
};
use commonware_codec::{Encode, FixedSize as _};
use commonware_cryptography::Hasher;
use commonware_utils::bitmap::BitMap;
use core::{fmt, marker::PhantomData};

type ProfileMarker<K, V, H, const N: usize> = fn() -> (K, V, H, [(); N]);

/// A log size and canonical Current root tied to one ordered database profile.
/// `F`, `K`, `V`, `H`, and `N` fix the tree family, key type, value encoding, hash,
/// and bitmap chunk bytes. Equality compares size and root within that profile.
/// The type prevents accidental profile mixing. Construction does not establish
/// that a root describes a valid database or is trusted by the application.
pub struct CurrentCommitment<F: Graftable, K, V, H: Hasher, const N: usize> {
    size: Location<F>,
    root: H::Digest,
    _profile: PhantomData<ProfileMarker<K, V, H, N>>,
}

impl<F: Graftable, K, V, H: Hasher, const N: usize> CurrentCommitment<F, K, V, H, N> {
    /// Package a root and nonzero log size supported by the selected Merkle family.
    /// Use this for a checkpoint received from the application's protocol. It does not
    /// verify the root or its relationship to the size. The caller must establish trust
    /// in a pre-state checkpoint before verification. A claimed post-checkpoint may be
    /// untrusted because verification compares it with a locally derived result.
    pub fn from_parts(size: Location<F>, root: H::Digest) -> Result<Self, merkle::Error<F>> {
        if !size.is_valid() {
            return Err(merkle::Error::LocationOverflow(size));
        }
        if size == Location::new(0) {
            return Err(merkle::Error::InvalidSize(0));
        }
        Ok(Self {
            size,
            root,
            _profile: PhantomData,
        })
    }

    /// Return the operation count committed by this checkpoint.
    pub const fn size(&self) -> Location<F> {
        self.size
    }

    /// Return the canonical Current root.
    pub const fn root(&self) -> &H::Digest {
        &self.root
    }

    /// Consume the checkpoint into its decoded parts.
    pub const fn into_parts(self) -> (Location<F>, H::Digest) {
        (self.size, self.root)
    }
}

impl<F: Graftable, K, V, H: Hasher, const N: usize> Copy for CurrentCommitment<F, K, V, H, N> {}

impl<F: Graftable, K, V, H: Hasher, const N: usize> Clone for CurrentCommitment<F, K, V, H, N> {
    /// Copy the log size and root while retaining the same database profile.
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: Graftable, K, V, H: Hasher, const N: usize> PartialEq for CurrentCommitment<F, K, V, H, N> {
    /// Compare both operation count and canonical root within the same profile.
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.root == other.root
    }
}

impl<F: Graftable, K, V, H: Hasher, const N: usize> Eq for CurrentCommitment<F, K, V, H, N> {}

impl<F: Graftable, K, V, H: Hasher, const N: usize> fmt::Debug
    for CurrentCommitment<F, K, V, H, N>
{
    /// Display the committed log size and root.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CurrentCommitment")
            .field("size", &self.size)
            .field("root", &self.root)
            .finish()
    }
}

/// Compute the fixed empty database checkpoint, containing one `CommitFloor(None, 0)`.
/// Use this as the trusted starting point when verifying a database's first batch.
/// Its only activity bit is the terminal at location zero; there are no user records,
/// grafted chunks, or pending chunks. The partial bitmap digest binds this one bit.
/// The chunk profile must satisfy the same constraints as `UserBatch::verify`.
pub fn initial_commitment<F, K, V, H, const N: usize>() -> CurrentCommitment<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    crate::qmdb::any::operation::Operation<F, update::Ordered<K, V>>: Encode,
{
    validate_profile::<H, N>();

    let ops_root = initial_root::<F, update::Ordered<K, V>, H>();
    // Only the terminal at location zero is active.
    let mut chunk = BitMap::<N>::EMPTY_CHUNK;
    if let Some(first) = chunk.first_mut() {
        *first = 1;
    }
    let partial_digest = H::hash(&[chunk.as_slice()]);
    let root = combine_roots::<H>(&ops_root, &ops_root, None, Some((1, &partial_digest)));
    // One operation is a valid size for every family.
    CurrentCommitment {
        size: Location::new(1),
        root,
        _profile: PhantomData,
    }
}

/// Enforce the chunk geometry required by bitmap arithmetic and grafting.
/// `N` must be a power of two, a multiple of the digest size, and small enough for
/// `8 * N` to fit in `usize`, which also makes the grafting height positive. These are
/// programmer configuration requirements, checked when the profile is compiled.
pub(crate) const fn validate_profile<H: Hasher, const N: usize>() {
    const {
        assert!(N.is_multiple_of(H::Digest::SIZE));
        assert!(N.is_power_of_two());
        assert!(N <= usize::MAX / 8);
        assert!(grafting::height::<N>() >= 1);
    }
}
