//! Root hashing shared by database and storage-free verification.

use crate::merkle::{
    Bagging, Family, Location,
    hasher::{Hasher as MerkleHasher, Standard as StandardHasher},
};
use commonware_codec::Encode;
use commonware_cryptography::Hasher;

/// Merkle peak bagging policy used by QMDB operation roots.
pub(crate) const ROOT_BAGGING: Bagging = Bagging::BackwardFold;

/// Return the Merkle hasher configuration used by QMDB operation roots and proofs.
pub const fn hasher<H: Hasher>() -> StandardHasher<H> {
    StandardHasher::new(ROOT_BAGGING)
}

/// Return the root of an operation log containing only `operation`.
///
/// This lets database variants derive their initial root from the bootstrap commit without
/// opening a database.
pub(super) fn single_operation_root<F: Family, H: Hasher>(operation: &impl Encode) -> H::Digest {
    let hasher = hasher::<H>();
    let leaf = MerkleHasher::<F>::leaf_digest(
        &hasher,
        F::location_to_position(Location::new(0)),
        &operation.encode(),
    );
    MerkleHasher::<F>::root(&hasher, Location::new(1), 0, [&leaf])
        .expect("a single-leaf Merkle root is always valid")
}
