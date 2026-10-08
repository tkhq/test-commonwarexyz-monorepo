//! Canonical Current root encoding shared by writing and verification.
use crate::qmdb;
use commonware_cryptography::Hasher;

/// Compute the canonical root from the ops root, grafted tree root, and optional pending /
/// partial chunk digests.
///
/// See [Canonical root structure](super::proof#canonical-root-structure) for the full layout.
/// The pending and partial inputs are independent: either, both, or neither may be set, and
/// pending precedes partial in hash order when both are present.
///
/// # Collision resistance
///
/// `pending` contributes `D` bytes when present; `partial` contributes `D + 8` bytes (`D` =
/// digest size). Different fixed lengths, so the two cannot produce the same input bytes,
/// even when their digests are identical. Collisions reduce to H.
pub(super) fn combine_roots<H: Hasher>(
    ops_root: &H::Digest,
    grafted_root: &H::Digest,
    pending: Option<&H::Digest>,
    partial: Option<(u64, &H::Digest)>,
) -> H::Digest {
    let hasher = qmdb::hasher::<H>();
    match (pending, partial) {
        (None, None) => hasher.hash(&[ops_root.as_ref(), grafted_root.as_ref()]),
        (Some(pe), None) => hasher.hash(&[ops_root.as_ref(), grafted_root.as_ref(), pe.as_ref()]),
        (None, Some((nb, p))) => {
            let nb_bytes = nb.to_be_bytes();
            hasher.hash(&[
                ops_root.as_ref(),
                grafted_root.as_ref(),
                nb_bytes.as_slice(),
                p.as_ref(),
            ])
        }
        (Some(pe), Some((nb, p))) => {
            let nb_bytes = nb.to_be_bytes();
            hasher.hash(&[
                ops_root.as_ref(),
                grafted_root.as_ref(),
                pe.as_ref(),
                nb_bytes.as_slice(),
                p.as_ref(),
            ])
        }
    }
}
