//! Wire encodings for commitments and proof bundles.
//!
//! A recipient decodes a bundle from untrusted bytes, so every length is bounded by a
//! [`BundleConfig`] before anything is allocated. Decoding checks only the encoding; the
//! verifier checks everything a bundle claims.

use super::{
    BatchProofBundle, BitmapProof, BitmapUnit, ClaimedBatchBounds, CurrentCommitment,
    PreStateCandidateProof, PreStateExclusionClaim, PreStateKeyClaim, PreStateMemberClaim,
    PreStateOpeningClaim, PreStateTerminalClaim, ProvenBitmapProof,
};
use crate::{
    merkle::{self, Family, Graftable, Location},
    qmdb::{
        any::{
            ValueEncoding,
            operation::{Operation, update},
        },
        current::{ordered::proof::constant, proof::OpsRootWitness},
        operation::Key,
    },
};
use alloc::vec::Vec;
use bytes::BufMut;
use commonware_codec::{Buf, EncodeSize, Error, RangeCfg, Read, ReadExt as _, Write, varint::UInt};
use commonware_cryptography::{Digest, Hasher};

const INITIAL: u8 = 0;
const PROVEN: u8 = 1;
const MEMBER: u8 = 0;
const EXCLUSION: u8 = 1;
const EXPLICIT_CHUNK: u8 = 0;
const ZERO_SUBTREE: u8 = 1;

/// Decoding limits and element configurations for a [`BatchProofBundle`].
///
/// A decoder allocates no more than these limits allow, so a hostile encoding cannot force
/// large allocations. Choose them from the batches a recipient accepts: an honest bundle
/// carries about one key claim per mutated key, member of a range delete, and rewritten
/// predecessor; one candidate per old record the floor walk reaches; a few bitmap units per
/// changed or scanned chunk, plus O(log n) for each clear run between them; and O(log n)
/// digests per Merkle proof, for `n` operations in the log. Range deletes and the floor
/// policy's entry limit therefore set the claim and candidate limits.
#[derive(Clone, Debug)]
pub struct BundleConfig<K, U, O, V> {
    /// Most key claims.
    pub max_claims: usize,
    /// Most floor candidates.
    pub max_candidates: usize,
    /// Most bitmap units.
    pub max_bitmap_units: usize,
    /// Most digests in any one Merkle proof, pinned-node list, or append frontier.
    pub max_digests: usize,
    /// Read configuration for keys.
    pub key: K,
    /// Read configuration for the ordered updates exclusion proofs carry.
    pub update: U,
    /// Read configuration for candidate operations.
    pub operation: O,
    /// Read configuration for values and commit metadata.
    pub value: V,
}

/// The [`BundleConfig`] for a bundle of profile `F`, `K`, `V`.
pub type BundleConfigFor<F, K, V> = BundleConfig<
    <K as Read>::Cfg,
    <update::Ordered<K, V> as Read>::Cfg,
    <Operation<F, update::Ordered<K, V>> as Read>::Cfg,
    <<V as ValueEncoding>::Value as Read>::Cfg,
>;

impl<F: Graftable, K, V, H: Hasher, const N: usize> Write for CurrentCommitment<F, K, V, H, N> {
    fn write(&self, buf: &mut impl BufMut) {
        self.size().write(buf);
        self.root().write(buf);
    }
}

impl<F: Graftable, K, V, H: Hasher, const N: usize> EncodeSize
    for CurrentCommitment<F, K, V, H, N>
{
    fn encode_size(&self) -> usize {
        self.size().encode_size() + self.root().encode_size()
    }
}

impl<F: Graftable, K, V, H: Hasher, const N: usize> Read for CurrentCommitment<F, K, V, H, N> {
    type Cfg = ();

    fn read_cfg(buf: &mut impl Buf, _: &()) -> Result<Self, Error> {
        let size = Location::<F>::read(buf)?;
        let root = H::Digest::read(buf)?;
        Self::from_parts(size, root)
            .map_err(|_| Error::Invalid("CurrentCommitment", "size must be nonzero"))
    }
}

impl<F: Family> Write for ClaimedBatchBounds<F> {
    fn write(&self, buf: &mut impl BufMut) {
        self.pre_size.write(buf);
        self.pre_floor.write(buf);
        self.post_size.write(buf);
        self.post_floor.write(buf);
    }
}

impl<F: Family> EncodeSize for ClaimedBatchBounds<F> {
    fn encode_size(&self) -> usize {
        self.pre_size.encode_size()
            + self.pre_floor.encode_size()
            + self.post_size.encode_size()
            + self.post_floor.encode_size()
    }
}

impl<F: Family> Read for ClaimedBatchBounds<F> {
    type Cfg = ();

    fn read_cfg(buf: &mut impl Buf, _: &()) -> Result<Self, Error> {
        Ok(Self {
            pre_size: Location::read(buf)?,
            pre_floor: Location::read(buf)?,
            post_size: Location::read(buf)?,
            post_floor: Location::read(buf)?,
        })
    }
}

impl<F: Family, V: ValueEncoding, D: Digest> Write for PreStateTerminalClaim<F, V, D> {
    fn write(&self, buf: &mut impl BufMut) {
        self.metadata.write(buf);
        self.floor.write(buf);
        self.raw_proof.write(buf);
        self.pins_before_terminal.write(buf);
    }
}

impl<F: Family, V: ValueEncoding, D: Digest> EncodeSize for PreStateTerminalClaim<F, V, D> {
    fn encode_size(&self) -> usize {
        self.metadata.encode_size()
            + self.floor.encode_size()
            + self.raw_proof.encode_size()
            + self.pins_before_terminal.encode_size()
    }
}

impl<F: Family, V: ValueEncoding, D: Digest> Read for PreStateTerminalClaim<F, V, D> {
    /// `(max_digests, value_cfg)`: the most digests in the proof and in the pinned nodes, and
    /// the read configuration for the metadata.
    type Cfg = (usize, <V::Value as Read>::Cfg);

    fn read_cfg(buf: &mut impl Buf, (max_digests, value): &Self::Cfg) -> Result<Self, Error> {
        Ok(Self {
            metadata: Option::<V::Value>::read_cfg(buf, value)?,
            floor: Location::read(buf)?,
            raw_proof: merkle::Proof::read_cfg(buf, max_digests)?,
            pins_before_terminal: Vec::read_cfg(buf, &((..=*max_digests).into(), ()))?,
        })
    }
}

impl<F: Graftable, V: ValueEncoding, D: Digest, const N: usize> Write
    for PreStateOpeningClaim<F, V, D, N>
{
    fn write(&self, buf: &mut impl BufMut) {
        match self {
            Self::Initial => INITIAL.write(buf),
            Self::Proven {
                ops_root,
                witness,
                terminal,
            } => {
                PROVEN.write(buf);
                ops_root.write(buf);
                witness.write(buf);
                terminal.write(buf);
            }
        }
    }
}

impl<F: Graftable, V: ValueEncoding, D: Digest, const N: usize> EncodeSize
    for PreStateOpeningClaim<F, V, D, N>
{
    fn encode_size(&self) -> usize {
        1 + match self {
            Self::Initial => 0,
            Self::Proven {
                ops_root,
                witness,
                terminal,
            } => ops_root.encode_size() + witness.encode_size() + terminal.encode_size(),
        }
    }
}

impl<F: Graftable, V: ValueEncoding, D: Digest, const N: usize> Read
    for PreStateOpeningClaim<F, V, D, N>
{
    /// `(max_digests, value_cfg)`, as for [`PreStateTerminalClaim`].
    type Cfg = (usize, <V::Value as Read>::Cfg);

    fn read_cfg(buf: &mut impl Buf, cfg: &Self::Cfg) -> Result<Self, Error> {
        match u8::read(buf)? {
            INITIAL => Ok(Self::Initial),
            PROVEN => Ok(Self::Proven {
                ops_root: D::read(buf)?,
                witness: OpsRootWitness::read(buf)?,
                terminal: PreStateTerminalClaim::read_cfg(buf, cfg)?,
            }),
            tag => Err(Error::InvalidEnum(tag)),
        }
    }
}

impl<F: Graftable, K: Key, V: Write, D: Digest, const N: usize> Write
    for PreStateMemberClaim<F, K, V, D, N>
{
    fn write(&self, buf: &mut impl BufMut) {
        self.key.write(buf);
        self.value.write(buf);
        self.proof.write(buf);
    }
}

impl<F: Graftable, K: Key, V: EncodeSize, D: Digest, const N: usize> EncodeSize
    for PreStateMemberClaim<F, K, V, D, N>
{
    fn encode_size(&self) -> usize {
        self.key.encode_size() + self.value.encode_size() + self.proof.encode_size()
    }
}

impl<F: Graftable, K: Key, V: Read, D: Digest, const N: usize> Read
    for PreStateMemberClaim<F, K, V, D, N>
{
    /// `(max_digests, key_cfg, value_cfg)`.
    type Cfg = (usize, K::Cfg, V::Cfg);

    fn read_cfg(buf: &mut impl Buf, (max_digests, key, value): &Self::Cfg) -> Result<Self, Error> {
        Ok(Self {
            key: K::read_cfg(buf, key)?,
            value: V::read_cfg(buf, value)?,
            proof: constant::KeyValueProof::read_cfg(buf, &(*max_digests, key.clone()))?,
        })
    }
}

impl<F, K, V, D, const N: usize> Write for PreStateExclusionClaim<F, K, V, D, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    update::Ordered<K, V>: Write,
{
    fn write(&self, buf: &mut impl BufMut) {
        self.queried_key.write(buf);
        self.proof.write(buf);
    }
}

impl<F, K, V, D, const N: usize> EncodeSize for PreStateExclusionClaim<F, K, V, D, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    update::Ordered<K, V>: EncodeSize,
{
    fn encode_size(&self) -> usize {
        self.queried_key.encode_size() + self.proof.encode_size()
    }
}

impl<F, K, V, D, const N: usize> Read for PreStateExclusionClaim<F, K, V, D, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    update::Ordered<K, V>: Read,
{
    /// `(max_digests, key_cfg, update_cfg, value_cfg)`.
    type Cfg = (
        usize,
        K::Cfg,
        <update::Ordered<K, V> as Read>::Cfg,
        <V::Value as Read>::Cfg,
    );

    fn read_cfg(
        buf: &mut impl Buf,
        (max_digests, key, update, value): &Self::Cfg,
    ) -> Result<Self, Error> {
        Ok(Self {
            queried_key: K::read_cfg(buf, key)?,
            proof: constant::ExclusionProof::read_cfg(
                buf,
                &(*max_digests, update.clone(), value.clone()),
            )?,
        })
    }
}

impl<F, K, V, D, const N: usize> Write for PreStateKeyClaim<F, K, V, D, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    update::Ordered<K, V>: Write,
{
    fn write(&self, buf: &mut impl BufMut) {
        match self {
            Self::Member(claim) => {
                MEMBER.write(buf);
                claim.write(buf);
            }
            Self::Exclusion(claim) => {
                EXCLUSION.write(buf);
                claim.write(buf);
            }
        }
    }
}

impl<F, K, V, D, const N: usize> EncodeSize for PreStateKeyClaim<F, K, V, D, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    update::Ordered<K, V>: EncodeSize,
{
    fn encode_size(&self) -> usize {
        1 + match self {
            Self::Member(claim) => claim.encode_size(),
            Self::Exclusion(claim) => claim.encode_size(),
        }
    }
}

impl<F, K, V, D, const N: usize> Read for PreStateKeyClaim<F, K, V, D, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    update::Ordered<K, V>: Read,
{
    /// `(max_digests, key_cfg, update_cfg, value_cfg)`, as for [`PreStateExclusionClaim`].
    type Cfg = (
        usize,
        K::Cfg,
        <update::Ordered<K, V> as Read>::Cfg,
        <V::Value as Read>::Cfg,
    );

    fn read_cfg(buf: &mut impl Buf, cfg: &Self::Cfg) -> Result<Self, Error> {
        let (max_digests, key, _, value) = cfg;
        match u8::read(buf)? {
            MEMBER => Ok(Self::Member(PreStateMemberClaim::read_cfg(
                buf,
                &(*max_digests, key.clone(), value.clone()),
            )?)),
            EXCLUSION => Ok(Self::Exclusion(PreStateExclusionClaim::read_cfg(buf, cfg)?)),
            tag => Err(Error::InvalidEnum(tag)),
        }
    }
}

impl<F, K, V, D> Write for PreStateCandidateProof<F, K, V, D>
where
    F: Family,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    Operation<F, update::Ordered<K, V>>: Write,
{
    fn write(&self, buf: &mut impl BufMut) {
        self.operations.write(buf);
        self.raw_multi_proof.write(buf);
    }
}

impl<F, K, V, D> EncodeSize for PreStateCandidateProof<F, K, V, D>
where
    F: Family,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    Operation<F, update::Ordered<K, V>>: EncodeSize,
{
    fn encode_size(&self) -> usize {
        self.operations.encode_size() + self.raw_multi_proof.encode_size()
    }
}

impl<F, K, V, D> Read for PreStateCandidateProof<F, K, V, D>
where
    F: Family,
    K: Key,
    V: ValueEncoding,
    D: Digest,
    Operation<F, update::Ordered<K, V>>: Read,
{
    /// `(max_candidates, max_digests, operation_cfg)`.
    type Cfg = (
        usize,
        usize,
        <Operation<F, update::Ordered<K, V>> as Read>::Cfg,
    );

    fn read_cfg(
        buf: &mut impl Buf,
        (max_candidates, max_digests, operation): &Self::Cfg,
    ) -> Result<Self, Error> {
        let range: RangeCfg<usize> = (..=*max_candidates).into();
        Ok(Self {
            operations: Vec::read_cfg(buf, &(range, ((), operation.clone())))?,
            raw_multi_proof: Option::<merkle::Proof<F, D>>::read_cfg(buf, max_digests)?,
        })
    }
}

impl<F: Graftable, D: Digest, const N: usize> Write for BitmapUnit<F, D, N> {
    fn write(&self, buf: &mut impl BufMut) {
        match self {
            Self::ExplicitChunk {
                chunk_index,
                bytes,
                raw_subtree_digest,
            } => {
                EXPLICIT_CHUNK.write(buf);
                UInt(*chunk_index).write(buf);
                bytes.write(buf);
                raw_subtree_digest.write(buf);
            }
            Self::ZeroSubtree {
                first_chunk,
                chunk_height,
                common_subtree_digest,
                ..
            } => {
                ZERO_SUBTREE.write(buf);
                UInt(*first_chunk).write(buf);
                UInt(*chunk_height).write(buf);
                common_subtree_digest.write(buf);
            }
        }
    }
}

impl<F: Graftable, D: Digest, const N: usize> EncodeSize for BitmapUnit<F, D, N> {
    fn encode_size(&self) -> usize {
        1 + match self {
            Self::ExplicitChunk {
                chunk_index,
                bytes,
                raw_subtree_digest,
            } => {
                UInt(*chunk_index).encode_size()
                    + bytes.encode_size()
                    + raw_subtree_digest.encode_size()
            }
            Self::ZeroSubtree {
                first_chunk,
                chunk_height,
                common_subtree_digest,
                ..
            } => {
                UInt(*first_chunk).encode_size()
                    + UInt(*chunk_height).encode_size()
                    + common_subtree_digest.encode_size()
            }
        }
    }
}

impl<F: Graftable, D: Digest, const N: usize> Read for BitmapUnit<F, D, N> {
    type Cfg = ();

    fn read_cfg(buf: &mut impl Buf, _: &()) -> Result<Self, Error> {
        match u8::read(buf)? {
            EXPLICIT_CHUNK => Ok(Self::ExplicitChunk {
                chunk_index: UInt::read(buf)?.into(),
                bytes: <[u8; N]>::read(buf)?,
                raw_subtree_digest: Option::<D>::read(buf)?,
            }),
            ZERO_SUBTREE => Ok(Self::zero_subtree(
                UInt::read(buf)?.into(),
                UInt::read(buf)?.into(),
                D::read(buf)?,
            )),
            tag => Err(Error::InvalidEnum(tag)),
        }
    }
}

impl<F: Graftable, D: Digest, const N: usize> Write for ProvenBitmapProof<F, D, N> {
    fn write(&self, buf: &mut impl BufMut) {
        self.pre_size.write(buf);
        self.floor_scan_end.write(buf);
        self.units.write(buf);
        self.raw_proof.write(buf);
        self.grafted_proof.write(buf);
        self.grafted_append_frontier.write(buf);
    }
}

impl<F: Graftable, D: Digest, const N: usize> EncodeSize for ProvenBitmapProof<F, D, N> {
    fn encode_size(&self) -> usize {
        self.pre_size.encode_size()
            + self.floor_scan_end.encode_size()
            + self.units.encode_size()
            + self.raw_proof.encode_size()
            + self.grafted_proof.encode_size()
            + self.grafted_append_frontier.encode_size()
    }
}

impl<F: Graftable, D: Digest, const N: usize> Read for ProvenBitmapProof<F, D, N> {
    /// `(max_units, max_digests)`: the most coverage units, and the most digests in each proof
    /// and in the append frontier.
    type Cfg = (usize, usize);

    fn read_cfg(buf: &mut impl Buf, (max_units, max_digests): &Self::Cfg) -> Result<Self, Error> {
        Ok(Self {
            pre_size: Location::read(buf)?,
            floor_scan_end: Location::read(buf)?,
            units: Vec::read_cfg(buf, &((..=*max_units).into(), ()))?,
            raw_proof: Option::<merkle::Proof<F, D>>::read_cfg(buf, max_digests)?,
            grafted_proof: Option::<merkle::Proof<F, D>>::read_cfg(buf, max_digests)?,
            grafted_append_frontier: Vec::read_cfg(buf, &((..=*max_digests).into(), ()))?,
        })
    }
}

impl<F: Graftable, D: Digest, const N: usize> Write for BitmapProof<F, D, N> {
    fn write(&self, buf: &mut impl BufMut) {
        match self {
            Self::Initial { floor_scan_end } => {
                INITIAL.write(buf);
                floor_scan_end.write(buf);
            }
            Self::Proven(proof) => {
                PROVEN.write(buf);
                proof.write(buf);
            }
        }
    }
}

impl<F: Graftable, D: Digest, const N: usize> EncodeSize for BitmapProof<F, D, N> {
    fn encode_size(&self) -> usize {
        1 + match self {
            Self::Initial { floor_scan_end } => floor_scan_end.encode_size(),
            Self::Proven(proof) => proof.encode_size(),
        }
    }
}

impl<F: Graftable, D: Digest, const N: usize> Read for BitmapProof<F, D, N> {
    /// `(max_units, max_digests)`, as for [`ProvenBitmapProof`].
    type Cfg = (usize, usize);

    fn read_cfg(buf: &mut impl Buf, cfg: &Self::Cfg) -> Result<Self, Error> {
        match u8::read(buf)? {
            INITIAL => Ok(Self::Initial {
                floor_scan_end: Location::read(buf)?,
            }),
            PROVEN => Ok(Self::Proven(ProvenBitmapProof::read_cfg(buf, cfg)?)),
            tag => Err(Error::InvalidEnum(tag)),
        }
    }
}

impl<F, K, V, H, const N: usize> Write for BatchProofBundle<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    update::Ordered<K, V>: Write,
    Operation<F, update::Ordered<K, V>>: Write,
{
    fn write(&self, buf: &mut impl BufMut) {
        self.claims.write(buf);
        self.pre_state.write(buf);
        self.pre_state_claims.write(buf);
        self.bitmap.write(buf);
        self.pre_state_candidates.write(buf);
    }
}

impl<F, K, V, H, const N: usize> EncodeSize for BatchProofBundle<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    update::Ordered<K, V>: EncodeSize,
    Operation<F, update::Ordered<K, V>>: EncodeSize,
{
    fn encode_size(&self) -> usize {
        self.claims.encode_size()
            + self.pre_state.encode_size()
            + self.pre_state_claims.encode_size()
            + self.bitmap.encode_size()
            + self.pre_state_candidates.encode_size()
    }
}

impl<F, K, V, H, const N: usize> Read for BatchProofBundle<F, K, V, H, N>
where
    F: Graftable,
    K: Key,
    V: ValueEncoding,
    H: Hasher,
    update::Ordered<K, V>: Read,
    Operation<F, update::Ordered<K, V>>: Read,
{
    type Cfg = BundleConfigFor<F, K, V>;

    fn read_cfg(buf: &mut impl Buf, cfg: &Self::Cfg) -> Result<Self, Error> {
        let claim_cfg = (
            cfg.max_digests,
            cfg.key.clone(),
            cfg.update.clone(),
            cfg.value.clone(),
        );
        Ok(Self {
            claims: ClaimedBatchBounds::read(buf)?,
            pre_state: PreStateOpeningClaim::read_cfg(buf, &(cfg.max_digests, cfg.value.clone()))?,
            pre_state_claims: Vec::read_cfg(buf, &((..=cfg.max_claims).into(), claim_cfg))?,
            bitmap: BitmapProof::read_cfg(buf, &(cfg.max_bitmap_units, cfg.max_digests))?,
            pre_state_candidates: PreStateCandidateProof::read_cfg(
                buf,
                &(cfg.max_candidates, cfg.max_digests, cfg.operation.clone()),
            )?,
        })
    }
}

#[cfg(feature = "arbitrary")]
mod arbitrary_impls {
    use super::*;
    use arbitrary::{Arbitrary, Result, Unstructured};

    impl<F: Graftable, K, V, H: Hasher, const N: usize> Arbitrary<'_>
        for CurrentCommitment<F, K, V, H, N>
    where
        H::Digest: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            let size = Location::new(u.int_in_range(1..=*F::MAX_LEAVES)?);
            Self::from_parts(size, u.arbitrary()?).map_err(|_| arbitrary::Error::IncorrectFormat)
        }
    }

    impl<F: Family> Arbitrary<'_> for ClaimedBatchBounds<F> {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            Ok(Self {
                pre_size: u.arbitrary()?,
                pre_floor: u.arbitrary()?,
                post_size: u.arbitrary()?,
                post_floor: u.arbitrary()?,
            })
        }
    }

    impl<F: Family, V: ValueEncoding, D: Digest> Arbitrary<'_> for PreStateTerminalClaim<F, V, D>
    where
        V::Value: for<'a> Arbitrary<'a>,
        D: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            Ok(Self {
                metadata: u.arbitrary()?,
                floor: u.arbitrary()?,
                raw_proof: u.arbitrary()?,
                pins_before_terminal: u.arbitrary()?,
            })
        }
    }

    impl<F: Graftable, V: ValueEncoding, D: Digest, const N: usize> Arbitrary<'_>
        for PreStateOpeningClaim<F, V, D, N>
    where
        V::Value: for<'a> Arbitrary<'a>,
        D: for<'a> Arbitrary<'a>,
        OpsRootWitness<F, D>: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            if u.arbitrary()? {
                return Ok(Self::Initial);
            }
            Ok(Self::Proven {
                ops_root: u.arbitrary()?,
                witness: u.arbitrary()?,
                terminal: u.arbitrary()?,
            })
        }
    }

    impl<F: Graftable, K: Key, V, D: Digest, const N: usize> Arbitrary<'_>
        for PreStateMemberClaim<F, K, V, D, N>
    where
        K: for<'a> Arbitrary<'a>,
        V: for<'a> Arbitrary<'a>,
        constant::KeyValueProof<F, K, D, N>: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            Ok(Self {
                key: u.arbitrary()?,
                value: u.arbitrary()?,
                proof: u.arbitrary()?,
            })
        }
    }

    impl<F: Graftable, K: Key, V: ValueEncoding, D: Digest, const N: usize> Arbitrary<'_>
        for PreStateExclusionClaim<F, K, V, D, N>
    where
        K: for<'a> Arbitrary<'a>,
        constant::ExclusionProof<F, K, V, D, N>: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            Ok(Self {
                queried_key: u.arbitrary()?,
                proof: u.arbitrary()?,
            })
        }
    }

    impl<F: Graftable, K: Key, V: ValueEncoding, D: Digest, const N: usize> Arbitrary<'_>
        for PreStateKeyClaim<F, K, V, D, N>
    where
        PreStateMemberClaim<F, K, V::Value, D, N>: for<'a> Arbitrary<'a>,
        PreStateExclusionClaim<F, K, V, D, N>: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            if u.arbitrary()? {
                Ok(Self::Member(u.arbitrary()?))
            } else {
                Ok(Self::Exclusion(u.arbitrary()?))
            }
        }
    }

    impl<F: Family, K: Key, V: ValueEncoding, D: Digest> Arbitrary<'_>
        for PreStateCandidateProof<F, K, V, D>
    where
        Operation<F, update::Ordered<K, V>>: for<'a> Arbitrary<'a>,
        D: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            Ok(Self {
                operations: u.arbitrary()?,
                raw_multi_proof: u.arbitrary()?,
            })
        }
    }

    impl<F: Graftable, D: Digest, const N: usize> Arbitrary<'_> for BitmapUnit<F, D, N>
    where
        D: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            if u.arbitrary()? {
                Ok(Self::ExplicitChunk {
                    chunk_index: u.arbitrary()?,
                    bytes: u.arbitrary()?,
                    raw_subtree_digest: u.arbitrary()?,
                })
            } else {
                Ok(Self::zero_subtree(
                    u.arbitrary()?,
                    u.arbitrary()?,
                    u.arbitrary()?,
                ))
            }
        }
    }

    impl<F: Graftable, D: Digest, const N: usize> Arbitrary<'_> for ProvenBitmapProof<F, D, N>
    where
        D: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            Ok(Self {
                pre_size: u.arbitrary()?,
                floor_scan_end: u.arbitrary()?,
                units: u.arbitrary()?,
                raw_proof: u.arbitrary()?,
                grafted_proof: u.arbitrary()?,
                grafted_append_frontier: u.arbitrary()?,
            })
        }
    }

    impl<F: Graftable, D: Digest, const N: usize> Arbitrary<'_> for BitmapProof<F, D, N>
    where
        D: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            if u.arbitrary()? {
                Ok(Self::Initial {
                    floor_scan_end: u.arbitrary()?,
                })
            } else {
                Ok(Self::Proven(u.arbitrary()?))
            }
        }
    }

    impl<F, K, V, H, const N: usize> Arbitrary<'_> for BatchProofBundle<F, K, V, H, N>
    where
        F: Graftable,
        K: Key,
        V: ValueEncoding,
        H: Hasher,
        H::Digest: for<'a> Arbitrary<'a>,
        PreStateOpeningClaim<F, V, H::Digest, N>: for<'a> Arbitrary<'a>,
        PreStateKeyClaim<F, K, V, H::Digest, N>: for<'a> Arbitrary<'a>,
        PreStateCandidateProof<F, K, V, H::Digest>: for<'a> Arbitrary<'a>,
    {
        fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
            Ok(Self::new(
                u.arbitrary()?,
                u.arbitrary()?,
                u.arbitrary()?,
                u.arbitrary()?,
                u.arbitrary()?,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        merkle::{mmb, mmr},
        qmdb::any::value::FixedEncoding,
    };
    use commonware_codec::{Decode as _, DecodeExt as _, Encode as _};
    use commonware_cryptography::Sha256;
    use commonware_utils::sequence::FixedBytes;

    type Key = FixedBytes<8>;
    type Commitment = CurrentCommitment<mmr::Family, Key, FixedEncoding<Key>, Sha256, 32>;

    #[test]
    fn a_commitment_round_trips_and_refuses_size_zero() {
        let commitment =
            Commitment::from_parts(Location::new(300), Sha256::hash(&[b"root".as_slice()]))
                .unwrap();
        let encoded = commitment.encode();
        assert_eq!(encoded.len(), commitment.encode_size());
        assert_eq!(Commitment::decode(encoded).unwrap(), commitment);

        let mut zero = Location::<mmr::Family>::new(0).encode().to_vec();
        zero.extend_from_slice(commitment.root().as_ref());
        assert!(Commitment::decode(bytes::Bytes::from(zero)).is_err());
    }

    #[test]
    fn bitmap_units_round_trip() {
        let digest = Sha256::hash(&[b"chunk".as_slice()]);
        let units = [
            BitmapUnit::<mmb::Family, _, 32>::ExplicitChunk {
                chunk_index: 1 << 40,
                bytes: [7; 32],
                raw_subtree_digest: Some(digest),
            },
            BitmapUnit::ExplicitChunk {
                chunk_index: 3,
                bytes: [0; 32],
                raw_subtree_digest: None,
            },
            BitmapUnit::zero_subtree(8, 3, digest),
        ];
        for unit in units {
            let encoded = unit.encode();
            assert_eq!(encoded.len(), unit.encode_size());
            assert_eq!(BitmapUnit::decode(encoded).unwrap(), unit);
        }
        assert!(matches!(
            BitmapUnit::<mmb::Family, <Sha256 as Hasher>::Digest, 32>::decode(
                bytes::Bytes::from_static(&[2])
            ),
            Err(Error::InvalidEnum(2))
        ));
    }

    #[test]
    fn bitmap_proofs_refuse_lengths_past_their_limits() {
        let digest = Sha256::hash(&[b"frontier".as_slice()]);
        let proof = BitmapProof::<mmr::Family, _, 32>::Proven(ProvenBitmapProof {
            pre_size: Location::new(1000),
            floor_scan_end: Location::new(600),
            units: (0..4)
                .map(|index| BitmapUnit::zero_subtree(index, 0, digest))
                .collect(),
            raw_proof: None,
            grafted_proof: None,
            grafted_append_frontier: alloc::vec![digest; 3],
        });
        let encoded = proof.encode();
        assert_eq!(
            BitmapProof::decode_cfg(encoded.clone(), &(4, 3)).unwrap(),
            proof
        );
        assert!(
            BitmapProof::<mmr::Family, <Sha256 as Hasher>::Digest, 32>::decode_cfg(
                encoded.clone(),
                &(3, 3)
            )
            .is_err()
        );
        assert!(
            BitmapProof::<mmr::Family, <Sha256 as Hasher>::Digest, 32>::decode_cfg(
                encoded,
                &(4, 2)
            )
            .is_err()
        );
    }

    #[cfg(feature = "arbitrary")]
    mod conformance {
        use super::*;
        use commonware_codec::conformance::CodecConformance;

        type MmrBundle = BatchProofBundle<mmr::Family, Key, FixedEncoding<Key>, Sha256, 32>;
        type MmbBundle = BatchProofBundle<mmb::Family, Key, FixedEncoding<Key>, Sha256, 32>;

        commonware_conformance::conformance_tests! {
            CodecConformance<Commitment>,
            CodecConformance<MmrBundle> => 256,
            CodecConformance<MmbBundle> => 256,
        }
    }
}
