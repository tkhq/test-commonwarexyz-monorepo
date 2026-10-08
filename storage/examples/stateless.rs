//! End-to-end ordered Current proof production and stateless verification.
//! Run with `cargo run -p commonware-storage --example stateless`.
//!
//! The database produces a proof and independently merkleizes the authorized batch, and
//! sends the bundle as bytes. The client starts with only a trusted commitment. It decodes
//! the bundle within its limits and verifies it before the writer applies the batch, then
//! carries the facts it learned to the next transition. Both sides advance the inactivity
//! floor with the same policy, `Proportional`.

use commonware_codec::{Decode as _, DecodeExt as _, Encode as _};
use commonware_cryptography::Sha256;
use commonware_parallel::{Rayon, Sequential};
use commonware_runtime::{Runner as _, buffer::paged::CacheRef, deterministic};
use commonware_storage::{
    journal::contiguous::fixed::Config as JournalConfig,
    merkle::{Graftable, full::Config as MerkleConfig, mmb, mmr},
    qmdb::{
        any::value::FixedEncoding,
        current::{
            FixedConfig,
            ordered::fixed::Db,
            verifier::{
                BatchProofBundle, BundleConfig, CurrentCommitment, TrustedMap, UserBatch, UserOp,
            },
        },
        floor::Proportional,
    },
    translator::OneCap,
};
use commonware_utils::{NZU16, NZU64, NZUsize, sequence::FixedBytes};

type Key = FixedBytes<8>;
type Commitment<F> = CurrentCommitment<F, Key, FixedEncoding<Key>, Sha256, 32>;
type Bundle<F> = BatchProofBundle<F, Key, FixedEncoding<Key>, Sha256, 32>;

/// The largest bundle the client accepts.
const LIMITS: BundleConfig<(), (), (), ()> = BundleConfig {
    max_claims: 4_096,
    max_candidates: 4_096,
    max_bitmap_units: 4_096,
    max_digests: 4_096,
    key: (),
    update: (),
    operation: (),
    value: (),
};

fn key(value: u64) -> Key {
    Key::from(value.to_be_bytes())
}

/// Batches covering an empty commit, inserts across bitmap chunks, a range delete split and
/// rejoined by later edits, emptying the database, and repopulating it.
fn batches() -> Vec<UserBatch<Key, Key>> {
    let mut insert = UserBatch::new();
    for i in 0..600 {
        insert.set(key(i), key(i + 1));
    }
    let mut remove = UserBatch::new();
    remove.delete_range(key(10)..=key(590)).unwrap();
    remove.set(key(100), key(900)); // Split the range; this key survives.
    remove.set(key(200), key(901));
    remove.delete(key(200)); // Last wins: join this gap again.
    remove.set_metadata(key(42));
    let mut empty = UserBatch::new();
    empty.delete_range(key(0)..=key(1000)).unwrap();
    vec![
        UserBatch::new(),
        insert,
        remove,
        UserBatch::new(),
        empty,
        UserBatch::from([UserOp::set(key(7), key(8))]),
    ]
}

async fn run<F: Graftable>(context: deterministic::Context) {
    let page_cache = CacheRef::new(NZU16!(4096), NZUsize!(16));
    let config = FixedConfig {
        merkle_config: MerkleConfig {
            journal_partition: "stateless-nodes".into(),
            replay_buffer: NZUsize!(4096),
            metadata_partition: "stateless-merkle-metadata".into(),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            strategy: Sequential,
            page_cache: page_cache.clone(),
        },
        journal_config: JournalConfig {
            partition: "stateless-operations".into(),
            replay_buffer: NZUsize!(4096),
            items_per_blob: NZU64!(1024),
            write_buffer: NZUsize!(4096),
            page_cache,
        },
        grafted_metadata_partition: "stateless-grafted".into(),
        translator: OneCap,
        init_cache: Some(NZUsize!(1024)),
        init_buffer: NZUsize!(4096),
        init_concurrency: (),
    };
    let mut db = Db::<F, _, Key, Key, Sha256, _, 32, _>::init(context, config, None)
        .await
        .unwrap();
    // The client verifies on its own CPU pool; it needs no async runtime or storage.
    let verifier_strategy = Rayon::new(NZUsize!(4)).unwrap();
    let mut pre = Commitment::<F>::from_parts(db.bounds().end, db.root()).unwrap();
    let mut cache = TrustedMap::empty_cache(pre);
    for batch in batches() {
        let proof = batch.build_batch_proof(&db, &Proportional).await.unwrap();
        let merkleized = batch.merkleize(&db, &mut Proportional).await.unwrap();
        let post =
            Commitment::<F>::from_parts(merkleized.bounds().tip.size, merkleized.root()).unwrap();
        let (proof_bytes, post_bytes) = (proof.encode(), post.encode());

        // The client decodes what it received within its limits; verification checks the rest.
        let proof = Bundle::<F>::decode_cfg(proof_bytes, &LIMITS).unwrap();
        let post = Commitment::<F>::decode(post_bytes).unwrap();
        // A client with no cached facts can verify the complete bundle and keep nothing.
        let verified = batch
            .verify(
                pre,
                TrustedMap::empty_cache(pre),
                &proof,
                post,
                &mut Proportional,
            )
            .unwrap();
        assert_eq!(verified.commitment(), &post);
        // A client that kept its cache from the last transition verifies too, in parallel,
        // and carries its facts to the next one.
        cache = batch
            .verify_with_strategy(
                pre,
                cache,
                &proof,
                post,
                &verifier_strategy,
                &mut Proportional,
            )
            .unwrap()
            .into_cache()
            .unwrap();
        assert_eq!(cache.commitment(), &post);

        db = db.apply_batch(merkleized).await.unwrap();
        assert_eq!(db.root(), *post.root());
        pre = post;
    }
    assert_eq!(db.get(&key(7)).await.unwrap(), Some(key(8)));
    db.destroy().await.unwrap();
}

fn main() {
    deterministic::Runner::default().start(run::<mmr::Family>);
    deterministic::Runner::default().start(run::<mmb::Family>);
    println!("Proved and verified every batch for MMR and MMB.");
}
