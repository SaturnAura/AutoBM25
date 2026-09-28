//! A small FxHash-style hasher.
//!
//! std's default hasher (SipHash) is chosen for HashDoS resistance; this workload
//! hashes millions of short tokens, where a multiply-xor hasher is several times
//! cheaper. Keys are corpus tokens, so the weaker avalanche is acceptable here
//! (same trade-off rustc itself makes internally).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            self.add(u64::from_le_bytes(buf));
        }
        let mut tail: u64 = 0;
        for (i, &b) in chunks.remainder().iter().enumerate() {
            tail |= (b as u64) << (8 * i);
        }
        self.add(tail);
    }

    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.add(v as u64);
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.add(v as u64);
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }

    #[inline]
    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
pub type FxHashSet<T> = HashSet<T, FxBuildHasher>;
