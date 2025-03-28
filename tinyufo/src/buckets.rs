// Copyright 2025 Cloudflare, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Concurrent storage backend

use std::fmt::Debug;

use super::{Bucket, Key};
use ahash::RandomState;
use crossbeam_skiplist::{map::Entry, SkipMap};
use flurry::HashMap;
use scc::HashIndex;

/// N-shard skip list. Memory efficient, constant time lookup on average, but a bit slower
/// than hash map
#[derive(Debug)]
pub struct Compact<T>(Box<[SkipMap<Key, Bucket<T>>]>);

impl<T: Send + 'static + Debug> Compact<T> {
    /// Create a new [Compact]
    pub fn new(total_items: usize, items_per_shard: usize) -> Self {
        assert!(items_per_shard > 0);

        let shards = std::cmp::max(total_items / items_per_shard, 1);
        let mut shard_array = vec![];
        for _ in 0..shards {
            shard_array.push(SkipMap::new());
        }
        Self(shard_array.into_boxed_slice())
    }

    pub fn get(&self, key: &Key) -> Option<Entry<Key, Bucket<T>>> {
        let shard = *key as usize % self.0.len();
        self.0[shard].get(key)
    }

    pub fn get_map<V, F: FnOnce(Entry<Key, Bucket<T>>) -> V>(&self, key: &Key, f: F) -> Option<V> {
        let v = self.get(key);
        v.map(f)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    fn insert(&self, key: Key, value: Bucket<T>) -> Option<()> {
        let shard = key as usize % self.0.len();
        let removed = self.0[shard].remove(&key);
        self.0[shard].insert(key, value);
        removed.map(|_| ())
    }

    fn remove(&self, key: &Key) {
        let shard = *key as usize % self.0.len();
        (&self.0)[shard].remove(key);
    }
}

// Concurrent hash map, fast but use more memory
#[derive(Debug)]
pub struct Baseline<T>(HashMap<Key, Bucket<T>, RandomState>);

impl<T: Send + Sync> Baseline<T> {
    pub fn new(total_items: usize) -> Self {
        Self(HashMap::with_capacity_and_hasher(
            total_items,
            RandomState::new(),
        ))
    }

    pub fn get_map<V, F: FnOnce(&Bucket<T>) -> V>(&self, key: &Key, f: F) -> Option<V> {
        let pinned = self.0.pin();
        let v = pinned.get(key);
        v.map(f)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    fn insert(&self, key: Key, value: Bucket<T>) -> Option<()> {
        let pinned = self.0.pin();
        pinned.insert(key, value).map(|_| ())
    }

    fn remove(&self, key: &Key) {
        let pinned = self.0.pin();
        pinned.remove(key);
    }
}

#[derive(Debug)]
pub struct Fast<T: Clone + 'static>(HashIndex<Key, Bucket<T>, RandomState>);

impl<T: Send + Sync + 'static + Clone + Debug> Fast<T> {
    pub fn new(total_items: usize) -> Self {
        Self(HashIndex::with_capacity_and_hasher(
            total_items,
            RandomState::new(),
        ))
    }

    pub fn get_map<V, F: FnOnce(&Bucket<T>) -> V>(&self, key: &Key, f: F) -> Option<V> {
        self.0.peek_with(key, |_, v| f(v))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    fn insert(&self, key: Key, value: Bucket<T>) -> Option<()> {
        match self.0.insert(key, value) {
            Ok(_) => None,
            Err(_) => Some(()),
        }
    }

    fn remove(&self, key: &Key) {
        self.0.remove(key);
    }
}

#[derive(Debug)]
pub enum Buckets<T: Clone + 'static> {
    Fast(Box<Fast<T>>),
    Baseline(Box<Baseline<T>>),
    Compact(Compact<T>),
}

impl<T: Send + Sync + 'static + Clone + Debug> Buckets<T> {
    pub fn new_fast(items: usize) -> Self {
        Self::Fast(Box::new(Fast::new(items)))
    }

    pub fn new(items: usize) -> Self {
        Self::Baseline(Box::new(Baseline::new(items)))
    }

    pub fn new_compact(items: usize, items_per_shard: usize) -> Self {
        Self::Compact(Compact::new(items, items_per_shard))
    }

    pub fn insert(&self, key: Key, value: Bucket<T>) -> Option<()> {
        match self {
            Self::Compact(c) => c.insert(key, value),
            Self::Baseline(b) => b.insert(key, value),
            Self::Fast(f) => f.insert(key, value),
        }
    }

    pub fn remove(&self, key: &Key) {
        match self {
            Self::Compact(c) => c.remove(key),
            Self::Baseline(b) => b.remove(key),
            Self::Fast(f) => f.remove(key),
        }
    }

    pub fn get_map<V, F: FnOnce(&Bucket<T>) -> V>(&self, key: &Key, fo: F) -> Option<V> {
        match self {
            Self::Compact(c) => c.get_map(key, |v| fo(v.value())),
            Self::Baseline(b) => b.get_map(key, fo),
            Self::Fast(f) => f.get_map(key, fo),
        }
    }

    // Used for debugging
    pub fn len(&self) -> usize {
        match self {
            Self::Compact(c) => c.len(),
            Self::Baseline(b) => b.len(),
            Self::Fast(f) => f.len(),
        }
    }

    #[cfg(test)]
    pub fn get_queue(&self, key: &Key) -> Option<bool> {
        self.get_map(key, |v| v.queue.is_main())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast() {
        let fast = Buckets::new_fast(10);

        assert!(fast.get_map(&1, |_| ()).is_none());

        let bucket = Bucket {
            queue: crate::Location::new_small(),
            weight: 1,
            uses: Default::default(),
            data: 1,
        };
        fast.insert(1, bucket);

        assert_eq!(fast.get_map(&1, |v| v.data), Some(1));

        fast.remove(&1);
        assert!(fast.get_map(&1, |_| ()).is_none());
    }

    #[test]
    fn test_baseline() {
        let baseline = Buckets::new(10);

        assert!(baseline.get_map(&1, |_| ()).is_none());

        let bucket = Bucket {
            queue: crate::Location::new_small(),
            weight: 1,
            uses: Default::default(),
            data: 1,
        };
        baseline.insert(1, bucket);

        assert_eq!(baseline.get_map(&1, |v| v.data), Some(1));

        baseline.remove(&1);
        assert!(baseline.get_map(&1, |_| ()).is_none());
    }

    #[test]
    fn test_compact() {
        let compact = Buckets::new_compact(10, 2);

        assert!(compact.get_map(&1, |_| ()).is_none());

        let bucket = Bucket {
            queue: crate::Location::new_small(),
            weight: 1,
            uses: Default::default(),
            data: 1,
        };
        compact.insert(1, bucket);

        assert_eq!(compact.get_map(&1, |v| v.data), Some(1));

        compact.remove(&1);
        assert!(compact.get_map(&1, |_| ()).is_none());
    }
}
