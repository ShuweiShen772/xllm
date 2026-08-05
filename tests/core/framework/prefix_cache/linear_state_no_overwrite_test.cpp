/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Scheduler-side unit tests for the linear-state "no-overwrite" (out-of-place)
// rotation work. These tests are pure CPU: they exercise slot allocation,
// checkpoint promotion, and reference counting on LinearStateBlockManager --
// none of the GDN operator numerics (those need a real Ascend device and are
// covered separately). They pin the slot-level invariants the out-of-place
// design relies on:
//
//   * A dst slot handed to the forward is a distinct physical slot from the src
//     it reads, so writing dst can never corrupt src.
//   * A slot promoted into the checkpoint index is shared/immutable and
//     outlives the sequence's own live handle -- the property that later makes
//     it offloadable, exactly like a KV block.
//   * Out-of-place rotation transiently holds BOTH src and dst, so the pool
//     must expose at least two concurrent live slots (the same peak the
//     existing in-place rotation already requires, since it allocates the new
//     slot before releasing the old one).
//
// These target the scheduler-side half of the linear-state out-of-place change
// (enable_linear_state_out_of_place); operator numerics are covered by the NPU
// test causal_conv1d_no_overwrite_tests.cpp.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "framework/block/block_manager_pool_test_peer.h"
#include "framework/block/linear_state_block_manager.h"
#include "util/hash_util.h"

namespace xllm {
namespace {

// Distinct prefix hash keyed by one byte so a checkpoint can be referenced by
// an easy-to-read tag (mirrors linear_state_prefix_cache_test.cpp).
XXH3Key make_hash(uint8_t tag) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> bytes{};
  bytes.fill(tag);
  return XXH3Key(bytes.data());
}

// Promote a freshly allocated slot into the checkpoint index under |hash|,
// returning its slot id (or -1 on pool exhaustion). Models the scheduler-side
// "old live slot becomes an immutable checkpoint" step of save-rotation.
int32_t insert_checkpoint(LinearStateBlockManager& cache, const XXH3Key& hash) {
  if (BlockManagerPoolTestPeer::contains(&cache, hash)) {
    Block matched = BlockManagerPoolTestPeer::match(&cache, hash);
    return matched.is_valid() ? matched.id() : -1;
  }
  Block slot_block = cache.allocate();
  if (!slot_block.is_valid()) {
    return -1;
  }
  const int32_t slot = slot_block.id();
  slot_block.set_hash_value(hash.data);
  std::vector<Block> checkpoint;
  checkpoint.emplace_back(std::move(slot_block));
  cache.cache(checkpoint);
  return slot;
}

}  // namespace

// The dst slot handed to the operator must be physically distinct from the src
// slot it reads. This is the core precondition of out-of-place writes: writing
// dst can never corrupt src. Padding slot 0 is reserved and never handed out.
TEST(LinearStateNoOverwriteTest, SrcAndDstSlotsAreDistinctPhysicalIds) {
  LinearStateBlockManager cache(/*num_slots=*/4, /*kv_block_size=*/16);

  Block src = cache.allocate();
  Block dst = cache.allocate();
  ASSERT_TRUE(src.is_valid());
  ASSERT_TRUE(dst.is_valid());
  EXPECT_NE(src.id(), dst.id());
  EXPECT_GE(src.id(), 1);
  EXPECT_GE(dst.id(), 1);
}

// Once an old slot is promoted to a checkpoint, it is shared (ref_count > 1):
// the index holds one alias. This is exactly what makes it safe to treat as an
// immutable snapshot -- the same property a KV block needs to be offloadable.
TEST(LinearStateNoOverwriteTest, CheckpointSlotIsSharedAndImmutable) {
  LinearStateBlockManager cache(/*num_slots=*/4, /*kv_block_size=*/16);
  const XXH3Key h = make_hash(7);

  const int32_t slot = insert_checkpoint(cache, h);
  ASSERT_GE(slot, 1);

  Block pinned = BlockManagerPoolTestPeer::match(&cache, h);
  ASSERT_TRUE(pinned.is_valid());
  EXPECT_EQ(pinned.id(), slot);
  // Held by index + our match handle -> shared. A shared checkpoint must not be
  // overwritten in place; the no-overwrite change routes new writes to a dst
  // slot instead.
  EXPECT_TRUE(pinned.is_shared());
}

// A checkpoint slot survives the drop of the sequence's own (live) handle: the
// index keeps it alive. This is the "old slot outlives the sequence as a
// snapshot" property the no-overwrite design relies on.
TEST(LinearStateNoOverwriteTest, CheckpointSurvivesLiveHandleDrop) {
  LinearStateBlockManager cache(/*num_slots=*/4, /*kv_block_size=*/16);
  const XXH3Key h = make_hash(3);

  int32_t slot = -1;
  {
    Block live = cache.allocate();
    ASSERT_TRUE(live.is_valid());
    slot = live.id();
    live.set_hash_value(h.data);
    std::vector<Block> checkpoint;
    checkpoint.emplace_back(std::move(live));
    cache.cache(checkpoint);
    // 'live' handle destroyed here at scope exit; index alias remains.
  }
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h));
  Block still_there = BlockManagerPoolTestPeer::match(&cache, h);
  ASSERT_TRUE(still_there.is_valid());
  EXPECT_EQ(still_there.id(), slot);
}

// Pool-sizing guard. Out-of-place rotation transiently holds BOTH the src (old
// live / becoming checkpoint) and the dst (new live) slot at a chunk boundary,
// so a pool must expose at least two concurrent live slots. With N physical
// slots, N-1 are usable (slot 0 is padding): N=3 yields two, N=2 yields one.
TEST(LinearStateNoOverwriteTest, PoolExposesTwoConcurrentLiveSlotsForRotation) {
  LinearStateBlockManager cache(/*num_slots=*/3, /*kv_block_size=*/16);

  Block src = cache.allocate();
  Block dst = cache.allocate();
  ASSERT_TRUE(src.is_valid());
  ASSERT_TRUE(dst.is_valid());
  EXPECT_NE(src.id(), dst.id());

  // A pool sized for exactly one live slot (num_slots=2 -> 1 usable) cannot
  // hold src and dst at once; document that boundary explicitly.
  LinearStateBlockManager tight(/*num_slots=*/2, /*kv_block_size=*/16);
  Block only = tight.allocate();
  ASSERT_TRUE(only.is_valid());
  EXPECT_FALSE(tight.allocate().is_valid())
      << "num_slots=2 yields a single live slot; out-of-place rotation needs "
         "the pool to expose two concurrent live slots.";
}

// A checkpoint promoted while the sequence keeps its own live alias is shared,
// and stays discoverable after the live alias is dropped -- the exact shape of
// out-of-place rotation, where the old slot is frozen into the index (via
// copy_block) while the sequence rotates onto a fresh dst. Models the two
// aliases (index + read-src pin) the block manager holds during rotation and
// asserts the checkpoint survives once those transient handles are released.
TEST(LinearStateNoOverwriteTest, FrozenSourceSurvivesRotationAliases) {
  LinearStateBlockManager cache(/*num_slots=*/4, /*kv_block_size=*/16);
  const XXH3Key h = make_hash(5);

  int32_t frozen_slot = -1;
  {
    Block old_live = cache.allocate();
    ASSERT_TRUE(old_live.is_valid());
    frozen_slot = old_live.id();
    old_live.set_hash_value(h.data);

    // Pin a second alias, mirroring the read-src handle the forward reads from
    // while writing the fresh dst; the shared old slot must not be reused.
    Block read_src_pin = old_live;
    ASSERT_TRUE(read_src_pin.is_valid());
    EXPECT_TRUE(old_live.is_shared());

    std::vector<Block> checkpoint;
    checkpoint.emplace_back(std::move(old_live));
    cache.cache(checkpoint);

    // The fresh dst live slot is a distinct physical slot.
    Block dst = cache.allocate();
    ASSERT_TRUE(dst.is_valid());
    EXPECT_NE(dst.id(), frozen_slot);
    // read-src pin and dst handle drop here; the index alias remains.
  }

  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h));
  Block matched = BlockManagerPoolTestPeer::match(&cache, h);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), frozen_slot);
}

// Re-inserting the same hash is idempotent: the checkpoint index dedups, so a
// repeated boundary hash never consumes a second physical slot. Guards against
// slot leaks when the same prefix boundary is promoted more than once.
TEST(LinearStateNoOverwriteTest, DuplicateCheckpointHashIsIdempotent) {
  LinearStateBlockManager cache(/*num_slots=*/4, /*kv_block_size=*/16);
  const XXH3Key h = make_hash(9);

  const int32_t first = insert_checkpoint(cache, h);
  ASSERT_GE(first, 1);
  const int32_t second = insert_checkpoint(cache, h);
  EXPECT_EQ(second, first);
}

}  // namespace xllm
