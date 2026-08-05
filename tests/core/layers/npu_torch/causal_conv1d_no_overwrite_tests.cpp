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

// NPU numeric test for the causal_conv1d out-of-place ("no-overwrite") write.
// It verifies the one property that makes the kernel change correct:
//
//   Reading from a src slot and writing to a distinct dst slot must produce, in
//   dst, the exact conv state that in-place (read X, write X) produces -- AND
//   must leave the src slot byte-for-byte unchanged (that untouched src is the
//   immutable checkpoint the scheduler freezes).
//
// Strategy (no golden file): run causal_conv1d_qkv TWICE from identical inputs
//   * pass A in-place  : cache_indices=[src], out_cache_indices=[] (empty)
//   * pass B out-of-place: cache_indices=[src], out_cache_indices=[dst]
// then assert conv_cache[dst] (after B) == conv_cache[src] (after A) and that
// conv_cache[src] is unchanged by pass B. The in-place run is its own
// reference, so the test needs no external baseline.
//
// Requires a real Ascend device AND the rebuilt custom operator package (the
// causal_conv1d op def carrying the new outCacheIndices input). The whole body
// is guarded by npu_available(); on a host without NPU it reports skipped.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "kernels/npu/npu_ops_api.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {
namespace {

bool npu_available() {
  return Platform::is_npu() && Platform::device_count() > 0;
}

// Small, deterministic packed-QKV conv config. head_k_dim=head_v_dim=128 is
// required by the direct packed-qkv path; keep one head each for a fast, tiny
// state.
constexpr int64_t kNumSlots =
    4;  // >=3 so src and dst are distinct usable slots
constexpr int64_t kSrcSlot = 1;
constexpr int64_t kDstSlot = 2;
constexpr int64_t kNumQkHeads = 1;
constexpr int64_t kNumVHeads = 1;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kConvStateLen = 3;  // conv_kernel_dim - 1
constexpr int64_t kWidth = 4;         // conv kernel width

}  // namespace

TEST(CausalConv1dNoOverwriteNpuTest, DstEqualsInPlaceAndSrcFrozen) {
  if (!npu_available()) {
    GTEST_SKIP() << "requires a real Ascend NPU device.";
  }

  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  const int64_t conv_dim =
      (kNumQkHeads * kHeadDim) * 2 + kNumVHeads * kHeadDim;  // q + k + v
  const int64_t num_tokens = 8;

  // Deterministic inputs, shared by both passes.
  torch::Tensor x =
      test::seeded_tensor(
          "conv.noov.x", {num_tokens, conv_dim}, torch::kFloat32, device)
          .to(torch::kBFloat16);
  torch::Tensor weight =
      test::seeded_tensor(
          "conv.noov.w", {kWidth, conv_dim}, torch::kFloat32, device)
          .to(torch::kBFloat16);

  // A conv_cache seeded so the src slot holds a non-trivial prior state; other
  // slots (incl. dst) start from a DIFFERENT pattern so a missing write or a
  // wrong-slot write is detectable.
  torch::Tensor conv_cache_seed =
      test::seeded_tensor("conv.noov.cache",
                          {kNumSlots, kConvStateLen, conv_dim},
                          torch::kFloat32,
                          device)
          .to(torch::kBFloat16);

  const std::vector<int64_t> query_start_loc = {0, num_tokens};
  const std::vector<int64_t> cache_indices = {kSrcSlot};
  const std::vector<int64_t> has_initial_state = {1};

  // ----- Pass A: in-place (write == read == src) -----
  torch::Tensor cache_a = conv_cache_seed.clone();
  const torch::Tensor src_before = cache_a[kSrcSlot].clone();
  {
    const std::vector<int64_t> empty_out;  // no out_cache -> in-place
    auto [q, k, v] = xllm::kernel::npu::causal_conv1d_qkv(
        x,
        weight,
        cache_a,
        torch::IntArrayRef(query_start_loc),
        torch::IntArrayRef(cache_indices),
        torch::IntArrayRef(has_initial_state),
        torch::IntArrayRef(empty_out),
        kNumQkHeads,
        kNumVHeads,
        kHeadDim,
        kHeadDim);
    (void)q;
    (void)k;
    (void)v;
  }
  Device(device).synchronize_default_stream();
  const torch::Tensor inplace_result = cache_a[kSrcSlot].clone();

  // ----- Pass B: out-of-place (read src, write dst) -----
  torch::Tensor cache_b = conv_cache_seed.clone();
  {
    const std::vector<int64_t> out_cache = {kDstSlot};
    auto [q, k, v] = xllm::kernel::npu::causal_conv1d_qkv(
        x,
        weight,
        cache_b,
        torch::IntArrayRef(query_start_loc),
        torch::IntArrayRef(cache_indices),
        torch::IntArrayRef(has_initial_state),
        torch::IntArrayRef(out_cache),
        kNumQkHeads,
        kNumVHeads,
        kHeadDim,
        kHeadDim);
    (void)q;
    (void)k;
    (void)v;
  }
  Device(device).synchronize_default_stream();

  // dst after out-of-place must equal what in-place produced in src.
  test::verify_tensor_close(cache_b[kDstSlot].to(torch::kFloat32),
                            inplace_result.to(torch::kFloat32),
                            /*rtol=*/1e-2,
                            /*atol=*/1e-2);
  // src must be frozen: out-of-place never touched it, so it still holds the
  // pre-forward state (identical to what pass A read in).
  test::verify_tensor_close(cache_b[kSrcSlot].to(torch::kFloat32),
                            src_before.to(torch::kFloat32),
                            /*rtol=*/0.0,
                            /*atol=*/0.0);
}

}  // namespace layer
}  // namespace xllm
