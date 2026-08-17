// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 fewtarius
//
// MoE expert residency tracker for CachyLLama / BeeLlama.cpp.
//
// Phase 1: tracks which MoE experts are "hot" per layer via a per-layer LRU.
// Uses madvise(MADV_WILLNEED / MADV_DONTNEED / MADV_FREE) on the existing mmap'd model
// regions to keep hot experts paged in and evict cold ones.

#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

struct ggml_tensor;

// Per-layer residency state for one MoE layer's expert tensors.
struct llama_moe_layer_residency_internal {
    // Model layer index (used to look up the corresponding expert_stats entry
    // in llama_context). May differ from the vector position in
    // llama_moe_residency_state::layers if the model has non-MoE layers.
    int model_layer = -1;

    // Pointers to the model tensors (gate / up / down, plus fused gate_up
    // for architectures that combine them, e.g. gemma4).
    ggml_tensor * t_gate    = nullptr;
    ggml_tensor * t_up      = nullptr;
    ggml_tensor * t_down    = nullptr;
    ggml_tensor * t_gate_up = nullptr;     // optional

    // Per-expert byte stride = tensor->nb[2]. All tensors in this layer
    // share the same per-expert count (= model n_expert) but may have
    // different strides due to layout (down is transposed).
    size_t gate_stride    = 0;
    size_t up_stride      = 0;
    size_t down_stride    = 0;
    size_t gate_up_stride = 0;

    // Number of experts in this layer (= model.hparams.n_expert).
    int n_expert = 0;

    // Recency+frequency cache.
    // Each entry tracks when it was last accessed and how many times.
    // Eviction score (lower = more evictable):
    //   0.5 * (1 / (1 + current_token - last_access)) +
    //   0.5 * (access_count / (1 + current_token - loaded_at))
    struct cache_entry {
        int      expert_id     = -1;
        uint64_t last_access   = 0;   // token counter at last touch
        uint64_t access_count  = 0;   // total touches since loaded
        uint64_t loaded_at     = 0;   // token counter when first loaded
        bool     occupied      = false;
    };
    std::vector<cache_entry> cache;            // size = max_resident_per_layer
    std::vector<int>         slot_of;          // size = n_expert
    uint64_t                 token_counter = 0;

    uint64_t hits   = 0;
    uint64_t misses = 0;
};

// Internal config (C++-side).
struct llama_moe_residency_internal_cfg {
    bool   enabled                = false;
    size_t max_resident_per_layer = 16;
    bool   prewarm_on_init        = true;
    int    prewarm_top_k          = 8;
    bool   log_per_decode         = true;
};

// Aggregate residency state. One entry per MoE layer.
struct llama_moe_residency_state {
    llama_moe_residency_internal_cfg cfg;
    std::vector<llama_moe_layer_residency_internal> layers;
    int n_layers = 0;
    int n_expert = 0;
    int n_expert_used = 0;

    uint64_t total_hits    = 0;
    uint64_t total_misses  = 0;
    uint64_t total_evicted = 0;
    uint64_t total_touched = 0;
    uint64_t decode_count  = 0;
};

bool llama_moe_residency_build(
        const struct llama_model * model,
        struct llama_moe_residency_internal_cfg cfg,
        struct llama_moe_residency_state * out);

void llama_moe_residency_prewarm(
        struct llama_moe_residency_state * st,
        const int * const * top_experts);

void llama_moe_residency_touch_layer_selection(
        struct llama_moe_residency_state * st,
        int model_layer,
        const int32_t * expert_ids,
        int n_expert_ids);

void llama_moe_residency_touch(
        struct llama_moe_residency_state * st,
        int layer_idx,
        int expert_id,
        bool * was_already_loaded);

void llama_moe_residency_release(
        struct llama_moe_residency_state * st);

void llama_moe_residency_log_stats(
        const struct llama_moe_residency_state * st);

bool llama_moe_residency_topk_from_stats(
        const struct llama_context * ctx,
        int k,
        std::vector<std::vector<int>> & out_top);
