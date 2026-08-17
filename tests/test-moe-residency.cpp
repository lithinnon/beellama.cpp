// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius

#undef NDEBUG

#include "llama.h"
#include "llama-moe-residency.h"
#include "llama-moe-coact.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int tests_run    = 0;
static int tests_failed = 0;

#define RUN(name) do {                              \
    printf("  %-50s ", #name);                      \
    fflush(stdout);                                 \
    tests_run++;                                    \
    try {                                           \
        test_##name();                              \
        printf("OK\n");                             \
    } catch (const std::exception & e) {            \
        printf("FAIL: %s\n", e.what());             \
        tests_failed++;                             \
    }                                               \
} while (0)

static void test_moe_coact_recording_and_predictions() {
    llama_moe_coact::matrix m;
    m.num_layers = 4;
    m.num_experts = 8;
    m.layer_pair_counts.assign(4, std::vector<uint32_t>(64, 0));
    m.cross_counts.assign(4, std::vector<std::vector<uint32_t>>(8, std::vector<uint32_t>(8, 0)));
    m.observation_counts.assign(4, 0);

    // Record co-activations on layer 0: experts {1, 3}
    int32_t sel_l0[2] = { 1, 3 };
    llama_moe_coact::record(m, 0, sel_l0, 2);

    // Record co-activations on layer 1: experts {2, 4}
    int32_t sel_l1[2] = { 2, 4 };
    llama_moe_coact::record(m, 1, sel_l1, 2);

    // Record cross-layer correlation 0 -> 1
    llama_moe_coact::record_cross_layer(m, 0, sel_l0, 2, sel_l1, 2);

    // Predict same-layer on layer 0 given {1}
    int32_t obs[1] = { 1 };
    std::vector<int32_t> same_pred = llama_moe_coact::predict_same_layer(m, 0, obs, 1, 2);
    assert(!same_pred.empty());
    assert(same_pred[0] == 3);

    // Predict next-layer on layer 0 -> layer 1 given {1}
    std::vector<int32_t> next_pred = llama_moe_coact::predict_next_layer(m, 0, obs, 1, 2);
    assert(next_pred.size() == 2);
    assert((next_pred[0] == 2 && next_pred[1] == 4) || (next_pred[0] == 4 && next_pred[1] == 2));
}

static void test_moe_coact_save_and_load() {
    llama_moe_coact::matrix m;
    m.num_layers = 2;
    m.num_experts = 4;
    m.layer_pair_counts.assign(2, std::vector<uint32_t>(16, 0));
    m.cross_counts.assign(2, std::vector<std::vector<uint32_t>>(4, std::vector<uint32_t>(4, 0)));
    m.observation_counts.assign(2, 0);

    int32_t sel[2] = { 0, 2 };
    for (int i = 0; i < 15; ++i) {
        llama_moe_coact::record(m, 0, sel, 2);
    }
    assert(m.has_data());

    fs::path temp_file = fs::temp_directory_path() / "test_coact_matrix.json";
    bool saved = llama_moe_coact::save(m, temp_file.string());
    assert(saved);

    llama_moe_coact::matrix m_loaded;
    bool loaded = llama_moe_coact::load(m_loaded, temp_file.string());
    assert(loaded);
    assert(m_loaded.num_layers == 2);
    assert(m_loaded.num_experts == 4);
    assert(m_loaded.observation_counts[0] == 15);
    assert(m_loaded.has_data());

    std::error_code ec;
    fs::remove(temp_file, ec);
}

static void test_moe_residency_lru_touch() {
    llama_moe_residency_state st;
    st.cfg.enabled = true;
    st.cfg.max_resident_per_layer = 3;
    st.cfg.prewarm_on_init = false;
    st.cfg.prewarm_top_k = 0;
    st.cfg.log_per_decode = false;

    llama_moe_layer_residency_internal lr;
    lr.model_layer = 0;
    lr.n_expert = 8;
    lr.cache.assign(3, llama_moe_layer_residency_internal::cache_entry{});
    lr.slot_of.assign(8, -1);

    st.layers.push_back(std::move(lr));
    st.n_layers = 1;
    st.n_expert = 8;
    st.n_expert_used = 2;

    bool was_loaded = false;

    // Touch expert 0 -> miss
    llama_moe_residency_touch(&st, 0, 0, &was_loaded);
    assert(!was_loaded);
    assert(st.total_misses == 1);
    assert(st.total_hits == 0);

    // Touch expert 0 again -> hit
    llama_moe_residency_touch(&st, 0, 0, &was_loaded);
    assert(was_loaded);
    assert(st.total_misses == 1);
    assert(st.total_hits == 1);

    // Touch expert 1, 2 -> misses, fills cache (capacity 3)
    llama_moe_residency_touch(&st, 0, 1, &was_loaded);
    assert(!was_loaded);
    llama_moe_residency_touch(&st, 0, 2, &was_loaded);
    assert(!was_loaded);
    assert(st.total_misses == 3);

    // Touch expert 3 -> evicts least recently / least frequently used
    llama_moe_residency_touch(&st, 0, 3, &was_loaded);
    assert(!was_loaded);
    assert(st.total_evicted == 1);
}

static void test_moe_residency_default_config() {
    struct llama_moe_residency_config cfg = llama_moe_residency_config_default();
    assert(cfg.enabled == 0);
    assert(cfg.max_resident_per_layer == 128);
    assert(cfg.prewarm_on_init == 1);
    assert(cfg.prewarm_top_k == 8);
    assert(cfg.log_per_decode == 1);
}

int main() {
    printf("running test-moe-residency...\n");

    RUN(moe_coact_recording_and_predictions);
    RUN(moe_coact_save_and_load);
    RUN(moe_residency_lru_touch);
    RUN(moe_residency_default_config);

    printf("\n%d/%d tests passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
