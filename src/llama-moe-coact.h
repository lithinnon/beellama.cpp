// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 fewtarius
//
// Co-activation matrix for MoE expert selection prediction.
//
// Tracks which experts tend to fire together within a layer and across
// adjacent layers. After enough observation, the matrix can predict which
// experts are likely to fire next given the currently-selected experts.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

namespace llama_moe_coact {

struct matrix {
    std::vector<std::vector<uint32_t>> layer_pair_counts;
    std::vector<std::vector<std::vector<uint32_t>>> cross_counts;
    std::vector<uint32_t> observation_counts;

    int num_layers = 0;
    int num_experts = 0;

    bool has_data() const {
        for (auto c : observation_counts) {
            if (c >= 10) return true;
        }
        return false;
    }
};

void init(matrix & m, const struct llama_model & model);

void record(matrix & m,
            int layer,
            const int32_t * selected_experts,
            int n_selected);

void record_cross_layer(matrix & m,
                        int layer,
                        const int32_t * selected_n,
                        int n_n,
                        const int32_t * selected_n1,
                        int n_n1);

std::vector<int32_t> predict_same_layer(
        const matrix & m,
        int layer,
        const int32_t * observed,
        int n_observed,
        int top_k);

std::vector<int32_t> predict_next_layer(
        const matrix & m,
        int layer,
        const int32_t * observed,
        int n_observed,
        int top_k);

std::string persistence_path(const std::string & model_path);

bool save(const matrix & m, const std::string & path);

bool load(matrix & m, const std::string & path);

} // namespace llama_moe_coact
