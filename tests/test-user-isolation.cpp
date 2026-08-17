// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lithinnon / fewtarius
//
// Unit tests for Phase 2: Multi-Tenant User Isolation, user_id validation,
// slot affinity, and namespace segregation.

#undef NDEBUG

#include "server-task.h"
#include "server-common.h"
#include "server-context-page-manager.h"
#include "kv-ssd-cache.h"
#include "common.h"

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
        fflush(stdout);                             \
    } catch (const std::exception & e) {            \
        printf("FAIL: %s\n", e.what());             \
        tests_failed++;                             \
    }                                               \
} while (0)

static fs::path make_scratch(const std::string & name) {
    fs::path p = fs::temp_directory_path() / ("user_isolation_test_" + name);
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p, ec);
    if (ec) {
        throw std::runtime_error("could not create scratch dir " + p.string() + ": " + ec.message());
    }
    return p;
}

// 1. Validate user_id parsing and rejection
static void test_validate_user_id() {
    // Valid cases
    assert(server_task::validate_user_id("") == "");
    assert(server_task::validate_user_id("user123") == "user123");
    assert(server_task::validate_user_id("tenant-42_user-7") == "tenant-42_user-7");
    assert(server_task::validate_user_id("A-Z_a-z_0-9") == "A-Z_a-z_0-9");

    // Max length 512 is accepted
    std::string len512(512, 'a');
    assert(server_task::validate_user_id(len512) == len512);

    // Too long (513 chars) should throw
    std::string len513(513, 'a');
    bool caught_length = false;
    try {
        server_task::validate_user_id(len513);
    } catch (const std::invalid_argument &) {
        caught_length = true;
    }
    assert(caught_length);

    // Invalid characters should throw
    std::vector<std::string> invalid_ids = {
        "user 123",      // space
        "user@domain",   // @
        "user/slash",    // /
        "../traversal",  // .
        "user#tag",      // #
        "user:colon",    // :
    };

    for (const auto & invalid_id : invalid_ids) {
        bool caught = false;
        try {
            server_task::validate_user_id(invalid_id);
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        assert(caught && "expected invalid_argument for bad character");
    }
}

// 2. Test user namespace disk layout and cache isolation
static void test_user_cache_namespace() {
    fs::path root = make_scratch("namespace");

    llama::kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.hot_window_tokens = 16384;
    cfg.warm_window_tokens = 32768;

    llama::server_context_page_manager mgr(root.string().c_str(), &cfg, 16384, 16);

    std::vector<llama_token> tokens = {10, 20, 30, 40, 50, 60, 70, 80};
    common_prompt_checkpoint ckpt;
    ckpt.n_tokens = 8;
    ckpt.pos_min = 0;
    ckpt.pos_max = 7;
    ckpt.data_tgt.resize(64);

    // Storing under user-alice creates u/ namespace on disk
    mgr.store_checkpoint_with_tokens(
        0, nullptr, nullptr, ckpt,
        tokens.data(), tokens.size(),
        1, 0, "user-alice");

    // Verify disk namespace contains u/
    assert(fs::exists(root / "u"));
}

// 3. Test cross-user checkpoint isolation
static void test_cross_user_isolation() {
    fs::path root = make_scratch("cross_user");

    llama::kv_eviction_config cfg;
    cfg.page_size_tokens = 1024;
    cfg.hot_window_tokens = 16384;
    cfg.warm_window_tokens = 32768;

    llama::server_context_page_manager mgr(root.string().c_str(), &cfg, 16384, 16);

    std::vector<llama_token> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    const uint64_t conv_hash = 0x12345678ULL;

    // Store a checkpoint under User Alice
    common_prompt_checkpoint ckpt;
    ckpt.n_tokens = 8;
    ckpt.pos_min = 0;
    ckpt.pos_max = 7;
    ckpt.data_tgt.resize(64);

    mgr.store_checkpoint_with_tokens(
        0, nullptr, nullptr, ckpt,
        tokens.data(), tokens.size(),
        1, conv_hash, "user-alice");

    // Try finding checkpoint as User Bob (same tokens)
    int32_t pos_min = 0, pos_max = 0;
    uint64_t n_tokens = 0;
    int32_t lcp = 0;
    float overlap = 0.0f;
    bool is_continuation = false;

    bool found_bob = mgr.find_and_load_checkpoint(
        tokens.data(), tokens.size(),
        1, nullptr, nullptr, 0,
        pos_min, pos_max, n_tokens,
        nullptr, conv_hash, 0, tokens.size(),
        &lcp, &overlap, &is_continuation,
        "user-bob");

    assert(!found_bob && "User Bob must NOT see User Alice's checkpoint!");

    // Try finding checkpoint as User Alice (should succeed)
    bool found_alice = mgr.find_and_load_checkpoint(
        tokens.data(), tokens.size(),
        1, nullptr, nullptr, 0,
        pos_min, pos_max, n_tokens,
        nullptr, conv_hash, 0, tokens.size(),
        &lcp, &overlap, &is_continuation,
        "user-alice");

    assert(found_alice && "User Alice must find her own checkpoint");
    assert(n_tokens == 8);
}

// 4. Test error response formatting for rate limiting (HTTP 429)
static void test_rate_limit_error_formatting() {
    json err = format_error_response("user 'user-1' exceeded limit", ERROR_TYPE_RATE_LIMIT);
    assert(err["code"] == 429);
    assert(err["type"] == "rate_limit_error");
    assert(err["message"] == "user 'user-1' exceeded limit");
}

int main() {
    printf("=== Running User Isolation Tests ===\n");
    RUN(validate_user_id);
    RUN(user_cache_namespace);
    RUN(cross_user_isolation);
    RUN(rate_limit_error_formatting);

    printf("\nSummary: %d / %d tests passed.\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
