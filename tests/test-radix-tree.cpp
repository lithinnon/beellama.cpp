#include "../tools/server/server-radix-tree.h"
#include "../tools/server/server-tier-manager.h"
#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

static server_prompt make_test_prompt(const llama_tokens & tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(tokens, false);
    return prompt;
}

static server_prompt_data make_test_data(size_t main_size, size_t drft_size = 0) {
    server_prompt_data data;
    data.main.resize(main_size);
    for (size_t i = 0; i < main_size; ++i) {
        data.main[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
    }
    if (drft_size > 0) {
        data.drft.resize(drft_size);
        for (size_t i = 0; i < drft_size; ++i) {
            data.drft[i] = static_cast<uint8_t>((i * 17 + 5) & 0xFF);
        }
    }
    return data;
}

static void test_radix_tree_basic_insertion_and_search() {
    server_radix_tree tree;

    auto p1 = make_test_prompt({10, 20, 30, 40, 50});
    auto d1 = make_test_data(64);
    auto n1 = tree.insert(p1, std::move(d1));
    assert(n1 != nullptr);
    assert(n1->token_depth == 5);
    assert(tree.total_checkpoints() == 1);

    // Exact search
    server_tokens req1(llama_tokens{10, 20, 30, 40, 50}, false);
    auto res1 = tree.find_best_match(req1, 1, 0);
    assert(res1.node == n1);
    assert(res1.restorable_tokens == 5);
    assert(res1.lexical_tokens == 5);

    // Prefix search
    server_tokens req2(llama_tokens{10, 20, 30, 40, 50, 60, 70}, false);
    auto res2 = tree.find_best_match(req2, 1, 0);
    assert(res2.node == n1);
    assert(res2.restorable_tokens == 5);
    assert(res2.lexical_tokens == 5);
}

static void test_radix_tree_branching_and_splitting() {
    server_radix_tree tree;

    // Insert Branch A: [1, 2, 3, 4, 5]
    auto p1 = make_test_prompt({1, 2, 3, 4, 5});
    auto d1 = make_test_data(128);
    auto n1 = tree.insert(p1, std::move(d1));
    assert(n1 != nullptr);

    // Insert Branch B sharing prefix [1, 2, 3] -> [1, 2, 3, 6, 7]
    auto p2 = make_test_prompt({1, 2, 3, 6, 7});
    auto d2 = make_test_data(128);
    auto n2 = tree.insert(p2, std::move(d2));
    assert(n2 != nullptr);

    assert(tree.total_checkpoints() == 2);

    // Query matching Branch A
    server_tokens req_a(llama_tokens{1, 2, 3, 4, 5, 99}, false);
    auto res_a = tree.find_best_match(req_a, 1, 0);
    assert(res_a.node == n1);
    assert(res_a.restorable_tokens == 5);

    // Query matching Branch B
    server_tokens req_b(llama_tokens{1, 2, 3, 6, 7, 100}, false);
    auto res_b = tree.find_best_match(req_b, 1, 0);
    assert(res_b.node == n2);
    assert(res_b.restorable_tokens == 5);

    // Query matching only shared prefix
    server_tokens req_c(llama_tokens{1, 2, 3, 8, 9}, false);
    auto res_c = tree.find_best_match(req_c, 1, 0);
    // Both branches diverge after token 3; neither has a checkpoint at 3 unless checkpoints exist
    assert(res_c.lexical_tokens == 3);
}

static void test_radix_tree_kvarn_alignment() {
    server_radix_tree tree;

    // 256 tokens prompt with intermediate checkpoint at 128
    llama_tokens tokens(256, 42);
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<llama_token>(i + 1);

    auto p = make_test_prompt(tokens);
    common_prompt_checkpoint ckpt;
    ckpt.n_tokens = 128;
    ckpt.pos_min = 0;
    ckpt.pos_max = 128;
    ckpt.data_tgt.resize(64);
    p.checkpoints.push_back(std::move(ckpt));

    auto node = tree.insert(p, make_test_data(512));
    assert(node != nullptr);

    // Request prefix with 200 tokens under alignment 128
    llama_tokens req_tokens(tokens.begin(), tokens.begin() + 200);
    server_tokens req(req_tokens, false);

    auto res = tree.find_best_match(req, 128, 0);
    assert(res.node == node);
    assert(res.restorable_tokens == 128); // Snapped to 128 boundary
}

static void test_tier_manager_atomic_disk_save_load() {
    const std::string test_dir = "/tmp/beellama_radix_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

    server_radix_tree tree;
    server_tier_manager tier_mgr;
    bool init_ok = tier_mgr.init(test_dir, 64); // 64 MiB
    assert(init_ok);

    auto p = make_test_prompt({100, 200, 300, 400});
    auto original_data = make_test_data(256, 128);
    auto insert_data = original_data;
    auto node = tree.insert(p, std::move(insert_data));
    assert(node != nullptr);
    assert(node->tier == RADIX_TIER_RAM);

    // Save to disk
    bool saved = tier_mgr.save_to_disk(node, tree);
    assert(saved);
    assert(node->tier == RADIX_TIER_DISK);
    assert(!node->disk_chunk_id.empty());
    assert(fs::exists(node->disk_chunk_id));
    assert(node->data.main.empty()); // RAM payload evicted

    // Load back from disk
    bool loaded = tier_mgr.load_from_disk(node);
    assert(loaded);
    assert(node->tier == RADIX_TIER_RAM);
    assert(node->data.main.size() == original_data.main.size());
    assert(node->data.main == original_data.main);
    assert(node->data.drft == original_data.drft);

    // Clean up
    fs::remove_all(test_dir);
}

static void test_tier_manager_quota_eviction() {
    const std::string test_dir = "/tmp/beellama_radix_quota_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

    server_radix_tree tree;
    server_tier_manager tier_mgr;
    tier_mgr.init(test_dir, 1); // 1 MiB quota

    // Insert and spill multiple nodes
    for (int i = 0; i < 5; ++i) {
        auto p = make_test_prompt({static_cast<llama_token>(i * 10), 1, 2, 3});
        auto d = make_test_data(300 * 1024); // 300 KiB each
        auto node = tree.insert(p, std::move(d));
        tier_mgr.save_to_disk(node, tree);
    }

    // Accounted disk size must be <= 1 MiB
    assert(tree.accounted_disk_bytes() <= 1024 * 1024);

    fs::remove_all(test_dir);
}

static void test_radix_cache_hits_and_misses() {
    const std::string test_dir = "/tmp/beellama_radix_hitmiss_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    server_prompt_cache cache(8, 0, true, 16, test_dir); // 8 MiB RAM, 16 MiB disk

    // 1. Insert Prompt A: [10, 20, 30, 40]
    auto pA = make_test_prompt({10, 20, 30, 40});
    auto * admittedA = cache.insert(pA, make_test_data(1024, 512));
    assert(admittedA != nullptr);

    // 2. Test Hit on exact prefix
    server_prompt dstA;
    server_tokens reqA(llama_tokens{10, 20, 30, 40, 50, 60}, false);
    size_t restored_main_len = 0;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ true,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t *, size_t main_sz,
                                       const uint8_t *, size_t drft_sz,
                                       const uint8_t *, size_t) {
            restored_main_len = main_sz;
            return main_sz == 1024 && drft_sz == 512;
        }
    };
    assert(cache.load(dstA, reqA, 0, 1, io));
    assert(restored_main_len == 1024);
    assert(dstA.tokens.size() == 4);
    assert(cache.radix_hits_ram == 1);

    // 3. Test Miss on completely unrelated prefix: [999, 888, 777]
    server_prompt dstMiss;
    server_tokens reqMiss(llama_tokens{999, 888, 777}, false);
    restored_main_len = 0;
    assert(cache.load(dstMiss, reqMiss, 0, 1, io));
    assert(restored_main_len == 0); // No restore callback triggered
    assert(dstMiss.tokens.empty());

    // Clean up
    fs::remove_all(test_dir);
}

static void test_radix_cache_quota_and_eviction_lifecycle() {
    server_radix_tree tree;
    server_tier_manager tier_mgr;
    const std::string test_dir = "/tmp/beellama_radix_lifecycle_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    
    // 1 MiB disk limit
    assert(tier_mgr.init(test_dir, 1));

    std::vector<std::shared_ptr<server_radix_node>> nodes;
    for (int i = 0; i < 10; ++i) {
        auto p = make_test_prompt({static_cast<llama_token>(i + 100), 1, 2, 3});
        auto node = tree.insert(p, make_test_data(256 * 1024)); // 256 KiB
        assert(node != nullptr);
        nodes.push_back(node);
    }

    // Accounted RAM before enforcement = 10 * 256 KiB = 2.5 MiB
    assert(tree.accounted_ram_bytes() >= 2500 * 1024);

    // Enforce 512 KiB RAM limit -> older nodes spill to disk
    tier_mgr.enforce_ram_limit(tree, 512 * 1024);
    assert(tree.accounted_ram_bytes() <= 512 * 1024);
    assert(tree.accounted_disk_bytes() > 0);

    // Enforce 1 MiB Disk limit -> coldest spilled nodes are pruned if exceeding limit
    tier_mgr.enforce_disk_limit(tree);
    assert(tree.accounted_disk_bytes() <= 1024 * 1024);

    // Verify evicted nodes have their memory buffers cleared
    size_t evicted_count = 0;
    for (auto & node : nodes) {
        if (node->tier == RADIX_TIER_EVICTED) {
            ++evicted_count;
            assert(node->data.main.empty());
            assert(node->data.drft.empty());
            assert(node->disk_chunk_id.empty() || !fs::exists(node->disk_chunk_id));
        }
    }
    assert(evicted_count > 0);

    fs::remove_all(test_dir);
}

static void test_radix_cache_recurrent_and_multiturn_state() {
    server_radix_tree tree;

    // Turn 1: User: "Hello" -> [1, 2, 3] + Assistant: "Hi" -> [4, 5]
    llama_tokens turn1_tokens = {1, 2, 3, 4, 5};
    auto p1 = make_test_prompt(turn1_tokens);
    common_prompt_checkpoint ckpt1;
    ckpt1.n_tokens = 5;
    ckpt1.pos_min = 0;
    ckpt1.pos_max = 5;
    ckpt1.data_tgt.resize(128, 0x11);
    p1.checkpoints.push_back(ckpt1);
    auto n1 = tree.insert(p1, make_test_data(256));
    assert(n1 != nullptr);

    // Turn 2: Followup: "How are you?" -> [6, 7] + Assistant: "Great" -> [8, 9]
    llama_tokens turn2_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto p2 = make_test_prompt(turn2_tokens);
    common_prompt_checkpoint ckpt2;
    ckpt2.n_tokens = 9;
    ckpt2.pos_min = 5;
    ckpt2.pos_max = 9;
    ckpt2.data_tgt.resize(128, 0x22);
    p2.checkpoints.push_back(ckpt1);
    p2.checkpoints.push_back(ckpt2);
    auto n2 = tree.insert(p2, make_test_data(512));
    assert(n2 != nullptr);

    // Branch Turn 2B: Alternative question: "What is 2+2?" -> [100, 101]
    llama_tokens turn2b_tokens = {1, 2, 3, 4, 5, 100, 101};
    auto p2b = make_test_prompt(turn2b_tokens);
    auto n2b = tree.insert(p2b, make_test_data(384));
    assert(n2b != nullptr);

    // Query Turn 2 request -> must match Turn 2 node (9 tokens)
    server_tokens req_turn2(turn2_tokens, false);
    auto res2 = tree.find_best_match(req_turn2, 1, 0);
    assert(res2.node == n2);
    assert(res2.restorable_tokens == 9);

    // Query Branch 2B request -> must match Branch 2B node
    server_tokens req_turn2b(turn2b_tokens, false);
    auto res2b = tree.find_best_match(req_turn2b, 1, 0);
    assert(res2b.node == n2b);
    assert(res2b.restorable_tokens == 7);

    // Query Branch 2C (new question from Turn 1 prefix) -> should reuse Turn 1 prefix (5 tokens)
    llama_tokens turn2c_tokens = {1, 2, 3, 4, 5, 200, 201};
    server_tokens req_turn2c(turn2c_tokens, false);
    auto res2c = tree.find_best_match(req_turn2c, 1, 0);
    assert(res2c.node == n1);
    assert(res2c.restorable_tokens == 5);
}

static void test_multimodel_cache_isolation_and_quotas() {
    std::string base_dir = "/tmp/beellama_radix_multimodel_test_" +
                           std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::string dir_a = base_dir + "/gemma4_26b_q4km_k-bf16_v-bf16";
    std::string dir_b = base_dir + "/qwen2.5_7b_q4km_k-q4_0_v-q4_0";

    server_radix_tree tree_a;
    server_tier_manager mgr_a;
    assert(mgr_a.init(dir_a, 2)); // 2 MiB quota for Model A

    server_radix_tree tree_b;
    server_tier_manager mgr_b;
    assert(mgr_b.init(dir_b, 2)); // 2 MiB quota for Model B

    // 1. Insert 3 separate 1MB chunks into Model A
    llama_tokens tokens_a1 = {10, 11, 12, 13};
    llama_tokens tokens_a2 = {10, 11, 12, 14};
    llama_tokens tokens_a3 = {10, 11, 12, 15};

    auto n_a1 = tree_a.insert(make_test_prompt(tokens_a1), make_test_data(1024 * 1024));
    assert(mgr_a.save_to_disk(n_a1, tree_a));

    auto n_a2 = tree_a.insert(make_test_prompt(tokens_a2), make_test_data(1024 * 1024));
    assert(mgr_a.save_to_disk(n_a2, tree_a));

    auto n_a3 = tree_a.insert(make_test_prompt(tokens_a3), make_test_data(1024 * 1024));
    assert(mgr_a.save_to_disk(n_a3, tree_a)); // Total 3 MB > 2 MB quota -> evicts oldest (n_a1)

    // Verify Model A directory state
    assert(n_a1->tier == RADIX_TIER_EVICTED);
    assert(n_a2->tier == RADIX_TIER_DISK);
    assert(n_a3->tier == RADIX_TIER_DISK);
    assert(tree_a.accounted_disk_bytes() <= 2 * 1024 * 1024);

    // 2. Insert 3 separate 1MB chunks into Model B
    llama_tokens tokens_b1 = {100, 101, 102, 103};
    llama_tokens tokens_b2 = {100, 101, 102, 104};
    llama_tokens tokens_b3 = {100, 101, 102, 105};

    auto n_b1 = tree_b.insert(make_test_prompt(tokens_b1), make_test_data(1024 * 1024));
    assert(mgr_b.save_to_disk(n_b1, tree_b));

    auto n_b2 = tree_b.insert(make_test_prompt(tokens_b2), make_test_data(1024 * 1024));
    assert(mgr_b.save_to_disk(n_b2, tree_b));

    auto n_b3 = tree_b.insert(make_test_prompt(tokens_b3), make_test_data(1024 * 1024));
    assert(mgr_b.save_to_disk(n_b3, tree_b)); // Total 3 MB > 2 MB quota -> evicts oldest (n_b1)

    // Verify Model B directory state
    assert(n_b1->tier == RADIX_TIER_EVICTED);
    assert(n_b2->tier == RADIX_TIER_DISK);
    assert(n_b3->tier == RADIX_TIER_DISK);
    assert(tree_b.accounted_disk_bytes() <= 2 * 1024 * 1024);

    // 3. Verify Model A's files in dir_a were completely untouched by Model B
    assert(n_a2->tier == RADIX_TIER_DISK);
    assert(n_a3->tier == RADIX_TIER_DISK);
    assert(fs::exists(n_a2->disk_chunk_id));
    assert(fs::exists(n_a3->disk_chunk_id));
    assert(mgr_a.load_from_disk(n_a2));
    assert(n_a2->tier == RADIX_TIER_RAM);

    // 4. Verify Model B files in dir_b load cleanly into Model B
    assert(fs::exists(n_b2->disk_chunk_id));
    assert(fs::exists(n_b3->disk_chunk_id));
    assert(mgr_b.load_from_disk(n_b2));
    assert(n_b2->tier == RADIX_TIER_RAM);

    // 5. Verify cross-model prefix query isolation
    server_tokens req_b(tokens_b2, false);
    auto res_a = tree_a.find_best_match(req_b, 1, 0);
    assert(res_a.restorable_tokens == 0); // Model A finds 0 match for Model B tokens

    server_tokens req_a(tokens_a2, false);
    auto res_b = tree_b.find_best_match(req_a, 1, 0);
    assert(res_b.restorable_tokens == 0); // Model B finds 0 match for Model A tokens

    // Cleanup
    std::filesystem::remove_all(base_dir);
}

int main() {
    std::cout << "[test-radix-tree] Running tests...\n";

    test_radix_tree_basic_insertion_and_search();
    std::cout << " - test_radix_tree_basic_insertion_and_search: PASSED\n";

    test_radix_tree_branching_and_splitting();
    std::cout << " - test_radix_tree_branching_and_splitting: PASSED\n";

    test_radix_tree_kvarn_alignment();
    std::cout << " - test_radix_tree_kvarn_alignment: PASSED\n";

    test_tier_manager_atomic_disk_save_load();
    std::cout << " - test_tier_manager_atomic_disk_save_load: PASSED\n";

    test_tier_manager_quota_eviction();
    std::cout << " - test_tier_manager_quota_eviction: PASSED\n";

    test_radix_cache_hits_and_misses();
    std::cout << " - test_radix_cache_hits_and_misses: PASSED\n";

    test_radix_cache_quota_and_eviction_lifecycle();
    std::cout << " - test_radix_cache_quota_and_eviction_lifecycle: PASSED\n";

    test_radix_cache_recurrent_and_multiturn_state();
    std::cout << " - test_radix_cache_recurrent_and_multiturn_state: PASSED\n";

    test_multimodel_cache_isolation_and_quotas();
    std::cout << " - test_multimodel_cache_isolation_and_quotas: PASSED\n";

    std::cout << "[test-radix-tree] All Radix Cache tests passed successfully!\n";
    return 0;
}
