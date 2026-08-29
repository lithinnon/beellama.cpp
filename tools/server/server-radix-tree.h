#pragma once

#include "common.h"
#include "llama.h"
#include "server-task.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <list>
#include <functional>
#include <chrono>
#include <unordered_set>

enum server_radix_tier : uint8_t {
    RADIX_TIER_NONE,
    RADIX_TIER_RAM,
    RADIX_TIER_DISK,
    RADIX_TIER_EVICTED,
};

struct server_radix_node;

struct server_radix_edge {
    std::vector<llama_token> tokens;
    std::shared_ptr<server_radix_node> child;
};

struct server_radix_match_result {
    std::shared_ptr<server_radix_node> node;
    size_t lexical_tokens = 0;
    size_t restorable_tokens = 0;
    server_prompt_reuse_reason reason = SERVER_PROMPT_REUSE_NONE;
};

struct server_radix_node : public std::enable_shared_from_this<server_radix_node> {
    uint64_t id = 0;
    size_t token_depth = 0; // Total tokens from root
    server_radix_tier tier = RADIX_TIER_NONE;

    bool is_checkpoint = false;
    server_prompt prompt;
    server_prompt_data data;
    std::string disk_chunk_id; // Unique identifier for on-disk file

    size_t payload_bytes = 0;
    int64_t last_accessed_time = 0; // Epoch milliseconds
    uint64_t access_count = 0;

    std::weak_ptr<server_radix_node> parent;
    std::vector<server_radix_edge> children;

    size_t accounted_size() const;
    void touch();
};

class server_radix_tree {
public:
    server_radix_tree();
    ~server_radix_tree() = default;

    // Disallow copy, allow move
    server_radix_tree(const server_radix_tree &) = delete;
    server_radix_tree & operator=(const server_radix_tree &) = delete;
    server_radix_tree(server_radix_tree &&) noexcept = default;
    server_radix_tree & operator=(server_radix_tree &&) noexcept = default;

    // Insert a prompt and its state payload into the Radix tree
    std::shared_ptr<server_radix_node> insert(
            const server_prompt & prompt,
            server_prompt_data && data,
            int32_t alignment = 1);

    // Find the deepest matching node with restorable state for the requested tokens
    server_radix_match_result find_best_match(
            const server_tokens & requested,
            int32_t alignment = 1,
            size_t live_native_restorable = 0);

    // Get all checkpoint nodes sorted by LRU (oldest accessed first)
    std::vector<std::shared_ptr<server_radix_node>> get_lru_nodes(server_radix_tier filter_tier = RADIX_TIER_NONE) const;

    // Evict node data from RAM
    void evict_ram_payload(const std::shared_ptr<server_radix_node> & node);

    // Remove a node entirely from tree
    bool remove_node(const std::shared_ptr<server_radix_node> & node);

    // Clear all nodes
    void clear();

    // Total deduplicated payload size in RAM (bytes)
    size_t accounted_ram_bytes() const;

    // Total accounted size on Disk (bytes)
    size_t accounted_disk_bytes() const;

    size_t total_nodes() const;
    size_t total_checkpoints() const;
    size_t total_tokens() const;

    const std::shared_ptr<server_radix_node> & get_root() const { return root; }

private:
    uint64_t next_node_id = 1;
    std::shared_ptr<server_radix_node> root;

    void collect_all_nodes(
            const std::shared_ptr<server_radix_node> & node,
            std::vector<std::shared_ptr<server_radix_node>> & out) const;

    void prune_empty_leaves(std::shared_ptr<server_radix_node> & node);
};
