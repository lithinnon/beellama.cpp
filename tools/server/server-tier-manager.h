#pragma once

#include "server-radix-tree.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

class server_tier_manager {
public:
    server_tier_manager();
    ~server_tier_manager() = default;

    // Initialize the disk tier with base directory and quota
    bool init(const std::string & base_dir, int32_t quota_mib);

    // Save a node's in-memory state to an on-disk .ckpt chunk atomically
    bool save_to_disk(const std::shared_ptr<server_radix_node> & node, server_radix_tree & tree);

    // Load a node's on-disk .ckpt state back into RAM
    bool load_from_disk(const std::shared_ptr<server_radix_node> & node);

    // Enforce RAM limit by spilling cold nodes to disk or evicting
    void enforce_ram_limit(server_radix_tree & tree, size_t limit_ram_bytes);

    // Enforce Disk quota by removing coldest disk chunks
    void enforce_disk_limit(server_radix_tree & tree);

    // Prune leftover .tmp files from previous crashes
    void cleanup_temp_files();

    const std::string & get_disk_dir() const { return disk_dir; }
    int32_t get_quota_mib() const { return quota_mib; }
    bool is_disk_enabled() const { return quota_mib != 0; }

private:
    std::string disk_dir;
    int32_t quota_mib = 0;
    size_t disk_quota_bytes = 0; // 0 = disabled, (size_t)-1 = unlimited

    mutable std::mutex tier_mutex;

    // Helper to calculate simple 32-bit checksum
    uint32_t compute_checksum(const uint8_t * data, size_t len) const;
};
