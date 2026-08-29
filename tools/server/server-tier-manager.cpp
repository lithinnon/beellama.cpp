#include "server-tier-manager.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cassert>

namespace fs = std::filesystem;

static constexpr uint32_t RADIX_CHUNK_MAGIC = 0x42454552; // 'BEER'
static constexpr uint32_t RADIX_CHUNK_VERSION = 1;

#pragma pack(push, 1)
struct radix_chunk_header {
    uint32_t magic;
    uint32_t version;
    uint64_t node_id;
    uint32_t n_tokens;
    uint64_t size_main;
    uint64_t size_drft;
    uint64_t size_spec;
    uint32_t n_checkpoints;
    uint32_t checksum;
};
#pragma pack(pop)

server_tier_manager::server_tier_manager() = default;

uint32_t server_tier_manager::compute_checksum(const uint8_t * data, size_t len) const {
    // 32-bit FNV-1a hash
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash;
}

bool server_tier_manager::init(const std::string & base_dir, int32_t quota_mib_in) {
    std::lock_guard<std::mutex> lock(tier_mutex);
    disk_dir = base_dir;
    quota_mib = quota_mib_in;

    if (quota_mib < 0) {
        disk_quota_bytes = (size_t) -1; // Unlimited
    } else if (quota_mib > 0) {
        disk_quota_bytes = static_cast<size_t>(quota_mib) * 1024ull * 1024ull;
    } else {
        disk_quota_bytes = 0; // Disabled
        return true;
    }

    try {
        if (!fs::exists(disk_dir)) {
            fs::create_directories(disk_dir);
        }
    } catch (const std::exception & e) {
        SRV_ERR("failed to create radix cache disk directory %s: %s\n", disk_dir.c_str(), e.what());
        disk_quota_bytes = 0;
        return false;
    }

    cleanup_temp_files();
    return true;
}

void server_tier_manager::cleanup_temp_files() {
    if (disk_dir.empty() || !fs::exists(disk_dir)) {
        return;
    }

    try {
        for (const auto & entry : fs::directory_iterator(disk_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".tmp") {
                SRV_TRC("removing orphaned radix cache tmp file: %s\n", entry.path().string().c_str());
                fs::remove(entry.path());
            }
        }
    } catch (const std::exception & e) {
        SRV_WRN("warning during radix tmp cleanup: %s\n", e.what());
    }
}

bool server_tier_manager::save_to_disk(const std::shared_ptr<server_radix_node> & node, server_radix_tree & tree) {
    if (!is_disk_enabled() || !node || node->tier != RADIX_TIER_RAM) {
        return false;
    }

    std::lock_guard<std::mutex> lock(tier_mutex);

    // Prepare paths
    std::string tmp_filename = disk_dir + "/chunk_" + std::to_string(node->id) + "_" +
                               std::to_string(node->last_accessed_time) + ".tmp";
    std::string ckpt_filename = disk_dir + "/chunk_" + std::to_string(node->id) + "_" +
                                std::to_string(node->last_accessed_time) + ".ckpt";

    radix_chunk_header header {};
    header.magic = RADIX_CHUNK_MAGIC;
    header.version = RADIX_CHUNK_VERSION;
    header.node_id = node->id;
    header.n_tokens = static_cast<uint32_t>(node->prompt.tokens.size());
    header.size_main = node->data.main.size();
    header.size_drft = node->data.drft.size();
    header.size_spec = node->data.spec.size();
    header.n_checkpoints = static_cast<uint32_t>(node->prompt.checkpoints.size());

    // Calculate checksum of main + drft + spec payloads
    uint32_t csum = 0;
    if (!node->data.main.empty()) {
        csum ^= compute_checksum(node->data.main.data(), node->data.main.size());
    }
    if (!node->data.drft.empty()) {
        csum ^= compute_checksum(node->data.drft.data(), node->data.drft.size());
    }
    if (!node->data.spec.empty()) {
        csum ^= compute_checksum(node->data.spec.data(), node->data.spec.size());
    }
    header.checksum = csum;

    try {
        std::ofstream out(tmp_filename, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            SRV_ERR("failed to open tmp radix file for writing: %s\n", tmp_filename.c_str());
            return false;
        }

        // Write header
        out.write(reinterpret_cast<const char *>(&header), sizeof(header));

        // Write main, drft, spec data
        if (header.size_main > 0) {
            out.write(reinterpret_cast<const char *>(node->data.main.data()), header.size_main);
        }
        if (header.size_drft > 0) {
            out.write(reinterpret_cast<const char *>(node->data.drft.data()), header.size_drft);
        }
        if (header.size_spec > 0) {
            out.write(reinterpret_cast<const char *>(node->data.spec.data()), header.size_spec);
        }

        // Write checkpoints
        for (const auto & ckpt : node->prompt.checkpoints) {
            int64_t n_tokens_ckpt = ckpt.n_tokens;
            llama_pos pos_min = ckpt.pos_min;
            llama_pos pos_max = ckpt.pos_max;
            uint64_t size_tgt = ckpt.data_tgt.size();
            uint64_t size_dft = ckpt.data_dft.size();
            uint64_t size_spc = ckpt.data_spec.size();

            out.write(reinterpret_cast<const char *>(&n_tokens_ckpt), sizeof(n_tokens_ckpt));
            out.write(reinterpret_cast<const char *>(&pos_min), sizeof(pos_min));
            out.write(reinterpret_cast<const char *>(&pos_max), sizeof(pos_max));
            out.write(reinterpret_cast<const char *>(&size_tgt), sizeof(size_tgt));
            out.write(reinterpret_cast<const char *>(&size_dft), sizeof(size_dft));
            out.write(reinterpret_cast<const char *>(&size_spc), sizeof(size_spc));

            if (size_tgt > 0) {
                out.write(reinterpret_cast<const char *>(ckpt.data_tgt.data()), size_tgt);
            }
            if (size_dft > 0) {
                out.write(reinterpret_cast<const char *>(ckpt.data_dft.data()), size_dft);
            }
            if (size_spc > 0) {
                out.write(reinterpret_cast<const char *>(ckpt.data_spec.data()), size_spc);
            }
        }

        out.flush();
        out.close();

        // Atomic rename
        fs::rename(tmp_filename, ckpt_filename);
    } catch (const std::exception & e) {
        SRV_ERR("exception writing radix chunk to disk: %s\n", e.what());
        if (fs::exists(tmp_filename)) {
            fs::remove(tmp_filename);
        }
        return false;
    }

    node->disk_chunk_id = ckpt_filename;
    tree.evict_ram_payload(node);

    // Enforce disk quota
    enforce_disk_limit(tree);
    return true;
}

bool server_tier_manager::load_from_disk(const std::shared_ptr<server_radix_node> & node) {
    if (!node || node->disk_chunk_id.empty() || !fs::exists(node->disk_chunk_id)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(tier_mutex);

    try {
        std::ifstream in(node->disk_chunk_id, std::ios::binary);
        if (!in.is_open()) {
            SRV_ERR("failed to open radix chunk file: %s\n", node->disk_chunk_id.c_str());
            return false;
        }

        radix_chunk_header header {};
        in.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (header.magic != RADIX_CHUNK_MAGIC || header.version != RADIX_CHUNK_VERSION) {
            SRV_ERR("corrupted radix chunk header in %s\n", node->disk_chunk_id.c_str());
            return false;
        }

        server_prompt_data loaded_data;
        if (header.size_main > 0) {
            loaded_data.main.resize(header.size_main);
            in.read(reinterpret_cast<char *>(loaded_data.main.data()), header.size_main);
        }
        if (header.size_drft > 0) {
            loaded_data.drft.resize(header.size_drft);
            in.read(reinterpret_cast<char *>(loaded_data.drft.data()), header.size_drft);
        }
        if (header.size_spec > 0) {
            loaded_data.spec.resize(header.size_spec);
            in.read(reinterpret_cast<char *>(loaded_data.spec.data()), header.size_spec);
        }

        // Verify checksum
        uint32_t csum = 0;
        if (!loaded_data.main.empty()) {
            csum ^= compute_checksum(loaded_data.main.data(), loaded_data.main.size());
        }
        if (!loaded_data.drft.empty()) {
            csum ^= compute_checksum(loaded_data.drft.data(), loaded_data.drft.size());
        }
        if (!loaded_data.spec.empty()) {
            csum ^= compute_checksum(loaded_data.spec.data(), loaded_data.spec.size());
        }

        if (csum != header.checksum) {
            SRV_ERR("radix chunk checksum mismatch in %s\n", node->disk_chunk_id.c_str());
            return false;
        }

        // Read checkpoints into local temporary list to guarantee atomic metadata commit
        std::list<common_prompt_checkpoint> loaded_ckpts;
        for (uint32_t i = 0; i < header.n_checkpoints; ++i) {
            int64_t n_tokens_ckpt = 0;
            llama_pos pos_min = 0;
            llama_pos pos_max = 0;
            uint64_t size_tgt = 0;
            uint64_t size_dft = 0;
            uint64_t size_spc = 0;

            in.read(reinterpret_cast<char *>(&n_tokens_ckpt), sizeof(n_tokens_ckpt));
            in.read(reinterpret_cast<char *>(&pos_min), sizeof(pos_min));
            in.read(reinterpret_cast<char *>(&pos_max), sizeof(pos_max));
            in.read(reinterpret_cast<char *>(&size_tgt), sizeof(size_tgt));
            in.read(reinterpret_cast<char *>(&size_dft), sizeof(size_dft));
            in.read(reinterpret_cast<char *>(&size_spc), sizeof(size_spc));

            common_prompt_checkpoint ckpt;
            ckpt.n_tokens = n_tokens_ckpt;
            ckpt.pos_min = pos_min;
            ckpt.pos_max = pos_max;

            if (size_tgt > 0) {
                ckpt.data_tgt.resize(size_tgt);
                in.read(reinterpret_cast<char *>(ckpt.data_tgt.data()), size_tgt);
            }
            if (size_dft > 0) {
                ckpt.data_dft.resize(size_dft);
                in.read(reinterpret_cast<char *>(ckpt.data_dft.data()), size_dft);
            }
            if (size_spc > 0) {
                ckpt.data_spec.resize(size_spc);
                in.read(reinterpret_cast<char *>(ckpt.data_spec.data()), size_spc);
            }

            loaded_ckpts.push_back(std::move(ckpt));
        }

        node->prompt.checkpoints = std::move(loaded_ckpts);
        node->data = std::move(loaded_data);
        node->tier = RADIX_TIER_RAM;
        node->touch();
        return true;
    } catch (const std::exception & e) {
        SRV_ERR("exception reading radix chunk %s: %s\n", node->disk_chunk_id.c_str(), e.what());
        return false;
    }
}

void server_tier_manager::enforce_ram_limit(server_radix_tree & tree, size_t limit_ram_bytes) {
    if (limit_ram_bytes == (size_t) -1) {
        return;
    }

    while (tree.accounted_ram_bytes() > limit_ram_bytes) {
        auto lru_nodes = tree.get_lru_nodes(RADIX_TIER_RAM);
        if (lru_nodes.empty()) {
            break;
        }

        auto & oldest = lru_nodes.front();
        if (is_disk_enabled()) {
            SRV_TRC("spilling cold radix node %lu to disk (%.3f MiB)\n",
                    (unsigned long)oldest->id, oldest->accounted_size() / (1024.0 * 1024.0));
            if (!save_to_disk(oldest, tree)) {
                // If save failed, evict directly
                tree.evict_ram_payload(oldest);
            }
        } else {
            SRV_TRC("evicting cold radix node %lu from RAM (%.3f MiB)\n",
                    (unsigned long)oldest->id, oldest->accounted_size() / (1024.0 * 1024.0));
            tree.evict_ram_payload(oldest);
            if (oldest->children.empty()) {
                tree.remove_node(oldest);
            }
        }
    }
}

static size_t compute_directory_ckpt_bytes(const std::string & path) {
    size_t total = 0;
    try {
        if (fs::exists(path)) {
            for (const auto & entry : fs::recursive_directory_iterator(path, fs::directory_options::skip_permission_denied)) {
                if (entry.is_regular_file() && entry.path().extension() == ".ckpt") {
                    std::error_code ec;
                    total += entry.file_size(ec);
                }
            }
        }
    } catch (...) {}
    return total;
}

void server_tier_manager::enforce_disk_limit(server_radix_tree & tree) {
    if (disk_quota_bytes == 0 || disk_quota_bytes == (size_t) -1) {
        return;
    }

    // 1. Enforce local tree disk limit
    while (tree.accounted_disk_bytes() > disk_quota_bytes) {
        auto disk_nodes = tree.get_lru_nodes(RADIX_TIER_DISK);
        if (disk_nodes.empty()) {
            break;
        }

        auto & oldest = disk_nodes.front();
        SRV_INF("pruning oldest disk chunk '%s' (node %lu) to satisfy disk quota\n",
                oldest->disk_chunk_id.c_str(), (unsigned long)oldest->id);

        if (!oldest->disk_chunk_id.empty()) {
            std::error_code ec;
            bool removed = fs::remove(oldest->disk_chunk_id, ec);
            if (!removed || ec) {
                SRV_WRN("failed to remove disk chunk '%s': %s (code=%d)\n",
                        oldest->disk_chunk_id.c_str(), ec.message().c_str(), ec.value());
            } else {
                SRV_INF("successfully removed disk chunk '%s'\n", oldest->disk_chunk_id.c_str());
            }
        }

        oldest->disk_chunk_id.clear();
        oldest->tier = RADIX_TIER_EVICTED;
        if (oldest->children.empty()) {
            tree.remove_node(oldest);
        }
    }

    // 2. Enforce global shared disk quota across all model subdirectories in root cache dir
    std::string root_scan_dir = fs::path(disk_dir).parent_path().string();
    if (root_scan_dir.empty() || !fs::exists(root_scan_dir)) {
        root_scan_dir = disk_dir;
    }

    size_t total_global_bytes = compute_directory_ckpt_bytes(root_scan_dir);
    if (total_global_bytes > disk_quota_bytes) {
        struct disk_file_entry {
            std::string path;
            fs::file_time_type write_time;
            size_t size;
        };
        std::vector<disk_file_entry> all_files;
        try {
            for (const auto & entry : fs::recursive_directory_iterator(root_scan_dir, fs::directory_options::skip_permission_denied)) {
                if (entry.is_regular_file() && entry.path().extension() == ".ckpt") {
                    std::error_code ec;
                    auto wt = entry.last_write_time(ec);
                    auto sz = entry.file_size(ec);
                    if (!ec) {
                        all_files.push_back({ entry.path().string(), wt, sz });
                    }
                }
            }
        } catch (...) {}

        std::sort(all_files.begin(), all_files.end(), [](const disk_file_entry & a, const disk_file_entry & b) {
            return a.write_time < b.write_time;
        });

        for (const auto & file : all_files) {
            if (total_global_bytes <= disk_quota_bytes) {
                break;
            }
            SRV_INF("global disk quota: pruning oldest chunk '%s' (%.2f MiB)\n",
                    file.path.c_str(), file.size / (1024.0 * 1024.0));
            std::error_code ec;
            if (fs::remove(file.path, ec)) {
                total_global_bytes = (total_global_bytes > file.size) ? (total_global_bytes - file.size) : 0;
                auto disk_nodes = tree.get_lru_nodes(RADIX_TIER_DISK);
                for (auto & node : disk_nodes) {
                    if (node->disk_chunk_id == file.path) {
                        node->disk_chunk_id.clear();
                        node->tier = RADIX_TIER_EVICTED;
                        if (node->children.empty()) {
                            tree.remove_node(node);
                        }
                        break;
                    }
                }
            }
        }
    }
}
