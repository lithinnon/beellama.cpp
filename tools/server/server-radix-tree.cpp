#include "server-radix-tree.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <queue>

static int64_t get_monotonic_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t server_radix_node::accounted_size() const {
    size_t res = data.size();
    for (const auto & ckpt : prompt.checkpoints) {
        res += ckpt.size();
    }
    return res;
}

void server_radix_node::touch() {
    last_accessed_time = get_monotonic_now_ms();
    ++access_count;
}

server_radix_tree::server_radix_tree() {
    root = std::make_shared<server_radix_node>();
    root->id = 0;
    root->token_depth = 0;
    root->tier = RADIX_TIER_NONE;
    root->touch();
}

std::shared_ptr<server_radix_node> server_radix_tree::insert(
        const server_prompt & prompt,
        server_prompt_data && data,
        int32_t alignment) {
    GGML_UNUSED(alignment);

    if (prompt.tokens.size() == 0) {
        return nullptr;
    }

    const llama_tokens & full_tokens = prompt.tokens.get_tokens();
    std::shared_ptr<server_radix_node> curr = root;
    size_t token_idx = 0;

    while (token_idx < full_tokens.size()) {
        const llama_token next_token = full_tokens[token_idx];
        bool edge_found = false;

        for (size_t i = 0; i < curr->children.size(); ++i) {
            auto & edge = curr->children[i];
            if (!edge.tokens.empty() && edge.tokens[0] == next_token) {
                edge_found = true;

                // Compute length of common prefix between edge and remaining tokens
                size_t match_len = 0;
                while (match_len < edge.tokens.size() &&
                       token_idx + match_len < full_tokens.size() &&
                       edge.tokens[match_len] == full_tokens[token_idx + match_len]) {
                    ++match_len;
                }

                if (match_len < edge.tokens.size()) {
                    // Split the edge
                    auto split_node = std::make_shared<server_radix_node>();
                    split_node->id = next_node_id++;
                    split_node->token_depth = curr->token_depth + match_len;
                    split_node->tier = RADIX_TIER_NONE;
                    split_node->parent = curr;
                    split_node->touch();

                    // Edge to old child with remaining suffix
                    std::vector<llama_token> old_child_tokens(
                            edge.tokens.begin() + match_len, edge.tokens.end());
                    auto old_child = edge.child;
                    old_child->parent = split_node;

                    split_node->children.push_back({ std::move(old_child_tokens), old_child });

                    // Update current edge to point to split_node
                    edge.tokens.resize(match_len);
                    edge.child = split_node;

                    curr = split_node;
                    token_idx += match_len;
                } else {
                    // Edge fully matched
                    curr = edge.child;
                    token_idx += match_len;
                }
                break;
            }
        }

        if (!edge_found) {
            // Create a new branch
            std::vector<llama_token> new_tokens(full_tokens.begin() + token_idx, full_tokens.end());
            auto new_node = std::make_shared<server_radix_node>();
            new_node->id = next_node_id++;
            new_node->token_depth = full_tokens.size();
            new_node->tier = RADIX_TIER_NONE;
            new_node->parent = curr;
            new_node->touch();

            curr->children.push_back({ std::move(new_tokens), new_node });
            curr = new_node;
            token_idx = full_tokens.size();
            break;
        }
    }

    // If destination node already had a disk chunk from an older checkpoint, remove the orphaned file
    if (!curr->disk_chunk_id.empty()) {
        if (std::filesystem::exists(curr->disk_chunk_id)) {
            try {
                std::filesystem::remove(curr->disk_chunk_id);
            } catch (...) {}
        }
        curr->disk_chunk_id.clear();
    }

    // Set state on destination node
    curr->prompt = prompt.clone();
    curr->data = std::move(data);
    curr->tier = RADIX_TIER_RAM;
    curr->is_checkpoint = true;
    curr->payload_bytes = curr->accounted_size();
    curr->touch();

    return curr;
}

server_radix_match_result server_radix_tree::find_best_match(
        const server_tokens & requested,
        int32_t alignment,
        size_t live_native_restorable) {
    server_radix_match_result best;
    best.node = nullptr;
    best.lexical_tokens = 0;
    best.restorable_tokens = 0;
    best.reason = SERVER_PROMPT_REUSE_NONE;

    if (requested.size() == 0) {
        return best;
    }

    const int32_t align = std::max(1, alignment);
    const llama_tokens & req_tokens = requested.get_tokens();

    std::shared_ptr<server_radix_node> curr = root;
    size_t token_idx = 0;

    while (curr) {
        // If current node has a valid checkpoint, evaluate reuse
        if (curr->is_checkpoint && curr->tier != RADIX_TIER_NONE && curr->tier != RADIX_TIER_EVICTED) {
            server_prompt_reuse_plan plan = server_prompt_plan_reuse(
                    curr->prompt, requested, align, live_native_restorable, true);

            if (plan.restorable_tokens > best.restorable_tokens ||
                (plan.restorable_tokens == best.restorable_tokens && plan.lexical_tokens > best.lexical_tokens)) {
                best.node = curr;
                best.lexical_tokens = plan.lexical_tokens;
                best.restorable_tokens = plan.restorable_tokens;
                best.reason = plan.reason;
            }
        }

        if (token_idx >= req_tokens.size()) {
            break;
        }

        const llama_token next_token = req_tokens[token_idx];
        bool child_matched = false;

        for (const auto & edge : curr->children) {
            if (!edge.tokens.empty() && edge.tokens[0] == next_token) {
                size_t match_len = 0;
                while (match_len < edge.tokens.size() &&
                       token_idx + match_len < req_tokens.size() &&
                       edge.tokens[match_len] == req_tokens[token_idx + match_len]) {
                    ++match_len;
                }

                if (match_len == edge.tokens.size()) {
                    // Fully traversed edge
                    curr = edge.child;
                    token_idx += match_len;
                    child_matched = true;
                    break;
                } else {
                    // Partial edge match - check if child has checkpoints applicable to requested prefix
                    if (edge.child && edge.child->is_checkpoint &&
                        edge.child->tier != RADIX_TIER_NONE && edge.child->tier != RADIX_TIER_EVICTED) {
                        server_prompt_reuse_plan plan = server_prompt_plan_reuse(
                                edge.child->prompt, requested, align, live_native_restorable, true);

                        if (plan.restorable_tokens > best.restorable_tokens ||
                            (plan.restorable_tokens == best.restorable_tokens && plan.lexical_tokens > best.lexical_tokens)) {
                            best.node = edge.child;
                            best.lexical_tokens = plan.lexical_tokens;
                            best.restorable_tokens = plan.restorable_tokens;
                            best.reason = plan.reason;
                        }
                    }
                    token_idx += match_len;
                    curr = nullptr;
                    break;
                }
            }
        }

        if (!child_matched) {
            break;
        }
    }

    best.lexical_tokens = std::max(best.lexical_tokens, token_idx);
    return best;
}

void server_radix_tree::collect_all_nodes(
        const std::shared_ptr<server_radix_node> & node,
        std::vector<std::shared_ptr<server_radix_node>> & out) const {
    if (!node) return;
    out.push_back(node);
    for (const auto & edge : node->children) {
        collect_all_nodes(edge.child, out);
    }
}

std::vector<std::shared_ptr<server_radix_node>> server_radix_tree::get_lru_nodes(
        server_radix_tier filter_tier) const {
    std::vector<std::shared_ptr<server_radix_node>> all;
    collect_all_nodes(root, all);

    std::vector<std::shared_ptr<server_radix_node>> result;
    for (const auto & node : all) {
        if (node == root) continue;
        if (filter_tier == RADIX_TIER_NONE || node->tier == filter_tier) {
            if (node->is_checkpoint || !node->data.main.empty() || !node->disk_chunk_id.empty()) {
                result.push_back(node);
            }
        }
    }

    std::sort(result.begin(), result.end(), [](const std::shared_ptr<server_radix_node> & a,
                                               const std::shared_ptr<server_radix_node> & b) {
        return a->last_accessed_time < b->last_accessed_time;
    });

    return result;
}

void server_radix_tree::evict_ram_payload(const std::shared_ptr<server_radix_node> & node) {
    if (!node) return;
    node->data.main.clear();
    node->data.drft.clear();
    node->data.spec.clear();
    node->data.main.shrink_to_fit();
    node->data.drft.shrink_to_fit();
    node->data.spec.shrink_to_fit();

    for (auto & ckpt : node->prompt.checkpoints) {
        ckpt.data_tgt.clear();
        ckpt.data_dft.clear();
        ckpt.data_spec.clear();
        ckpt.data_spec.shrink_to_fit();
    }

    if (!node->disk_chunk_id.empty()) {
        node->tier = RADIX_TIER_DISK;
    } else {
        node->tier = RADIX_TIER_EVICTED;
    }
}

bool server_radix_tree::remove_node(const std::shared_ptr<server_radix_node> & node) {
    if (!node || node == root) return false;

    auto parent = node->parent.lock();
    if (!parent) return false;

    for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
        if (it->child == node) {
            parent->children.erase(it);
            prune_empty_leaves(parent);
            return true;
        }
    }
    return false;
}

void server_radix_tree::prune_empty_leaves(std::shared_ptr<server_radix_node> & node) {
    if (!node || node == root) return;
    if (node->children.empty() && (!node->is_checkpoint || node->tier == RADIX_TIER_EVICTED || node->tier == RADIX_TIER_NONE)) {
        auto parent = node->parent.lock();
        if (parent) {
            for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
                if (it->child == node) {
                    parent->children.erase(it);
                    prune_empty_leaves(parent);
                    break;
                }
            }
        }
    }
}

void server_radix_tree::clear() {
    root->children.clear();
    root->prompt = server_prompt();
    root->data = server_prompt_data();
    root->tier = RADIX_TIER_NONE;
    root->is_checkpoint = false;
    next_node_id = 1;
}

size_t server_radix_tree::accounted_ram_bytes() const {
    std::vector<std::shared_ptr<server_radix_node>> all;
    collect_all_nodes(root, all);

    std::unordered_set<const void *> retained_checkpoint_storage;
    size_t total = 0;

    for (const auto & node : all) {
        if (node->tier == RADIX_TIER_RAM) {
            total += node->data.size();
            for (const auto & ckpt : node->prompt.checkpoints) {
                if (!ckpt.data_tgt.empty() &&
                    retained_checkpoint_storage.insert(ckpt.data_tgt.storage_id()).second) {
                    total += ckpt.data_tgt.size();
                }
                if (!ckpt.data_dft.empty() &&
                    retained_checkpoint_storage.insert(ckpt.data_dft.storage_id()).second) {
                    total += ckpt.data_dft.size();
                }
                total += ckpt.data_spec.size();
            }
        }
    }
    return total;
}

size_t server_radix_tree::accounted_disk_bytes() const {
    std::vector<std::shared_ptr<server_radix_node>> all;
    collect_all_nodes(root, all);

    size_t total = 0;
    for (const auto & node : all) {
        if (node->tier == RADIX_TIER_DISK) {
            total += node->payload_bytes;
        }
    }
    return total;
}

size_t server_radix_tree::total_nodes() const {
    std::vector<std::shared_ptr<server_radix_node>> all;
    collect_all_nodes(root, all);
    return all.size();
}

size_t server_radix_tree::total_checkpoints() const {
    std::vector<std::shared_ptr<server_radix_node>> all;
    collect_all_nodes(root, all);

    size_t count = 0;
    for (const auto & node : all) {
        if (node->is_checkpoint && node->tier != RADIX_TIER_NONE && node->tier != RADIX_TIER_EVICTED) {
            ++count;
        }
    }
    return count;
}

size_t server_radix_tree::total_tokens() const {
    std::vector<std::shared_ptr<server_radix_node>> all;
    collect_all_nodes(root, all);

    size_t max_depth = 0;
    for (const auto & node : all) {
        if (node->token_depth > max_depth) {
            max_depth = node->token_depth;
        }
    }
    return max_depth;
}
