# Radix Caching (RXC) Walkthrough and Guide

Radix Caching (RXC) is BeeLlama's high-performance hierarchical prefix caching system. It structures KV-cache checkpoints into a compressed token-prefix trie (Radix Tree) and manages a 3-tier memory hierarchy across **GPU VRAM**, **Host System RAM**, and **NVMe SSD Disk**.

This guide covers the architecture, storage tiers, KVarN alignment rules, checkpointing strategies, CLI flags, monitoring, and practical deployment workflows.

---

## 1. Architectural Overview

### The Problem with Flat Linear Prompt Caching
In standard llama.cpp, cached prompt states are stored in a linear list. Every incoming prompt must scan all cached entries sequentially and compute the longest common prefix (LCP). Under high concurrency or long multi-turn sessions, this approach causes:
- $O(N \cdot L)$ scan overhead across $N$ cache entries of length $L$.
- No awareness of shared sub-prefixes across branching conversations.
- Redundant KV-cache recomputation when users branch off a shared document or system prompt.
- High VRAM pressure, forcing cache entries to be discarded rather than staged to cheaper storage tiers.

```
       Linear Cache List (Legacy)
[ Prompt A: SysPrompt + User1 + Assistant1 ]
[ Prompt B: SysPrompt + User2 + Assistant2 ]  --> O(N * L) linear scan
[ Prompt C: SysPrompt + DocChunk1 + User3  ]
```

### The Radix Tree Solution (RXC)
RXC replaces the linear list with a token-level **Radix Tree** (compressed prefix trie) where common token sequences are merged into tree edges. Nodes represent branching points and checkpoint anchors.

```
                   [ Root Node ]
                         |
                 "System Prompt..." (Edge: 256 tokens)
                         |
                  [ Branch Node 1 ]
                 /                 \
     "User 1: ..."                  "User 2: ..."
     (Edge: 128 tok)                (Edge: 128 tok)
           |                              |
    [ Checkpoint A ]               [ Checkpoint B ]
    (Turn 1 KV State)              (Turn 1 KV State)
           |                              |
    "Assistant 1: ..."             "Assistant 2: ..."
```

Lookup complexity is $O(L)$ where $L$ is the token length of the incoming prompt, completely independent of how many thousands of conversation branches are cached.

---

## 2. 3-Tier Storage Hierarchy

RXC decouples KV-cache retention from GPU VRAM limits by managing three tiers of memory:

```
+-------------------------------------------------------------------------+
| Tier 1: GPU VRAM (Active Contexts)                                      |
|  - Zero-latency attention compute directly on device                    |
|  - Managed via model context slots                                      |
+------------------------------------+------------------------------------+
                                     | (Evict / Restore on Turn Completion)
                                     v
+-------------------------------------------------------------------------+
| Tier 2: Host System RAM (--cache-ram, default: 8192 MiB)                |
|  - Nanosecond-scale state restoration over PCIe                         |
|  - Deduplicated target + draft + speculative state buffers              |
+------------------------------------+------------------------------------+
                                     | (Spill cold nodes / Load on demand)
                                     v
+-------------------------------------------------------------------------+
| Tier 3: Persistent NVMe Disk (--cache-disk, default: 0 / disabled)      |
|  - Atomic `.ckpt` chunks on NVMe SSD                                    |
|  - Survives VRAM/RAM pressure and multi-gigabyte document caches         |
+-------------------------------------------------------------------------+
```

### Storage Quota Scavenging & Multi-Model Sharing
- **Host RAM (`--cache-ram`, `-cram`):** Limits total memory occupied by warm Radix checkpoints. Setting `-cram -1` allows unlimited RAM utilization. Setting `-cram 0` completely disables host RAM retention and streams checkpoints directly to NVMe SSD with true 0-RAM overhead.
- **NVMe Disk (`--cache-disk`, `-cdisk`):** Global persistent SSD quota in MiB. Setting `-cdisk -1` allows unlimited disk caching up to physical drive capacity.
- **Global Root Scavenging:** When `--cache-disk` is configured, the disk limit applies globally across all model subdirectories in `~/.cache/beellama.cpp/radix/`. If the combined footprint exceeds the quota, cross-model LRU scavenging automatically purges the globally oldest `.ckpt` files first.
- **Eviction Policies (`--radix-eviction`):** When quotas are reached, colder nodes are selected via `lru` (Least Recently Used), `lfu` (Least Frequently Used), or `cost` (recomputation cost = token count $\times$ access frequency).

---

## 3. Automatic Model & Quantization Isolation

To eliminate vocabulary collisions, tensor shape mismatches, and quantization drift across model switches, BeeLlama automatically isolates disk checkpoints into dedicated model fingerprint subfolders:

$$\text{Subdirectory} = \texttt{<model\_name>}\_\texttt{<quant>}\_\texttt{k-<ctk>}\_\texttt{v-<ctv>}$$

```
~/.cache/beellama.cpp/radix/
├── gemma4_26b_a4b_q4_k_medium_k-bf16_v-bf16/
│   ├── chunk_1_35767465.ckpt
│   └── chunk_3_35767518.ckpt
└── qwen2.5_7b_instruct_q4_k_m_k-q4_0_v-q4_0/
    ├── chunk_1_35856028.ckpt
    └── chunk_2_35856035.ckpt
```

- **Seamless Switching:** You can serve Gemma 4 in the morning and Qwen 2.5 in the afternoon without manually altering cache directories.
- **Custom Overrides:** If you provide a custom path (e.g. `--cache-disk-dir /mnt/nvme/my_cache`), BeeLlama automatically constructs isolated fingerprint folders within your target directory.

---

## 4. KVarN & Precision Tail Integration

### 128-Token Descriptor Alignment
BeeLlama's KVarN target KV-cache compression groups attention state into 128-token rotated and normalized tiles (`KVAR_N_GROUP = 128`).

- **Durable Checkpoints:** RXC enforces that durable KVarN checkpoints snap to complete 128-token boundaries.
- **Precision Tail Overlay:** Active trailing tokens ($< 128$ tokens) remain in exact F16/BF16 staging buffers without corrupting compressed historical tiles.
- **Safe Prefix Reuse:** During prefix lookup, RXC validates that restorable token counts match KVarN tile alignments, preventing partial-tile decoding artifacts.

---

## 5. Boundary-Driven Checkpointing & $1\text{ Node} = 1\text{ State}$ Normalization

RXC operates on a lean **boundary-driven checkpointing** policy to maximize prefix reuse while eliminating storage bloat:

1. **System & Turn Boundaries:** KV-cache state snapshots are automatically captured upon generation turn completion and prompt completion boundaries. **Zero prefill interrupt latency.**
2. **Single Canonical State per Node:** When inserting into the Radix Tree, intermediate ancestral checkpoint lists are pruned so each node holds exactly $1$ canonical checkpoint ($1\text{ Node} = 1\text{ State}$), reducing checkpoint storage by over $6\times$.

> [!TIP]
> **Turn Checkpointing Advantage:** Prefill and generation operate at 100% full hardware speed without intermediate intra-prompt interrupts. The snapshot is saved cleanly upon generation completion.

---

## 6. Crash Resilience & Atomic Disk I/O

To guarantee zero cache corruption across server crashes or power failures:

1. **Two-Phase Atomic Commits:**
   - Checkpoint payloads are written to a temporary file (`.ckpt.tmp.<pid>.<uuid>`).
   - The file is flushed to physical NVMe media via POSIX `fsync()`.
   - The file is atomically renamed to its canonical chunk path (`chunk_<id>_<hash>.ckpt`).
2. **Checksum Verification:**
   - Every `.ckpt` chunk header includes an FNV-1a checksum of the payload.
   - Corrupt or truncated chunks are automatically detected and discarded.
3. **Startup Orphan Pruning:**
   - When `llama-server` boots with `--cache-disk`, it automatically purges orphaned `.tmp` files left behind by prior ungraceful shutdowns.

---

## 7. CLI Flags & Environment Variables

| Short | Long Flag | Environment Variable | Default | Description |
|---|---|---|---|---|
| `-rxc` | `--radix-cache` | `LLAMA_ARG_RADIX_CACHE` | `true` (server) | Enables dynamic Radix Tree prefix caching. |
| `-no-rxc`| `--no-radix-cache`| — | — | Disables Radix Tree prefix caching. |
| `-cram` | `--cache-ram N` | `LLAMA_ARG_CACHE_RAM` | `8192` | Host RAM quota in MiB (`-1` = unlimited, `0` = direct NVMe spill). |
| `-cdisk`| `--cache-disk N` | `LLAMA_ARG_CACHE_DISK` | `0` | Global NVMe SSD quota in MiB (`-1` = unlimited, `0` = disabled). |
| — | `--cache-disk-dir PATH`| `LLAMA_ARG_CACHE_DISK_DIR` | `~/.cache/beellama.cpp/radix` | Root directory path for disk `.ckpt` files. |
| `-ctxcp`| `--ctx-checkpoints N`| `LLAMA_ARG_CTX_CHECKPOINTS` | `32` | Max checkpoints retained along a single branch. |
| — | `--radix-eviction POLICY`| `LLAMA_ARG_RADIX_EVICTION` | `lru` | Eviction strategy: `lru`, `lfu`, or `cost`. |

---

## 9. Practical Deployment Examples

### Example 1: Multi-Turn Server with RAM Caching (Default Setup)
Serves chat with 16 GB Host RAM allocated for instant prefix reuse:
```bash
llama-server \
  -m models/qwen3.6-27b-q5_k_s.gguf \
  -c 32768 -b 2048 -ub 512 \
  -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024 \
  -rxc -cram 16384 -cm turn \
  --port 8080
```

### Example 2: APU / UMA 0-RAM Direct NVMe Spill (Strix Halo / Radeon 8060S)
Spills warm checkpoints directly to NVMe SSD without duplicate host RAM consumption:
```bash
llama-server \
  -m models/gemma-4-26b-a4b.gguf \
  -c 32768 -b 1024 -ub 1024 -ngl all \
  -cram 0 -cdisk 16384 -cm turn \
  --load-mode dio \
  -ctk bf16 -ctv bf16 -fa on \
  --port 8080
```

### Example 3: Document Ingestion with Step Checkpointing
Ingests massive 100K+ token books or codebases with checkpoints every 4096 tokens:
```bash
llama-server \
  -m models/qwen3.6-27b-q5_k_s.gguf \
  -c 131072 -b 4096 -ub 512 \
  -rxc -cdisk -1 -cm both -cms 4096 \
  --port 8080
```

---

## 10. Observability & Telemetry

### OpenAI API Response Telemetry
Standard `/v1/chat/completions` responses expose cached token breakdowns:
```json
{
  "usage": {
    "prompt_tokens": 1024,
    "completion_tokens": 64,
    "total_tokens": 1088,
    "prompt_tokens_details": {
      "cached_tokens": 896
    }
  },
  "timings": {
    "cache_source": "live_plan",
    "cache_n": 896,
    "prompt_ms": 112.4
  }
}
```

### Server Logs
During prompt evaluation, `llama-server` logs Radix hit diagnostics:
```
srv  prompt_load: slot 0 matched radix prefix (n_tokens = 4096, lcp = 4096, source = "ram")
srv  prompt_load: slot 1 restored cold chunk from disk (n_tokens = 16384, source = "disk")
```

### Prometheus Metrics
When Prometheus metrics are enabled (`--metrics`), RXC exports:
- `beellama_radix_hits_total{tier="vram"}`: Direct slot prefix reuses.
- `beellama_radix_hits_total{tier="ram"}`: Checkpoints restored from Host RAM.
- `beellama_radix_hits_total{tier="disk"}`: Checkpoints restored from NVMe SSD.
- `beellama_radix_ram_bytes`: Active RAM memory usage of the Radix Tree.
- `beellama_radix_disk_bytes`: Active disk storage used by `.ckpt` files.
