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

### Storage Quota Scavenging
- **Host RAM (`--cache-ram`, `-cram`):** Limits total memory occupied by warm Radix checkpoints. Setting `-cram -1` allows unlimited RAM utilization.
- **NVMe Disk (`--cache-disk`, `-cdisk`):** Limits persistent SSD storage. Setting `-cdisk -1` allows unlimited disk caching up to physical drive capacity.
- **Eviction Policies (`--radix-eviction`):** When quotas are reached, colder nodes are selected via `lru` (Least Recently Used), `lfu` (Least Frequently Used), or `cost` (recomputation cost = token count $\times$ access frequency).

---

## 3. KVarN & Precision Tail Integration

### 128-Token Descriptor Alignment
BeeLlama's KVarN target KV-cache compression groups attention state into 128-token rotated and normalized tiles (`KVAR_N_GROUP = 128`).

- **Durable Checkpoints:** RXC enforces that durable KVarN checkpoints snap to complete 128-token boundaries.
- **Precision Tail Overlay:** Active trailing tokens ($< 128$ tokens) remain in exact F16/BF16 staging buffers without corrupting compressed historical tiles.
- **Safe Prefix Reuse:** During prefix lookup, RXC validates that restorable token counts match KVarN tile alignments, preventing partial-tile decoding artifacts.

---

## 4. Checkpoint Strategies & Modes

RXC provides granular control over when KV-cache states are snapshotted:

| Mode (`-cm`, `--checkpoint-mode`) | Description | Best For |
|---|---|---|
| `turn` | Snapshots state only when a generation turn completes. **Zero overhead during prefill/streaming.** | Interactive chat, coding assistants, multi-turn agents. |
| `step` | Snapshots state at fixed token intervals during prefill (`-cms`, `--checkpoint-min-step`). | Extremely long document ingestion (32K+ tokens). |
| `both` | Combines turn-end checkpoints with intermediate step checkpoints. | Long document ingestion followed by multi-turn Q&A. |
| `off` | Disables dynamic state snapshotting. | Stateless single-turn batch benchmarks. |

> [!TIP]
> **Turn Checkpointing Advantage:** When using `-cm turn`, prefill and generation operate at 100% full hardware speed without intermediate `llama_state_seq_get_data` interrupts. The snapshot is saved asynchronously upon generation completion.

---

## 5. Crash Resilience & Atomic Disk I/O

To guarantee zero cache corruption across server crashes or power failures:

1. **Two-Phase Atomic Commits:**
   - Checkpoint payloads are written to a temporary file (`.ckpt.tmp.<pid>.<uuid>`).
   - The file is flushed to physical NVMe media via POSIX `fsync()`.
   - The file is atomically renamed to its canonical chunk path (`<hash>.ckpt`).
2. **Checksum Verification:**
   - Every `.ckpt` chunk header includes an FNV-1a checksum of the payload.
   - Corrupt or truncated chunks are automatically detected and discarded.
3. **Startup Orphan Pruning:**
   - When `llama-server` boots with `--cache-disk`, it automatically purges orphaned `.tmp` files left behind by prior ungraceful shutdowns.

---

## 6. Multimodal & M-RoPE Support

RXC natively tracks multimodal media chunks (images, audio) and M-RoPE 2D/3D positional metadata within token sequences:
- Media chunks are identified by `LLAMA_TOKEN_NULL` token placeholders linked to their underlying high-dimensional feature embeddings.
- Prefix matching preserves media boundaries, ensuring that partial image embeddings are never split across checkpoint edges.

---

## 7. CLI Flags & Environment Variables

| Short | Long Flag | Environment Variable | Default | Description |
|---|---|---|---|---|
| `-rxc` | `--radix-cache` | `LLAMA_ARG_RADIX_CACHE` | `true` (server) | Enables dynamic Radix Tree prefix caching. |
| `-no-rxc`| `--no-radix-cache`| — | — | Disables Radix Tree prefix caching. |
| `-cram` | `--cache-ram N` | `LLAMA_ARG_CACHE_RAM` | `8192` | Host RAM quota in MiB (`-1` = unlimited, `0` = disabled). |
| `-cdisk`| `--cache-disk N` | `LLAMA_ARG_CACHE_DISK` | `0` | NVMe SSD quota in MiB (`-1` = unlimited, `0` = disabled). |
| — | `--cache-disk-dir PATH`| `LLAMA_ARG_CACHE_DISK_DIR` | `~/.cache/beellama.cpp/radix` | Directory path for disk `.ckpt` files. |
| `-cm` | `--checkpoint-mode MODE`| `LLAMA_ARG_CHECKPOINT_MODE`| `turn` (with radix) | Trigger policy: `turn`, `step`, `both`, or `off`. |
| `-cms` | `--checkpoint-min-step N`| `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT`| `0` (with radix) | Intra-prompt checkpoint token interval. |
| `-ctxcp`| `--ctx-checkpoints N`| `LLAMA_ARG_CTX_CHECKPOINTS` | `32` | Max checkpoints retained along a single branch. |
| — | `--radix-eviction POLICY`| `LLAMA_ARG_RADIX_EVICTION` | `lru` | Eviction strategy: `lru`, `lfu`, or `cost`. |

---

## 8. Practical Deployment Examples

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

### Example 2: Ultra-Low VRAM + NVMe SSD Offloading
Offloads long document KV states to NVMe SSD (100 GB disk cache):
```bash
llama-server \
  -m models/gemma-4-31b-q4_k_m.gguf \
  -c 65536 -b 2048 -ub 512 \
  -ctk kvarn4 -ctv kvarn3 \
  -rxc -cram 4096 -cdisk 102400 \
  --cache-disk-dir /mnt/fast-nvme/beellama-radix \
  -cm turn \
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

## 9. Observability & Telemetry

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
