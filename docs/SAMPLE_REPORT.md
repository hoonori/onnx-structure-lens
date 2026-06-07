# Structure Lens Report: tiny_transformer_block

## Summary
- Nodes: **15**
- Tensors: **22**
- Known FLOPs proxy: **17,186,816**
- Known bytes moved proxy: **1,835,008**

## Recommendations
- What-if `S=256` changes known FLOPs by 17,186,816 (2.00x) across 15 nodes.
- Inspect fanout node `ln1` first: it branches into 3 consumers and may define a major structural split.
- Inspect fanin node `qk_scores` first: it merges 2 producers and may correspond to residual/concat behavior.
- Most impactful visible shape parameter is `B=1` with 15 affected nodes and 17,186,816 known FLOPs.
- Detected 6 collapsible subgroups (attention, chain, pattern, residual); use these as a first graph simplification layer.
- 1 nodes have unknown FLOPs formulas; add op-specific formulas for: k_transpose.

## Topology
- Source nodes: ln1
- Sink nodes: residual_2
- Critical path length: 13
- Critical path: ln1 → k_proj → k_transpose → qk_scores → softmax → attn_ctx → out_proj → residual_1 → ln2 → fc1 → gelu → fc2 → residual_2
- Top fanout nodes: ln1(3), residual_1(2)
- Top fanin nodes: qk_scores(2), attn_ctx(2), residual_2(2)

## Shape / Hyperparameter Linkage
| Param | Dim | Tensor axes | Nodes touched |
|---|---:|---:|---:|
| `B=1` | `1` | 16 | 15 |
| `S=128` | `128` | 15 | 15 |
| `C=64` | `64` | 11 | 13 |
| `M=64` | `64` | 5 | 5 |
| `N=64` | `64` | 5 | 5 |
| `C=128` | `128` | 3 | 4 |
| `C=256` | `256` | 2 | 3 |
| `S=64` | `64` | 1 | 2 |
| `M=256` | `256` | 1 | 1 |
| `N=256` | `256` | 1 | 1 |

## Parameter Impact
| Param | Affected nodes | Known FLOPs | Known bytes |
|---|---:|---:|---:|
| `B=1` | 15 | 17,186,816 | 1,835,008 |
| `S=128` | 15 | 17,186,816 | 1,835,008 |
| `C=64` | 13 | 16,875,520 | 1,441,792 |
| `C=256` | 3 | 8,650,752 | 720,896 |
| `M=64` | 5 | 8,388,608 | 557,056 |
| `N=64` | 5 | 8,388,608 | 557,056 |
| `C=128` | 4 | 4,243,456 | 458,752 |
| `M=256` | 1 | 4,194,304 | 229,376 |
| `N=256` | 1 | 4,194,304 | 229,376 |
| `S=64` | 2 | 2,097,152 | 196,608 |

## What-if Analysis
### `S=256`
- Known FLOPs: 17,186,816 → 34,373,632 (+17,186,816, 2.00x)
- Known bytes: 1,835,008 → 3,604,480 (+1,769,472)
- Changed nodes: 15
| Node | Op | FLOPs before | FLOPs after | Delta | Ratio |
|---|---|---:|---:|---:|---:|
| `fc1` | MatMul | 4,194,304 | 8,388,608 | +4,194,304 | 2.00x |
| `fc2` | MatMul | 4,194,304 | 8,388,608 | +4,194,304 | 2.00x |
| `qk_scores` | MatMul | 2,097,152 | 4,194,304 | +2,097,152 | 2.00x |
| `attn_ctx` | MatMul | 2,097,152 | 4,194,304 | +2,097,152 | 2.00x |
| `q_proj` | MatMul | 1,048,576 | 2,097,152 | +1,048,576 | 2.00x |
| `k_proj` | MatMul | 1,048,576 | 2,097,152 | +1,048,576 | 2.00x |
| `v_proj` | MatMul | 1,048,576 | 2,097,152 | +1,048,576 | 2.00x |
| `out_proj` | MatMul | 1,048,576 | 2,097,152 | +1,048,576 | 2.00x |
| `gelu` | Gelu | 262,144 | 524,288 | +262,144 | 2.00x |
| `softmax` | Softmax | 49,152 | 98,304 | +49,152 | 2.00x |
| `ln1` | LayerNormalization | 40,960 | 81,920 | +40,960 | 2.00x |
| `ln2` | LayerNormalization | 40,960 | 81,920 | +40,960 | 2.00x |
| `residual_1` | Add | 8,192 | 16,384 | +8,192 | 2.00x |
| `residual_2` | Add | 8,192 | 16,384 | +8,192 | 2.00x |
| `k_transpose` | Transpose | ? | ? | ? | ? |

## Detected Subgroups
- **AttentionCore:softmax** (`attention`): attn_ctx, qk_scores, softmax — MatMul/Gemm -> Softmax -> MatMul/Gemm motif
- **LinearChain:ln2..fc2** (`chain`): ln2, fc1, gelu, fc2 — Long single-producer/single-consumer chain
- **LinearBias** (`pattern`): out_proj, residual_1 — Matched op pattern LinearBias
- **LinearBias** (`pattern`): fc2, residual_2 — Matched op pattern LinearBias
- **NormThenLinear** (`pattern`): ln2, fc1 — Matched op pattern NormThenLinear
- **ResidualJoin:residual_2** (`residual`): fc2, residual_1, residual_2 — Add/Sum node merges multiple producer branches

## Top Node Costs
| Node | Op | FLOPs | Bytes | Formula |
|---|---|---:|---:|---|
| `fc1` | MatMul | 4,194,304 | 229,376 | 2 * batch * M * K * N |
| `fc2` | MatMul | 4,194,304 | 229,376 | 2 * batch * M * K * N |
| `qk_scores` | MatMul | 2,097,152 | 131,072 | 2 * batch * M * K * N |
| `attn_ctx` | MatMul | 2,097,152 | 131,072 | 2 * batch * M * K * N |
| `q_proj` | MatMul | 1,048,576 | 81,920 | 2 * batch * M * K * N |
| `k_proj` | MatMul | 1,048,576 | 81,920 | 2 * batch * M * K * N |
| `v_proj` | MatMul | 1,048,576 | 81,920 | 2 * batch * M * K * N |
| `out_proj` | MatMul | 1,048,576 | 81,920 | 2 * batch * M * K * N |
| `gelu` | Gelu | 262,144 | 262,144 | ~8 * output_elements activation approximation |
| `softmax` | Softmax | 49,152 | 131,072 | ~3 * elements along softmax axis |
| `ln1` | LayerNormalization | 40,960 | 65,536 | ~5 * elements |
| `ln2` | LayerNormalization | 40,960 | 65,536 | ~5 * elements |
| `residual_1` | Add | 8,192 | 98,304 | 1 * output_elements |
| `residual_2` | Add | 8,192 | 98,304 | 1 * output_elements |
| `k_transpose` | Transpose | ? | 65,536 | shape-only / metadata op |
