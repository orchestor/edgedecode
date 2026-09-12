# Evidence ledger

| Claim | Evidence | Status |
|---|---|---|
| Host executes arm64 code | `manifest.json` platform.machine | observed |
| Exact host is M4 Pro, 24 GB | `manifest.json` sanitized system query | observed |
| CPU packed output matches dequantized reference | native tests + runtime assertions | measured |
| 28 CPU configuration timings | `operators.json` raw samples | measured pilot |
| Custom Metal kernel runs correctly | optional runtime test on M4 Pro | measured |
| 28 Metal operator configurations | `results/m4-metal/operators.json` | measured reference-kernel pilot |
| Synthetic quality constraint satisfied | `search.json`, full candidate ledger | established for development set; independent test reported |
| Qwen2.5-1.5B Q8 quality matches F16 pilot | `quality.json`, WikiText-2/HellaSwag subsets | measured pilot |
| Qwen2.5-1.5B Q4 has a small quality cost | 3.94% PPL increase; -1.5 HellaSwag points | measured pilot; HellaSwag intervals overlap |
| Q4 improves 128-token decode throughput | 1.95x CPU; 2.25x Metal versus F16 | measured llama.cpp, seven repetitions |
| Low precision universally improves prefill | Metal F16 beats Q8/Q4 | rejected on this host/workload |
| NPU deployment works | hypothetical static contract only | NOT established |
| Additional decode throughput may change bottlenecks | scenario equations and assumptions | hypothetical |
| Energy improves or area is known | no power meter or RTL synthesis | NOT established |

No fitted latency calibration is presented. Scenario energy omits leakage/control/complete hierarchy; feature variants reuse the same energy coefficients, so their energy difference is deliberately marked unassessed. The area number is a *maximum tolerable normalized budget*, not an estimate of a circuit.

The custom CPU code is single-threaded C with compiler optimization. The pinned llama.cpp build reports NEON, FP16 vector arithmetic, INT8 matmul, dot product, SME, Accelerate and repacking features, but this feature report alone does not prove which instruction path dominates each measurement. Apple Metal evidence must not be presented as Arm GPU evidence.
