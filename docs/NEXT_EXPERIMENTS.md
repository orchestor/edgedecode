# Experiments required for stronger claims

## Completed gate: real model and task quality pilot

Qwen2.5-1.5B-Instruct is pinned by repository revision and file hash. F16, Q8_0 and Q4_K_M were evaluated on 32 WikiText-2 chunks and 200 fixed-seed HellaSwag tasks. This clears the model-level pilot gate, not a broad quality claim. Next, run the full quality sets and add instruction following plus a second architecture family.

## Completed gate: optimized software baseline pilot

A pinned llama.cpp revision now provides CPU-only and Metal measurements for prefill and decode. Next, capture Instruments profiles for each phase, identify dominant kernels and instructions, and add a native Arm platform with a pinned KleidiAI/ExecuTorch path.

## 3. GPU/NPU evidence

The custom Metal path and llama.cpp Metal backend now run on the M4 Pro. This is Apple GPU evidence, not Arm GPU evidence. The next decisive comparison needs a native Arm CPU/GPU and an Arm NPU target with its documented compiler. Export a supported graph, save compiler logs, inspect fallback/partitioning and actual quantization requirements.

## 4. Calibrate architecture model

Separate calibration workloads from held-out validation shapes. Fit/check bandwidth, compute and decode throughput within valid regimes; report residuals and uncertainty. Include tiling, SRAM footprint, activation conversion, launch costs and scheduling before treating the model as predictive. The present ideal-overlap scenario model is explicitly uncalibrated.

## 5. Power and area

Use documented power telemetry or an external meter with baseline and integration methodology; report the measured domain. Design a decoder RTL prototype with a throughput target, buffering, scale handling, and flow control. Synthesize against an identified library and constraints; state what is excluded. Only then compare actual area cost to the normalized break-even budget in this repository.

## Decision gates

- Quality gate: does low precision survive independent model-level evaluation?
- Software gate: does an optimized implementation still leave a relevant bottleneck?
- Architecture gate: does the benefit survive model uncertainty and full-system overhead?
- PPA gate: does it improve the intended objective after hardware cost is counted?

A rejected feature is a legitimate outcome. The current synthetic pilot rejects mixed *bitwidth* benefits at its selected quality threshold; it does not reject all quantization or dedicated decode hardware.
