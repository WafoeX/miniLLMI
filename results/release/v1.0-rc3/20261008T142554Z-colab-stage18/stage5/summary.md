# Stage 5 CPU-only planner memory gate

Memory gate: passed on predeclared chain. No latency gate, model or CUDA claim. Slower results retained. Ratio = median of three paired dynamic/reuse median-latency ratios. Resident bytes exclude metadata/RSS.

| Workload | Capacity reuse / no-reuse | Paired latency ratios | Median ratio |
|---|---:|---|---:|
| chain | 0.166667 | 0.6059, 0.7394, 0.6811 | 0.6811 |
| diamond | 0.600000 | 0.7022, 0.7170, 0.6933 | 0.7022 |

Optional inplace: skipped_optional. Decoder workload: unavailable_pending_S13. Prepare statistics and exact counts: planner.csv / analysis.json.
