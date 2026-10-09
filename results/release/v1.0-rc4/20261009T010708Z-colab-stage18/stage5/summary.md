# Stage 5 CPU-only planner memory gate

Memory gate: passed on predeclared chain. No latency gate, model or CUDA claim. Slower results retained. Ratio = median of three paired dynamic/reuse median-latency ratios. Resident bytes exclude metadata/RSS.

| Workload | Capacity reuse / no-reuse | Paired latency ratios | Median ratio |
|---|---:|---|---:|
| chain | 0.166667 | 0.6835, 0.6968, 0.7090 | 0.6968 |
| diamond | 0.600000 | 0.6703, 0.6850, 0.7705 | 0.6850 |

Optional inplace: skipped_optional. Decoder workload: unavailable_pending_S13. Prepare statistics and exact counts: planner.csv / analysis.json.
