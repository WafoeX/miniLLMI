# Stage 5 CPU-only planner memory gate

Memory gate: passed on predeclared chain. No latency gate, model or CUDA claim. Slower results retained. Ratio = median of three paired dynamic/reuse median-latency ratios. Resident bytes exclude metadata/RSS.

| Workload | Capacity reuse / no-reuse | Paired latency ratios | Median ratio |
|---|---:|---|---:|
| chain | 0.166667 | 0.9090, 0.5793, 0.5617 | 0.5793 |
| diamond | 0.600000 | 0.7199, 0.6215, 0.5371 | 0.6215 |

Optional inplace: skipped_optional. Decoder workload: unavailable_pending_S13. Prepare statistics and exact counts: planner.csv / analysis.json.
