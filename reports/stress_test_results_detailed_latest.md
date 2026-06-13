# Detailed Stress Test Results

Generated at: 2026-05-28 11:36:12
API: http://127.0.0.1:8000
Runs per preset: 5

## deep_chain_d20

| preset | elapsed_avg_sec | nodes_avg | nodes_min | nodes_max | max_depth_set | merged_ratio_avg | semantic_merge_ratio_avg | frontier_final_all_zero |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| low | 0.019 | 6 | 6 | 6 | [5] | 0.0 | 0.0 | 1 |
| medium | 0.026 | 8 | 8 | 8 | [7] | 0.0 | 0.0 | 1 |
| high | 0.035 | 10 | 10 | 10 | [9] | 0.0 | 0.0 | 1 |

## duplicate_fanout_d7_b3

| preset | elapsed_avg_sec | nodes_avg | nodes_min | nodes_max | max_depth_set | merged_ratio_avg | semantic_merge_ratio_avg | frontier_final_all_zero |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| low | 0.701 | 40 | 25 | 55 | [5] | 0.4245 | 0.0 | 1 |
| medium | 3.955 | 125.8 | 64 | 187 | [7] | 0.3725 | 0.0 | 1 |
| high | 4.848 | 136.6 | 64 | 214 | [7] | 0.6449 | 0.0 | 1 |

## unique_fanout_d7_b3

| preset | elapsed_avg_sec | nodes_avg | nodes_min | nodes_max | max_depth_set | merged_ratio_avg | semantic_merge_ratio_avg | frontier_final_all_zero |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| low | 0.703 | 38.8 | 25 | 49 | [5] | 0.4325 | 0.0 | 1 |
| medium | 3.229 | 130.6 | 64 | 202 | [7] | 0.3721 | 0.0 | 1 |
| high | 4.471 | 125.2 | 64 | 184 | [7] | 0.6438 | 0.0 | 1 |

## Sample Best Leaf Paths

### deep_chain_d20 / low / run#1

```json
[
  {
    "id": "0f31648b",
    "depth": 0,
    "equations": [
      "chain_eq_s137_L0"
    ],
    "thought_step": "Chain step seed=137 level=0",
    "scheduler_action": "root",
    "route_family": "chain",
    "score": 9.12
  },
  {
    "id": "b9195a81",
    "depth": 1,
    "equations": [
      "local_step_1 = pending"
    ],
    "thought_step": "Refine only the current subproblem: choose one active correction or closure. Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other p",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "401defc4",
    "depth": 2,
    "equations": [
      "chain_eq_s137_L2"
    ],
    "thought_step": "Chain step seed=137 level=2",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.09
  },
  {
    "id": "c8ff8624",
    "depth": 3,
    "equations": [
      "local_step_3 = pending"
    ],
    "thought_step": "Refine only the current subproblem: isolate the remaining unknown or boundary condition. Add or correct exactly one quantity, relation, approximation, or correction term, and leave",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "c2c7efc5",
    "depth": 4,
    "equations": [
      "chain_eq_s137_L4"
    ],
    "thought_step": "Chain step seed=137 level=4",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.06
  },
  {
    "id": "ba186418",
    "depth": 5,
    "equations": [
      "local_step_5 = pending"
    ],
    "thought_step": "Refine only the current subproblem: reduce to one solvable local relation. Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other pen",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  }
]
```

### deep_chain_d20 / medium / run#1

```json
[
  {
    "id": "800d05df",
    "depth": 0,
    "equations": [
      "chain_eq_s137_L0"
    ],
    "thought_step": "Chain step seed=137 level=0",
    "scheduler_action": "root",
    "route_family": "chain",
    "score": 9.12
  },
  {
    "id": "052cd8d7",
    "depth": 1,
    "equations": [
      "local_step_1 = pending"
    ],
    "thought_step": "Refine only the current subproblem: choose one active correction or closure. Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other p",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "ed8cb91c",
    "depth": 2,
    "equations": [
      "chain_eq_s137_L2"
    ],
    "thought_step": "Chain step seed=137 level=2",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.09
  },
  {
    "id": "78559cf0",
    "depth": 3,
    "equations": [
      "local_step_3 = pending"
    ],
    "thought_step": "Refine only the current subproblem: isolate the remaining unknown or boundary condition. Add or correct exactly one quantity, relation, approximation, or correction term, and leave",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "e18463d3",
    "depth": 4,
    "equations": [
      "chain_eq_s137_L4"
    ],
    "thought_step": "Chain step seed=137 level=4",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.06
  },
  {
    "id": "940e21c1",
    "depth": 5,
    "equations": [
      "local_step_5 = pending"
    ],
    "thought_step": "Refine only the current subproblem: reduce to one solvable local relation. Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other pen",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "7f0851a6",
    "depth": 6,
    "equations": [
      "chain_eq_s137_L6"
    ],
    "thought_step": "Chain step seed=137 level=6",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.04
  },
  {
    "id": "0b00ac89",
    "depth": 7,
    "equations": [
      "chain_eq_s137_L7"
    ],
    "thought_step": "Chain step seed=137 level=7",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.03
  }
]
```

### deep_chain_d20 / high / run#1

```json
[
  {
    "id": "8742d672",
    "depth": 0,
    "equations": [
      "chain_eq_s137_L0"
    ],
    "thought_step": "Chain step seed=137 level=0",
    "scheduler_action": "root",
    "route_family": "chain",
    "score": 9.12
  },
  {
    "id": "a12895b9",
    "depth": 1,
    "equations": [
      "local_step_1 = pending"
    ],
    "thought_step": "Refine only the current subproblem: choose one active correction or closure. Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other p",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "9b796f15",
    "depth": 2,
    "equations": [
      "chain_eq_s137_L2"
    ],
    "thought_step": "Chain step seed=137 level=2",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.09
  },
  {
    "id": "bb6d7e1b",
    "depth": 3,
    "equations": [
      "local_step_3 = pending"
    ],
    "thought_step": "Refine only the current subproblem: isolate the remaining unknown or boundary condition. Add or correct exactly one quantity, relation, approximation, or correction term, and leave",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "ac00a19b",
    "depth": 4,
    "equations": [
      "chain_eq_s137_L4"
    ],
    "thought_step": "Chain step seed=137 level=4",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.06
  },
  {
    "id": "c27ebc2c",
    "depth": 5,
    "equations": [
      "local_step_5 = pending"
    ],
    "thought_step": "Refine only the current subproblem: reduce to one solvable local relation. Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other pen",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "a029d7e3",
    "depth": 6,
    "equations": [
      "chain_eq_s137_L6"
    ],
    "thought_step": "Chain step seed=137 level=6",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.04
  },
  {
    "id": "04e1fb71",
    "depth": 7,
    "equations": [
      "local_step_7 = pending"
    ],
    "thought_step": "Refine only the current subproblem: verify consistency against limits and constraints. Add or correct exactly one quantity, relation, approximation, or correction term, and leave a",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 0.0
  },
  {
    "id": "158ac0ab",
    "depth": 8,
    "equations": [
      "chain_eq_s137_L8"
    ],
    "thought_step": "Chain step seed=137 level=8",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.01
  },
  {
    "id": "4686ae0b",
    "depth": 9,
    "equations": [
      "chain_eq_s137_L9"
    ],
    "thought_step": "Chain step seed=137 level=9",
    "scheduler_action": "expanded",
    "route_family": "chain",
    "score": 9.0
  }
]
```

### duplicate_fanout_d7_b3 / low / run#1

```json
[
  {
    "id": "28e6f9ba",
    "depth": 0,
    "equations": [
      "eq_momentum_dup_s137_L0"
    ],
    "thought_step": "Refine momentum via quadratic seed=137 level=0 branch=0",
    "scheduler_action": "root",
    "route_family": "momentum",
    "score": 8.83
  },
  {
    "id": "1d4cbaa4",
    "depth": 1,
    "equations": [
      "eq_closure_stability_L1_B0"
    ],
    "thought_step": "Refine closure via stability seed=154 level=1 branch=0",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.76
  },
  {
    "id": "fd408d14",
    "depth": 2,
    "equations": [
      "eq_energy_consistency_L2_B2"
    ],
    "thought_step": "Refine energy via consistency seed=238 level=2 branch=2",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 8.7
  }
]
```

### duplicate_fanout_d7_b3 / medium / run#1

```json
[
  {
    "id": "4ef1ae02",
    "depth": 0,
    "equations": [
      "eq_momentum_dup_s137_L0"
    ],
    "thought_step": "Refine momentum via quadratic seed=137 level=0 branch=0",
    "scheduler_action": "root",
    "route_family": "momentum",
    "score": 8.83
  },
  {
    "id": "1b96d5fd",
    "depth": 1,
    "equations": [
      "eq_closure_stability_L1_B2"
    ],
    "thought_step": "Refine closure via stability seed=216 level=1 branch=2",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.79
  },
  {
    "id": "ee813600",
    "depth": 2,
    "equations": [
      "eq_energy_dup_s272_L2"
    ],
    "thought_step": "Refine energy via consistency seed=272 level=2 branch=0",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 8.75
  },
  {
    "id": "525db259",
    "depth": 3,
    "equations": [
      "local_step_3 = pending"
    ],
    "thought_step": "Refine only the current subproblem: isolate the remaining unknown or boundary condition. Add or correct exactly one quantity, relation, approximation, or correction term, and leave",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 0.0
  },
  {
    "id": "0171f46c",
    "depth": 4,
    "equations": [
      "eq_closure_dup_s410_L4"
    ],
    "thought_step": "Refine closure via stability seed=410 level=4 branch=1",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.62
  },
  {
    "id": "6f05d2d6",
    "depth": 5,
    "equations": [
      "eq_momentum_quadratic_L5_B1"
    ],
    "thought_step": "Refine momentum via quadratic seed=495 level=5 branch=1",
    "scheduler_action": "expanded",
    "route_family": "momentum",
    "score": 8.61
  },
  {
    "id": "0efe2b8f",
    "depth": 6,
    "equations": [
      "eq_constraint_dup_s554_L6"
    ],
    "thought_step": "Refine constraint via linear seed=554 level=6 branch=0",
    "scheduler_action": "merged-duplicate",
    "route_family": "constraint",
    "score": 8.6
  }
]
```

### duplicate_fanout_d7_b3 / high / run#1

```json
[
  {
    "id": "68825f6a",
    "depth": 0,
    "equations": [
      "eq_momentum_dup_s137_L0"
    ],
    "thought_step": "Refine momentum via quadratic seed=137 level=0 branch=0",
    "scheduler_action": "root",
    "route_family": "momentum",
    "score": 8.83
  },
  {
    "id": "7f23a803",
    "depth": 1,
    "equations": [
      "eq_closure_stability_L1_B2"
    ],
    "thought_step": "Refine closure via stability seed=216 level=1 branch=2",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.79
  },
  {
    "id": "37160df3",
    "depth": 2,
    "equations": [
      "eq_energy_dup_s272_L2"
    ],
    "thought_step": "Refine energy via consistency seed=272 level=2 branch=0",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 8.75
  },
  {
    "id": "3c69f79b",
    "depth": 3,
    "equations": [
      "local_step_3 = pending"
    ],
    "thought_step": "Refine only the current subproblem: isolate the remaining unknown or boundary condition. Add or correct exactly one quantity, relation, approximation, or correction term, and leave",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 0.0
  },
  {
    "id": "8cf3456c",
    "depth": 4,
    "equations": [
      "eq_closure_dup_s458_L4"
    ],
    "thought_step": "Refine closure via stability seed=458 level=4 branch=1",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.62
  },
  {
    "id": "aeee2311",
    "depth": 5,
    "equations": [
      "eq_momentum_quadratic_L5_B1"
    ],
    "thought_step": "Refine momentum via quadratic seed=543 level=5 branch=1",
    "scheduler_action": "expanded",
    "route_family": "momentum",
    "score": 8.61
  },
  {
    "id": "cb1f5642",
    "depth": 6,
    "equations": [
      "eq_constraint_dup_s602_L6"
    ],
    "thought_step": "Refine constraint via linear seed=602 level=6 branch=0",
    "scheduler_action": "merged-duplicate",
    "route_family": "constraint",
    "score": 8.6
  }
]
```

### unique_fanout_d7_b3 / low / run#1

```json
[
  {
    "id": "36f3e5d2",
    "depth": 0,
    "equations": [
      "eq_unique_momentum_quadratic_s137_L0_B0_137"
    ],
    "thought_step": "Refine momentum via quadratic seed=137 level=0 branch=0",
    "scheduler_action": "root",
    "route_family": "momentum",
    "score": 8.83
  },
  {
    "id": "ed25fa70",
    "depth": 1,
    "equations": [
      "eq_unique_closure_stability_s154_L1_B0_161"
    ],
    "thought_step": "Refine closure via stability seed=154 level=1 branch=0",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.76
  },
  {
    "id": "722468a2",
    "depth": 2,
    "equations": [
      "eq_unique_energy_consistency_s238_L2_B2_278"
    ],
    "thought_step": "Refine energy via consistency seed=238 level=2 branch=2",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 8.7
  }
]
```

### unique_fanout_d7_b3 / medium / run#1

```json
[
  {
    "id": "f33c15ca",
    "depth": 0,
    "equations": [
      "eq_unique_momentum_quadratic_s137_L0_B0_137"
    ],
    "thought_step": "Refine momentum via quadratic seed=137 level=0 branch=0",
    "scheduler_action": "root",
    "route_family": "momentum",
    "score": 8.83
  },
  {
    "id": "50b2add8",
    "depth": 1,
    "equations": [
      "eq_unique_closure_stability_s216_L1_B2_249"
    ],
    "thought_step": "Refine closure via stability seed=216 level=1 branch=2",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.79
  },
  {
    "id": "e9c8a427",
    "depth": 2,
    "equations": [
      "eq_unique_energy_consistency_s272_L2_B0_286"
    ],
    "thought_step": "Refine energy via consistency seed=272 level=2 branch=0",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 8.75
  },
  {
    "id": "0061a336",
    "depth": 3,
    "equations": [
      "local_step_3 = pending"
    ],
    "thought_step": "Refine only the current subproblem: isolate the remaining unknown or boundary condition. Add or correct exactly one quantity, relation, approximation, or correction term, and leave",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 0.0
  },
  {
    "id": "6eff53bc",
    "depth": 4,
    "equations": [
      "eq_unique_closure_stability_s458_L4_B1_499"
    ],
    "thought_step": "Refine closure via stability seed=458 level=4 branch=1",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.62
  },
  {
    "id": "84f6ffa0",
    "depth": 5,
    "equations": [
      "eq_unique_momentum_quadratic_s543_L5_B1_591"
    ],
    "thought_step": "Refine momentum via quadratic seed=543 level=5 branch=1",
    "scheduler_action": "expanded",
    "route_family": "momentum",
    "score": 8.61
  },
  {
    "id": "8b0ec3c6",
    "depth": 6,
    "equations": [
      "eq_unique_constraint_linear_s602_L6_B0_644"
    ],
    "thought_step": "Refine constraint via linear seed=602 level=6 branch=0",
    "scheduler_action": "merged-duplicate",
    "route_family": "constraint",
    "score": 8.6
  }
]
```

### unique_fanout_d7_b3 / high / run#1

```json
[
  {
    "id": "e12a0b7c",
    "depth": 0,
    "equations": [
      "eq_unique_momentum_quadratic_s137_L0_B0_137"
    ],
    "thought_step": "Refine momentum via quadratic seed=137 level=0 branch=0",
    "scheduler_action": "root",
    "route_family": "momentum",
    "score": 8.83
  },
  {
    "id": "315b620c",
    "depth": 1,
    "equations": [
      "eq_unique_closure_stability_s216_L1_B2_249"
    ],
    "thought_step": "Refine closure via stability seed=216 level=1 branch=2",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.79
  },
  {
    "id": "7519c620",
    "depth": 2,
    "equations": [
      "eq_unique_energy_consistency_s272_L2_B0_286"
    ],
    "thought_step": "Refine energy via consistency seed=272 level=2 branch=0",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 8.75
  },
  {
    "id": "65582bfe",
    "depth": 3,
    "equations": [
      "local_step_3 = pending"
    ],
    "thought_step": "Refine only the current subproblem: isolate the remaining unknown or boundary condition. Add or correct exactly one quantity, relation, approximation, or correction term, and leave",
    "scheduler_action": "expanded",
    "route_family": "energy",
    "score": 0.0
  },
  {
    "id": "19fdca84",
    "depth": 4,
    "equations": [
      "eq_unique_closure_stability_s410_L4_B1_451"
    ],
    "thought_step": "Refine closure via stability seed=410 level=4 branch=1",
    "scheduler_action": "expanded",
    "route_family": "closure",
    "score": 8.62
  },
  {
    "id": "26656ab9",
    "depth": 5,
    "equations": [
      "eq_unique_momentum_quadratic_s495_L5_B1_543"
    ],
    "thought_step": "Refine momentum via quadratic seed=495 level=5 branch=1",
    "scheduler_action": "expanded",
    "route_family": "momentum",
    "score": 8.61
  },
  {
    "id": "c0b96c0d",
    "depth": 6,
    "equations": [
      "eq_unique_constraint_linear_s554_L6_B0_596"
    ],
    "thought_step": "Refine constraint via linear seed=554 level=6 branch=0",
    "scheduler_action": "merged-duplicate",
    "route_family": "constraint",
    "score": 8.6
  }
]
```

