"""Task 1 step 1: exact bytes-per-VN ledger for the adjacency.

The spec budgeted 64 B/VN for the edge list (uint32 dst + float32 w).
The population actually carries five [E] arrays, four of which the hot
loop never reads.  Measure before compressing.
"""
from __future__ import annotations
import json
import numpy as np
from vn_mobile.core.neuron import build_population

N, DEG = 1000, 8
pop = build_population(n=N, degree=DEG, seed=0)
E = pop.n_edges
per_edge = ["src", "dst", "weight", "dir_cos", "wedge"]
per_node = ["ptr"]
led = {}
for k in per_edge:
    a = getattr(pop, k)
    led[k] = dict(items=int(a.size), bytes=int(a.nbytes),
                  per_vn=round(a.nbytes / N, 1))
for k in per_node:
    a = getattr(pop, k)
    led[k] = dict(items=int(a.size), bytes=int(a.nbytes),
                  per_vn=round(a.nbytes / N, 1))
tot = sum(led[k]["bytes"] for k in per_edge + per_node)
led["_total_adj"] = dict(bytes=tot, per_vn=round(tot / N, 1))
led["_full_pop"] = dict(bytes=int(pop.nbytes()),
                        per_vn=round(pop.nbytes() / N, 1))
led["_E"] = E
led["_deg"] = DEG
print(json.dumps(led, indent=1))
with open("adj_ledger.json", "w") as f:
    json.dump(led, f, indent=1)
