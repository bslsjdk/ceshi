"""VN v3: two-layer concept wave.

v1/v1.1 used a FIXED kernel K = w_ij * cos(c_i, c_j); iterating a fixed linear
map converges to K's dominant eigenvector, so concept_sim_top5 stalled ~0.37.
v2 replaced K with a learned predictor F and gave concept a real gradient, but
concept_sim_top5 went NEGATIVE (-0.116): F learned to address neurons that are
*anti*-correlated with the injected character.

v3 adds the second spatial dimension the spec asked for: L=2 layers coupled by
a learned [2*Dc, 2*Dc] projection on the pooled field summaries.  Gradient now
reaches layer 0 through layer 1.
"""
from __future__ import annotations
