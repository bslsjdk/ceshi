"""FD check in SOFT mode: if analytic==FD here but not in hard mode, the
mismatch is the straight-through estimator, not a wrong gradient formula."""
import json
import numpy as np
from conceptwave2.predict import Predictor
from conceptwave2.field2 import Field2
from conceptwave2.train2 import Codec2, mse_term

txt = open("tinyshakespeare.txt", encoding="utf-8").read()
chars = sorted(set(txt)); stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in txt], np.int64); V = len(chars)
dc, n, lam = 32, 1000, 0.1

def loss_at(field, F, codec, emb_t, tgt):
    field.reset(); field.seed_from(emb_t); field.settle(F, soft=True)
    aff = codec.aff(field); lg = codec.logits(field, aff)
    l, _ = codec.head_grads(field, tgt, lg, aff)
    s, x, _, _, _ = field._last
    m, _, _ = mse_term(field, codec, x, s, tgt, lam)
    return l + m

field = Field2(n=n, dc=dc, seed=0); F = Predictor(dc=dc, hidden=64, seed=0)
codec = Codec2(V, dc, seed=0)
tgt = int(ids[1]); emb_t = codec.emb[int(ids[0])]
field.reset(); field.seed_from(emb_t); field.settle(F, soft=True)
aff = codec.aff(field); lg = codec.logits(field, aff)
l0, hg = codec.head_grads(field, tgt, lg, aff)
srcs, nxt, nn, M, top = field._last
ml, d_nxt, de = mse_term(field, codec, nxt, srcs, tgt, lam)
bg = field.backward(F, hg["act"], d_nxt)
g = (bg["concept"] + hg["cn"]).astype(np.float64)
rows = []
for i in np.argsort(-np.abs(g).sum(1))[:5]:
    d = int(np.argmax(np.abs(g[i]))); ana = float(g[i, d]); best = None
    for eps in [1e-3, 1e-2, 5e-2]:
        o = field.concept[i, d]
        field.concept[i, d] = o + eps; field.renorm(); lp = loss_at(field, F, codec, emb_t, tgt)
        field.concept[i, d] = o - eps; field.renorm(); lm = loss_at(field, F, codec, emb_t, tgt)
        field.concept[i, d] = o; field.renorm()
        num = (lp - lm) / (2 * eps); rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
        if best is None or rel < best["relerr"]:
            best = {"eps": eps, "fd": round(float(num), 8), "ana": round(ana, 8), "relerr": round(float(rel), 6)}
    rows.append({"row": int(i), **best})
res = {"soft_mode_fd": rows, "max_relerr": max(r["relerr"] for r in rows)}
json.dump(res, open("probe_soft.json", "w"), indent=2)
print(json.dumps(res, indent=2))
