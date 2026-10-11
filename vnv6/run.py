"""Driver: baseline -> fuse -> cluster -> pack -> fine-tune."""
from __future__ import annotations

import numpy as np

from vn_mobile.io.adam import Adam
from vnv6.compress import cluster, cluster_report, factorize, fuse
from vnv6.model import ConceptNet
from vnv6.packed import PackedNet
from vnv6.train import evaluate


def finetune(net: PackedNet, tr, va, steps, lr=3e-3, val_every=25000,
             train_B=False, clip=5.0):
    opt = Adam(lr=lr)
    hist = []
    run = np.zeros(val_every, np.float32)
    for t in range(steps):
        i = t % (tr.size - 2)
        cur, prev, tgt = int(tr[i + 1]), int(tr[i]), int(tr[i + 2])
        net.rebuild()
        f = net.forward(cur, prev)
        f['cur'], f['prev'] = cur, prev
        run[t % val_every] = net.loss(f['logits'], tgt)
        grads = net.backward(f, tgt)
        pairs = net.step_pairs(grads, train_B=train_B)
        opt.clip(pairs, clip)
        opt.step(pairs)
        if (t + 1) % val_every == 0:
            net.rebuild()
            rec = dict(step=t + 1,
                       train_bits=round(float(run.mean()) / np.log(2), 4))
            rec.update(evaluate(net, va))
            hist.append(rec)
    net.rebuild()
    return hist


def pipeline(base: ConceptNet, tr, va, k: int, r: int = 8,
             hi: float = 0.99, mode: str = 'pca', steps: int = 50000,
             lr: float = 3e-3, train_B: bool = False):
    """Full two-level compression of ``base``.  Returns report + net."""
    c = base.concepts()
    fused, groups = fuse(c, hi)
    lab = cluster(fused, k)
    rep = dict(n_base=int(c.shape[0]),
               n_fused=int(fused.shape[0]),
               fused_groups=int(sum(1 for g in groups if len(g) > 1)),
               hi=hi)
    rep['fuse'] = dict(n_removed=int(c.shape[0] - fused.shape[0]))
    rep['cluster'] = cluster_report(fused, lab)
    fac = factorize(fused, lab, r, mode)
    rep['pack'] = dict(r=r, mode=mode,
                       recon_rel_err=fac['recon_rel_err'],
                       resid_var_kept=fac['resid_var_kept'])

    net = PackedNet(base, lab, fac['B'], fac['m'], fac['P'])
    rep['params'] = dict(baseline_block=int(c.size),
                         packed_block=int(net.block_params()),
                         frozen_basis=int(net.frozen_params()),
                         head=int(base.out.size + base.bias.size),
                         emb=int(base.emb.size))
    if steps > 0:
        rep['finetune'] = finetune(net, tr, va, steps, lr=lr, train_B=train_B)
    return rep, net
