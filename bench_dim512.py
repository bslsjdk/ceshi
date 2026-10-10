"""DIM=512 full-system prototype: energy + phase + wave diffusion + state.
No [N,D,D] fused cache -- low-rank two-hop instead."""
import numpy as np, time, json, gc
from scipy.sparse import csr_matrix

TWO_PI = 2.0*np.pi

def build(n, dim, rank, deg, seed=0, zone_share=False):
    rng = np.random.default_rng(seed)
    # CSR adjacency, undirected-ish random, degree deg
    src = np.repeat(np.arange(n), deg)
    dst = rng.integers(0, n, size=n*deg)
    w   = rng.normal(0, 0.5, size=n*deg).astype(np.float32)
    d   = rng.normal(size=(n,8)).astype(np.float32)
    d  /= (np.linalg.norm(d,axis=1,keepdims=True)+1e-8)
    # direction cosine per edge
    dij = np.einsum('ij,ij->i', d[src], d[dst]).astype(np.float32)
    A1 = csr_matrix((np.ones(n*deg,np.float32),(src,dst)), shape=(n,n))
    Aw = csr_matrix(((w*dij).astype(np.float32),(src,dst)), shape=(n,n))
    degi = np.maximum(np.asarray(A1.sum(axis=1)).ravel(), 1.0).astype(np.float32)
    n_ad = 20 if zone_share else n
    zof  = (np.arange(n) % 20) if zone_share else None
    return dict(
        n=n, dim=dim, rank=rank, degi=degi, A1=A1, Aw=Aw, zof=zof,
        e=np.full(n,0.3,np.float32), phi=rng.uniform(0,TWO_PI,n).astype(np.float32),
        h=np.zeros((n,dim),np.float32),
        W=rng.normal(0,1/np.sqrt(dim),size=(dim,dim)).astype(np.float32),
        A=rng.normal(0,0.01,size=(n_ad,rank,dim)).astype(np.float32),
        B=rng.normal(0,0.01,size=(n_ad,dim,rank)).astype(np.float32),
        bias=np.zeros(dim,np.float32),
        omega=rng.normal(0,1.0,n).astype(np.float32),
        lam=0.1, kap=0.25, etarget=0.3, K=0.5, alpha=0.9,
    )

def step(s):
    n,dim = s['n'], s['dim']
    # --- energy: g = sum_j w_ij e_j cos(phi_j - phi_i) d_ij ---
    cs = np.cos(s['phi']); sn = np.sin(s['phi'])
    g = (s['Aw'] @ (s['e']*cs))*cs + (s['Aw'] @ (s['e']*sn))*sn
    s['e'] = ((1-s['lam'])*s['e'] + s['lam']*s['etarget']
              + s['kap']*np.tanh(g/np.sqrt(s['degi']))
              + np.float32(0.001)*np.random.default_rng().normal(0,1,n).astype(np.float32))
    # --- phase: Kuramoto ---
    cs = np.cos(s['phi']); sn = np.sin(s['phi'])
    s['phi'] = (s['phi'] + s['omega']
                + (s['K']/s['degi'])*(cs*(s['A1']@sn) - sn*(s['A1']@cs))) % TWO_PI
    # --- state: low-rank two-hop + shared W ---
    x = s['h'].mean(axis=1, keepdims=True)  # neighbour-mean proxy
    x = np.repeat(x, dim, axis=1)
    Ai = s['A'] if s['zof'] is None else s['A'][s['zof']]
    Bi = s['B'] if s['zof'] is None else s['B'][s['zof']]
    Ax  = np.einsum('nrd,nd->nr', Ai, x, optimize=True)
    BAx = np.einsum('ndr,nr->nd', Bi, Ax, optimize=True)
    s['h'] = (s['alpha']*s['h']
              + (1-s['alpha'])*np.tanh(x @ s['W'].T + BAx + s['bias'])).astype(np.float32)

def mem(s):
    t = s['e'].nbytes+s['phi'].nbytes+s['h'].nbytes+s['W'].nbytes
    t += s['A'].nbytes+s['B'].nbytes+s['bias'].nbytes+s['omega'].nbytes+s['degi'].nbytes
    t += s['A1'].data.nbytes+s['A1'].indices.nbytes+s['Aw'].data.nbytes+s['Aw'].indices.nbytes
    return t

def run(n, dim, rank, deg=8, reps=5, zone_share=True):
    s = build(n, dim, rank, deg, zone_share=zone_share)
    m = mem(s)
    step(s)
    t0=time.perf_counter()
    for _ in range(reps): step(s)
    dt=(time.perf_counter()-t0)/reps
    pv = 2*rank*dim
    param = (20*pv) if zone_share else (n*pv)
    del s; gc.collect()
    return dict(N=n, DIM=dim, R=rank, mem_MB=round(m/1e6,1),
                ms_step=round(dt*1e3,2), param_total=param,
                param_per_VN=pv)

print(f"{'N':>7} {'DIM':>5} {'R':>3} | {'mem(MB)':>9} {'ms/step':>9} | {'总参数':>12} {'param/VN':>9} | judge")
print("-"*80)
rows=[]
for n,dim,rank in [(1000,8,4),(1000,128,16),(1000,512,16),(10000,512,16),(10000,512,4),(10000,256,16)]:
    try:
        r = run(n,dim,rank)
        rows.append(r)
        j = "OK" if r['ms_step']<500 else "OVER"
        print(f"{n:>7} {dim:>5} {rank:>3} | {r['mem_MB']:>9.1f} {r['ms_step']:>9.2f} | {r['param_total']/1e6:>11.1f}M {r['param_per_VN']:>9} | {j}")
    except MemoryError:
        print(f"{n:>7} {dim:>5} {rank:>3} | MemoryError")
json.dump(rows, open('/data/workspace/ceshi/dim512_bench.json','w'), indent=1)
