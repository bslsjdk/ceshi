import json, numpy as np
from vn_mobile.core.field import Field
from vn_mobile.core.neuron import build_population

def run(steps=25000, seed=0, omega_std=0.3, omega_mix=None, n=1000):
    pop = build_population(n=n, seed=seed, omega_std=omega_std, omega_mix=omega_mix)
    f = Field(pop, seed=seed)
    ser = []
    for i in range(steps):
        f.tick()
        if (i+1) % 1000 == 0:
            ser.append(f.measure().sync_r)
    a = np.asarray(ser)
    w = a[:len(a)//10*10].reshape(-1,10)
    rng = float((w.max(1)-w.min(1)).max())
    st = f.measure()
    return dict(R_mean=round(float(a[len(a)//2:].mean()),4), R_range_1k=round(rng,4),
                frozen=bool(abs(a[-1]-a[-2])<1e-4), e_mean=round(st.energy_mean,4),
                e_std=round(st.energy_std,4), e_max=round(st.energy_max,4))

print("=== fine single-omega sweep 0.20-0.32 ===")
for w in (0.20,0.22,0.24,0.26,0.28,0.30):
    r = run(omega_std=w); r["omega_std"]=w
    print(json.dumps(r), flush=True)

print("=== layered ===")
for mix in [(0.8,0.2,1.0),(0.8,0.3,1.0),(0.8,0.25,1.0)]:
    r = run(omega_mix=mix); r["mix"]=list(mix)
    print(json.dumps(r), flush=True)
