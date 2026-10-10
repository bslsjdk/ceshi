import numpy as np

def probe(n, dim, rank=4, deg=8):
    state = n*dim*4
    other = n*4*3 + n*8            # e, phi, direction
    W     = dim*dim*4              # shared
    A     = n*rank*dim*4
    B     = n*dim*rank*4
    WM    = n*dim*dim*4            # *** fused cache ***
    csr   = n*deg*4*2
    return dict(state=state, other=other, W=W, AB=A+B, WM=WM, csr=csr,
                total=state+other+W+A+B+WM+csr)

def fmt(b):
    return f"{b/1e6:>8.2f}M" if b < 1e9 else f"{b/1e9:>7.2f}G"

print(f"{'N':>7} {'DIM':>5} | {'state':>9} {'WM[N,d,d]':>10} {'A+B':>9} {'TOTAL':>9}  2GB?")
print("-"*68)
for n in (1000, 10000, 100000, 1000000):
    for dim in (8, 64, 128, 256, 512):
        m = probe(n, dim)
        ok = "OK" if m['total'] < 2e9 else "OOM"
        print(f"{n:>7} {dim:>5} | {fmt(m['state'])} {fmt(m['WM'])} {fmt(m['AB'])} {fmt(m['total'])}  {ok}")
    print()
