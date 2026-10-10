import json, os, subprocess, urllib.request, fnmatch
TOK=open('/data/workspace/.tok_new').read().strip()
API='https://api.github.com'; REPO='bslsjdk/ceshi'; BR='main'
def req(m,u,b=None):
    r=urllib.request.Request(API+u, data=json.dumps(b).encode() if b else None, method=m)
    r.add_header('Authorization','Bearer '+TOK); r.add_header('Accept','application/vnd.github+json')
    r.add_header('X-GitHub-Api-Version','2022-11-28')
    return json.load(urllib.request.urlopen(r))
sha=req('GET',f'/repos/{REPO}/git/ref/heads/{BR}')['object']['sha']
tree=req('GET',f'/repos/{REPO}/git/trees/{sha}?recursive=1')
remote=sorted(i['path'] for i in tree['tree'] if i['type']=='blob')
print('HEAD', sha[:12], 'remote blobs', len(remote))
# local tracked-ish: everything except cache/pycache/large data
skip={'.git','__pycache__','.pytest_cache'}
local=[]
for root,dirs,fs in os.walk('.'):
    dirs[:]=[d for d in dirs if d not in skip and not d.startswith('.')]
    for f in fs:
        p=os.path.relpath(os.path.join(root,f),'.')
        if f.endswith(('.pyc','.npz','.npz.tmp')): continue
        local.append(p)
local=sorted(local)
# only compare the 9 files we pushed (rest of tree predates this turn)
pushed=['exp1.py','tf_micro.py','run_exp1.sh','exp1_vn.json','exp1_tf64.json',
        'exp1_tf256.json','exp1_tf1024.json','exp3_sparsity.json','exp1_report.md']
missing=[f for f in pushed if f not in remote]
print('pushed files present on remote:', len(pushed)-len(missing), '/', len(pushed))
print('missing:', missing)
# byte-identical check via contents API
allok=True
for f in pushed:
    r=req('GET',f'/repos/{REPO}/contents/{f}?ref={sha}')
    import base64
    got=base64.b64decode(r['content'])
    same = got==open(f,'rb').read()
    allok &= same
    print(f'  {f:24s} remote={len(got):7d} identical={same}')
print('ALL BYTE-IDENTICAL:', allok)
