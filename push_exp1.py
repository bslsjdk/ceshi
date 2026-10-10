import base64, json, os, urllib.request
TOK = open('/data/workspace/.tok_new').read().strip()
API='https://api.github.com'; REPO='bslsjdk/ceshi'; BR='main'
def req(m,u,b=None):
    r=urllib.request.Request(API+u, data=json.dumps(b).encode() if b else None, method=m)
    r.add_header('Authorization','Bearer '+TOK); r.add_header('Accept','application/vnd.github+json')
    r.add_header('X-GitHub-Api-Version','2022-11-28')
    try: return json.load(urllib.request.urlopen(r))
    except Exception as e:
        print('ERR',m,u,e, getattr(e,'read',lambda:b'')()[:200]); return None
me=req('GET','/user'); print('auth as', (me or {}).get('login'))
ref=req('GET',f'/repos/{REPO}/git/ref/heads/{BR}'); base=ref['object']['sha']
print('base', base[:12])
files=['exp1.py','tf_micro.py','run_exp1.sh','exp1_vn.json','exp1_tf64.json',
       'exp1_tf256.json','exp1_tf1024.json','exp3_sparsity.json','exp1_report.md']
tree=[]
for f in files:
    if not os.path.exists(f): print('skip',f); continue
    b=open(f,'rb').read()
    bl=req('POST',f'/repos/{REPO}/git/blobs',dict(content=base64.b64encode(b).decode(),encoding='base64'))
    if bl: tree.append(dict(path=f,mode='100644',type='blob',sha=bl['sha'])); print('blob',f,len(b))
t=req('POST',f'/repos/{REPO}/git/trees',dict(base_tree=base,tree=tree))
c=req('POST',f'/repos/{REPO}/git/commits',dict(message='exp1-3: VN vs same-parameter Transformer; all three signals fail',tree=t['sha'],parents=[base]))
u=req('PATCH',f'/repos/{REPO}/git/refs/heads/{BR}',dict(sha=c['sha']))
print('commit', c['sha'][:12], 'pushed', bool(u))
