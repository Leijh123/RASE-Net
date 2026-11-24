"""Immutable recipes, complete recovery, patient-aware metrics and bounded storage."""
from pathlib import Path
import contextlib, functools, gzip, hashlib, json, math, os, random, re, time, io, sqlite3
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
METRICS = ['dice','iou','precision','recall','hd95','assd','nsd']

def sha(path):
    p=Path(path);s=p.stat()
    return _sha_cached(str(p.resolve()),s.st_size,s.st_mtime_ns,s.st_ino)

@functools.lru_cache(maxsize=256)
def _sha_cached(path,size,mtime,inode):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');torch.save(x,t);t.replace(p)

def save_cache(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with gzip.open(t,'wb',compresslevel=1) as f:torch.save(x,f)
    t.replace(p)

def load(p):
    with open(p,'rb') as f:compressed=f.read(2)==b'\x1f\x8b'
    if compressed:
        with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)
    return torch.load(p,map_location='cpu',weights_only=False)
def seed(n): random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if x['cuda']: torch.cuda.set_rng_state_all(x['cuda'])

def state_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b

@contextlib.contextmanager
def preserved_rng(n=None):
    x=rng_state()
    try:
        if n is not None: seed(n)
        yield
    finally: restore_rng(x)

def amp(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()

def patient(item):
    if item.get('patient_id'): return item['patient_id']
    m=re.match(r'^(ABD_\d+|amos_\d+|word_\d+|patient\d+)(?:_|$)',Path(item['image']).stem,re.I)
    if not m: raise ValueError('Unrecognized patient: '+item['image'])
    return m.group(1)

class Stream:
    """No prefetch: sampler cursor + global RNG recover the next exact batch."""
    def __init__(self,n,seed_value):
        if n<1: raise ValueError('Empty training stream')
        self.n=n;self.generator=np.random.RandomState(seed_value);self.order=[];self.cursor=0
    def next(self,batch):
        result=[]
        while len(result)<batch:
            if self.cursor==len(self.order):self.order=self.generator.permutation(self.n).tolist();self.cursor=0
            take=min(batch-len(result),len(self.order)-self.cursor)
            result.extend(self.order[self.cursor:self.cursor+take]);self.cursor+=take
        return result
    def state(self):return dict(n=self.n,order=self.order,cursor=self.cursor,rng=self.generator.get_state())
    def restore(self,x):
        assert x['n']==self.n
        self.order=x['order'];self.cursor=x['cursor'];self.generator.set_state(x['rng'])

class Lock:
    def __init__(self,path):self.path=Path(path)
    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open('a+')
        fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB);return self
    def __exit__(self,*a):self.file.close()

def finite_step(loss,model,opt,clip=1.):
    if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss; no skipped batch')
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError('Missing or non-finite gradients; recovery required')
    if clip is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    opt.step();opt.zero_grad(set_to_none=True)

def checkpoint(folder,model,opt,scheduler,stream,state,identity,best=False):
    folder=Path(folder);state=dict(state)
    if best:
        slot=1-int(state.get('best_slot',1));state['best_slot']=slot
        save(folder/f'best_slot{slot}.pt',dict(model=model.state_dict(),state=state,identity=identity))
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                 sampler=stream.state(),rng=rng_state(),state=state,identity=identity)
    save(folder/'latest.pt',payload)
    return state

def resume(folder,model,opt,scheduler,stream,identity):
    p=Path(folder)/'latest.pt'
    if not p.exists():return dict(step=0,optimizer_updates=0,best=-1.,best_slot=1,history=[])
    x=load(p)
    if x['identity']!=identity:raise ValueError('Checkpoint recipe/input identity differs; use a new run directory')
    model.load_state_dict(x['model'],strict=True);opt.load_state_dict(x['optimizer']);scheduler.load_state_dict(x['scheduler'])
    stream.restore(x['sampler']);restore_rng(x['rng']);return x['state']

def metrics(pred,gt):
    from legacy_metrics import seg_metrics
    if not np.asarray(gt).any():raise ValueError('Only GT-positive queries are evaluated')
    # Serialization alone maps NaN to null. Numerical formulas are unchanged.
    return {k:float(v) if np.isfinite(v) else None for k,v in seg_metrics(np.asarray(pred),np.asarray(gt)).items()}

def average(rows,keys=METRICS):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        means={k:float(np.nanmean([np.nan if r[k] is None else r[k] for r in rows])) for k in keys}
    return {k:v if np.isfinite(v) else None for k,v in means.items()}

def summarize(rows,rare):
    names=sorted({r['organ_name'] for r in rows});cases=sorted({r['patient'] for r in rows})
    per={n:dict(n=sum(r['organ_name']==n for r in rows),**average([r for r in rows if r['organ_name']==n])) for n in names}
    return dict(n=len(rows),patients=len(cases),slices=len({r['image'] for r in rows}),mean=average(rows),
                macro=average(list(per.values())),
                miou=average(list(per.values()))['iou'],per_organ=per,
                rare=average([r for r in rows if r['organ_name'] in rare]),
                other=average([r for r in rows if r['organ_name'] not in rare]),
                per_patient={c:average([r for r in rows if r['patient']==c]) for c in cases},
                valid_count_per_metric={k:int(sum(r[k] is not None and np.isfinite(r[k]) for r in rows)) for k in METRICS})

class TensorStore:
    """Lossless transactional tensors in one file: avoids per-slice file quotas."""
    def __init__(self,root,identity,read_only=False):
        root=Path(root)
        self.read_only=read_only
        if read_only:
            self.conn=sqlite3.connect((root/'cache.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
            old=self.conn.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if not old or json.loads(old[0])!=identity:
                self.conn.close();raise ValueError('Read-only cache identity mismatch')
            return
        root.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(root/'cache.sqlite3')
        self.conn.execute('PRAGMA journal_mode=DELETE');self.conn.execute('PRAGMA synchronous=FULL')
        self.conn.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS payloads (key TEXT PRIMARY KEY,value BLOB NOT NULL)')
        value=json.dumps(identity,sort_keys=True,ensure_ascii=False)
        old=self.conn.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if old and old[0]!=value:self.conn.close();raise ValueError('Cache protocol/source/checkpoint identity changed')
        with self.conn:self.conn.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)",(value,))
    def get(self,key):
        row=self.conn.execute('SELECT value FROM payloads WHERE key=?',(str(key),)).fetchone()
        if row is None:return None
        return torch.load(io.BytesIO(gzip.decompress(row[0])),map_location='cpu',weights_only=False)
    def put(self,key,payload):
        if self.read_only:raise ValueError('Immutable reused Base cache')
        raw=io.BytesIO();torch.save(payload,raw);blob=gzip.compress(raw.getvalue(),compresslevel=1,mtime=0)
        if self.conn.execute('SELECT 1 FROM payloads WHERE key=?',(str(key),)).fetchone():raise ValueError('Refusing to overwrite an existing slice')
        with self.conn:self.conn.execute('INSERT INTO payloads VALUES (?,?)',(str(key),blob))
    def close(self):self.conn.close()

def auc(scores,labels):
    from scipy.stats import rankdata
    s=np.asarray(scores);y=np.asarray(labels,bool);p=y.sum();n=(~y).sum()
    if not p or not n:raise ValueError('AUROC requires both classes')
    return float((rankdata(s,method='average')[y].sum()-p*(p+1)/2)/(p*n))

def paired_patient_statistics(left,right):
    from scipy.stats import wilcoxon
    a=left['summary']['per_patient'];b=right['summary']['per_patient'];assert set(a)==set(b)
    delta=np.asarray([a[p]['dice']-b[p]['dice'] for p in sorted(a)])
    rng=np.random.RandomState(42);boot=delta[rng.randint(len(delta),size=(10000,len(delta)))].mean(1)
    return dict(unit='patient-aggregated 2D positive-query Dice; not 3D Dice',n=len(delta),
                mean_delta=float(delta.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                wilcoxon_p=float(wilcoxon(delta,method='auto').pvalue) if np.any(delta) else 1.,
                warning='Six validation patients; descriptive uncertainty, not independent test generalization.')
