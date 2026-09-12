"""Controlled synthetic pilot and optional user-supplied operator tensors."""
import argparse, hashlib, itertools, json, os, platform, random, time
from pathlib import Path
import numpy as np
from .quant import pack, unpack, relative_mse
from .backend import CPU, Metal, ROOT
CHOICES=[(b,g) for b in (4,8) for g in (32,64,128)]

def measure(runner,repeats=9,warmup=2):
    for _ in range(warmup):runner.run()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();runner.run();values.append((time.perf_counter_ns()-start)/1e6)
    return {'median_ms':float(np.median(values)),'p25_ms':float(np.percentile(values,25)),
            'p75_ms':float(np.percentile(values,75)),'samples_ms':values}

def random_weight(rng,m,k):
    w=rng.normal(0,1/np.sqrt(k),(m,k)).astype(np.float32)
    # Heterogeneous columns and sparse outliers, deliberately synthetic.
    w[:,::31]*=4
    return w

def operators(backends,out,repeats=9,tensors=None):
    rng=np.random.default_rng(20260910); rows=[]; skipped=[]
    workloads=[]
    if tensors:
        with np.load(tensors,allow_pickle=False) as data:
            w=np.asarray(data['weight'],dtype=np.float32);x=np.asarray(data['activation'],dtype=np.float32)
        if w.ndim!=2 or x.ndim!=2 or w.shape[1]!=x.shape[1] or not np.isfinite(w).all() or not np.isfinite(x).all():
            raise ValueError('NPZ must contain finite weight[M,K], activation[N,K]')
        workloads=[('user_npz',w,x)]
    else:
        for m,k,n in [(256,512,1),(256,512,16),(1024,1024,1),(1024,1024,16)]:
            workloads.append((f'm{m}_k{k}_n{n}',random_weight(rng,m,k),rng.normal(size=(n,k)).astype(np.float32)))
    for name,w,x in workloads:
        ref=x@w.T
        variants=[('f32',w,32,0)]+[(f'q{b}_g{g}',pack(w,b,g),b,g) for b,g in CHOICES]
        jobs=list(itertools.product(backends,variants));random.Random(71).shuffle(jobs)
        for backend,(fmt,weights,bits,group) in jobs:
            try:runner={'cpu':CPU,'metal':Metal}[backend](weights,x)
            except (OSError,RuntimeError) as exc:
                skipped.append({'backend':backend,'workload':name,'format':fmt,'reason':str(exc)});continue
            try:
                y=runner.run().copy()
                numerical=x@(unpack(weights) if bits!=32 else weights).T
                np.testing.assert_allclose(y,numerical,rtol=2e-4,atol=2e-4)
                row={'workload':name,'backend':backend,'format':fmt,'m':len(w),'k':w.shape[1],'n':len(x),
                     'bits':bits,'group':group,'weight_bytes':weights.nbytes,'relative_mse':relative_mse(ref,y),
                     'evidence':'measured_operator','data_origin':'user_npz' if tensors else 'synthetic'}
                row.update(measure(runner,repeats));rows.append(row)
            finally:runner.close()
    result={'rows':rows,'skipped':skipped,'tensor_sha256':hashlib.sha256(Path(tensors).read_bytes()).hexdigest() if tensors else None,
            'timing_scope':'resident inputs; Python dispatch, kernel and synchronous completion; Metal includes output copy; excludes pack/compile/allocation',
            'order':'seeded shuffled configuration blocks; repeated hot inputs; no thermal control or DRAM residency claim'}
    (out/'operators.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def forward(x,weights):
    for i,w in enumerate(weights):
        x=x@w.T
        if i<len(weights)-1:x=np.tanh(x)
    return x

def native_forward(x,weights):
    for i,w in enumerate(weights):
        r=CPU(w,x)
        try:x=r.run().copy()
        finally:r.close()
        if i<len(weights)-1:x=np.tanh(x)
    return x

class Chain:
    def __init__(self,x,weights):self.x=x;self.weights=weights
    def run(self):return native_forward(self.x,self.weights)

def search(out,seeds=(11,23,37),threshold=0.005,repeats=9):
    records=[];all_candidates=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        shapes=[(192,256),(128,192),(64,128)]
        w=[random_weight(rng,m,k) for m,k in shapes]
        # Development and test inputs use distinct RNG seeds, no test-driven selection.
        dev=np.random.default_rng(seed+1000).normal(size=(96,256)).astype(np.float32)
        test=np.random.default_rng(seed+2000).normal(size=(256,256)).astype(np.float32)
        yd,yt=forward(dev,w),forward(test,w)
        packed=[[pack(layer,b,g) for b,g in CHOICES] for layer in w]
        deq=[[unpack(p) for p in layer] for layer in packed]
        costs=[]
        for i,layer in enumerate(packed):
            layer_cost=[]
            order=list(range(len(CHOICES)));random.Random(seed+i).shuffle(order)
            measured={}
            for j in order:
                r=CPU(layer[j],np.ones((1,shapes[i][1]),dtype=np.float32))
                try:measured[j]=measure(r,repeats)['median_ms']
                finally:r.close()
            costs.append([measured[j] for j in range(len(CHOICES))])
        budget=sum(layer.nbytes for layer in w)*0.30
        candidates=[]
        for config in itertools.product(range(6),repeat=3):
            size=sum(packed[i][c].nbytes for i,c in enumerate(config))
            error=relative_mse(yd,forward(dev,[deq[i][c] for i,c in enumerate(config)]))
            latency=sum(costs[i][c] for i,c in enumerate(config))
            candidates.append({'config':list(config),'bytes':size,'dev_relative_mse':error,'predicted_cpu_ms':latency})
        feasible=[c for c in candidates if c['bytes']<=budget and c['dev_relative_mse']<=threshold]
        memory_feasible=[c for c in candidates if c['bytes']<=budget]
        selected={
            'quality_only':min(memory_feasible,key=lambda c:c['dev_relative_mse']),
            'uniform_q4_g128':next(c for c in candidates if c['config']==[2,2,2]),
            'uniform_q8_g64':next(c for c in candidates if c['config']==[4,4,4])}
        if feasible:
            selected['hardware_aware']=min(feasible,key=lambda c:c['predicted_cpu_ms'])
            # Uniform-random quality-and-memory-feasible control, not exactly byte-matched.
            selected['random_feasible']=random.Random(seed).choice(feasible)
        for label,c in selected.items():
            ws=[packed[i][j] for i,j in enumerate(c['config'])]
            pred=forward(test,[unpack(p) for p in ws])
            # Native composition checked independently of search path.
            actual=native_forward(test[:4],ws)
            np.testing.assert_allclose(actual,pred[:4],rtol=4e-4,atol=4e-4)
            row=dict(c,seed=seed,policy=label,test_relative_mse=relative_mse(yt,pred),
                     budget_bytes=budget,threshold=threshold,feasible_count=len(feasible),
                     evidence='synthetic_composed_network',config_names=[f'q{CHOICES[j][0]}_g{CHOICES[j][1]}' for j in c['config']])
            row.update(measure(Chain(test[:1],ws),repeats));records.append(row)
        all_candidates.append({'seed':seed,'layer_costs_ms':costs,'candidates':candidates})
    result={'rows':records,'search_space':216,'threshold':threshold,'budget_fraction_of_f32':0.30,
            'data_origin':'untrained synthetic three-layer tanh network; no language-model accuracy claims',
            'selection':'minimize additive measured CPU operator latency over development-quality feasible configurations; test never used for selection',
            'timing_scope':'composition includes runner creation, ctypes dispatch, output copies and tanh; differs from additive operator proxy'}
    (out/'search.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'candidates.json').write_text(json.dumps(all_candidates,indent=2)+'\n')
    return result

def manifest(out):
    files=[p for d in ('edgedecode','native','configs') for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    result={'schema_version':1,'machine_arch_observed':platform.machine(),'os':platform.platform(),'python':platform.python_version(),
            'numpy':np.__version__,'device_user_report':'Apple M4 Pro, 24 GB (not independently verified by sysctl)',
            'threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
            'sources_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'measurement_notice':'hot-cache pilot; no CPU affinity/frequency lock; no power/area measurement'}
    if (ROOT/'build/build.json').exists():result['native_build']=json.loads((ROOT/'build/build.json').read_text())
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/local'))
    p.add_argument('--backends',nargs='+',choices=['cpu','metal'],default=['cpu'])
    p.add_argument('--repeats',type=int,default=9);p.add_argument('--tensors',type=Path);p.add_argument('--skip-search',action='store_true')
    a=p.parse_args()
    if a.repeats<3:p.error('at least three repeats required')
    a.out.mkdir(parents=True,exist_ok=True)
    manifest(a.out);r=operators(a.backends,a.out,a.repeats,a.tensors)
    print('operators:',len(r['rows']),'skipped:',len(r['skipped']),flush=True)
    if not a.skip_search:
        r=search(a.out,repeats=a.repeats);print('search selections:',len(r['rows']),flush=True)

if __name__=='__main__':main()
