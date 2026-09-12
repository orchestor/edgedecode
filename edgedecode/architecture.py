"""Ideal-overlap scenarios; never label these numbers as measured PPA."""
import argparse,json,itertools
from pathlib import Path

def cost(m,k,n,bits,group,p,energy,decode_scale=1,compute_scale=1):
    if any(type(v) is not int or v<=0 for v in (m,k,n,group)) or bits not in (4,8):raise ValueError('invalid shape/format')
    if any(p[key]<=0 for key in ['bandwidth_GB_s','compute_GFLOP_s','decode_Gweight_s']) or decode_scale<=0 or compute_scale<=0:raise ValueError('invalid throughput')
    ng=(k+group-1)//group;kp=ng*group
    wb=m*kp*bits/8+4*m*ng
    transfer=wb+4*n*(k+m)
    mem=transfer/(p['bandwidth_GB_s']*1e9)
    comp=2*m*k*n/(p['compute_GFLOP_s']*1e9*compute_scale)
    decode=m*kp/(p['decode_Gweight_s']*1e9*decode_scale)
    stages={'memory':mem,'compute':comp,'decode':decode}
    seconds=max(stages.values())+p['launch_us']*1e-6
    epj=transfer*energy['external_pJ_byte']+m*k*n*energy['mac_pJ']+m*kp*energy['decode_pJ_weight']
    return {'latency_us':seconds*1e6,'energy_uJ':epj*1e-6,'weight_bytes':wb,
            'bottleneck':max(stages,key=stages.get),'stage_us':{k:v*1e6 for k,v in stages.items()}}

def sweep(config):
    rows=[];energy=config['energy_assumptions']
    for p,n,bits,g in itertools.product(config['profiles'],[1,16,128],[4,8],[32,128]):
        base=cost(1024,1024,n,bits,g,p,energy)
        for feature,ds,cs in [('baseline',1,1),('double_compute',1,2),('double_decode',2,1)]:
            change=cost(1024,1024,n,bits,g,p,energy,ds,cs)
            speedup=base['latency_us']/change['latency_us']
            # S/(1+a)>1 => normalized additional area a<S-1, assuming equal frequency.
            rows.append(dict(profile=p['name'],n=n,bits=bits,group=g,feature=feature,**change,
                             speedup=speedup,max_extra_area_fraction_for_perf_per_area=speedup-1,
                             energy_change_assessed=False,evidence='hypothetical_uncalibrated'))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/architecture.json'));p.add_argument('--out',type=Path,default=Path('results/local/architecture.json'));a=p.parse_args()
    conf=json.loads(a.config.read_text());a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'assumptions':conf,'rows':sweep(conf)},indent=2)+'\n')
    print('scenario rows:',108)
if __name__=='__main__':main()
