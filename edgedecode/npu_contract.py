"""Static contract inspection, NOT a compiler, simulator, or silicon validation."""
import argparse,json
from pathlib import Path

def inspect(spec,contract):
    errors=[]
    for field in ('m','k','n','bits','group'):
        if type(spec.get(field)) is not int or spec[field]<=0:raise ValueError('invalid '+field)
    if spec['bits'] not in contract['weight_bits']:errors.append('unsupported weight precision')
    if spec['group'] not in contract['group_sizes']:errors.append('unsupported scale grouping')
    if contract['activation_dtype']!='f32':errors.append('activation quantization/retraining not implemented')
    if spec['k']%contract['k_multiple']:errors.append('K padding or tail kernel required')
    if contract['native_format']!='edgedecode_v1':errors.append('format conversion or custom operator required')
    return {'compatible_with_declared_contract':not errors,'issues':errors,
            'evidence':'static_hypothetical_contract_check','compiled':False,'cycle_estimate':None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,default=Path('configs/operator.json'));p.add_argument('--contract',type=Path,default=Path('configs/npu-contract.json'));a=p.parse_args()
    print(json.dumps(inspect(json.loads(a.spec.read_text()),json.loads(a.contract.read_text())),indent=2))
if __name__=='__main__':main()
