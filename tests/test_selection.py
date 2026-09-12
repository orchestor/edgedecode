import json,unittest
from pathlib import Path
from edgedecode.npu_contract import inspect
from edgedecode.backend import ROOT

class ContractTests(unittest.TestCase):
    def test_no_false_compilation_claim(self):
        c=json.loads((ROOT/'configs/npu-contract.json').read_text())
        r=inspect({'m':32,'k':33,'n':1,'bits':4,'group':32},c)
        self.assertFalse(r['compatible_with_declared_contract'])
        self.assertFalse(r['compiled']);self.assertIsNone(r['cycle_estimate']);self.assertEqual(len(r['issues']),5)
    def test_compatible_only_with_hypothetical_contract(self):
        c={'weight_bits':[4],'group_sizes':[32],'activation_dtype':'f32','k_multiple':32,'native_format':'edgedecode_v1'}
        self.assertTrue(inspect({'m':32,'k':32,'n':1,'bits':4,'group':32},c)['compatible_with_declared_contract'])
    def test_invalid_spec(self):
        with self.assertRaises(ValueError):inspect({}, {})

class CommittedEvidenceTests(unittest.TestCase):
    def test_selected_configuration_satisfies_development_constraints(self):
        path=ROOT/'results/pilot/search.json'
        if not path.exists():self.skipTest('pilot not included')
        for r in json.loads(path.read_text())['rows']:
            if r['policy']=='hardware_aware':
                self.assertLessEqual(r['dev_relative_mse'],r['threshold'])
                self.assertLessEqual(r['bytes'],r['budget_bytes'])
    def test_optimizer_matches_exhaustive_candidates(self):
        p=ROOT/'results/pilot'
        if not (p/'search.json').exists():self.skipTest('pilot not included')
        search=json.loads((p/'search.json').read_text());candidates=json.loads((p/'candidates.json').read_text())
        for r in search['rows']:
            if r['policy']!='hardware_aware':continue
            allc=next(c['candidates'] for c in candidates if c['seed']==r['seed'])
            feasible=[c for c in allc if c['bytes']<=r['budget_bytes'] and c['dev_relative_mse']<=r['threshold']]
            self.assertEqual(r['predicted_cpu_ms'],min(c['predicted_cpu_ms'] for c in feasible))
    def test_pilot_timing_samples(self):
        path=ROOT/'results/pilot/operators.json'
        if not path.exists():self.skipTest('pilot not included')
        for r in json.loads(path.read_text())['rows']:
            self.assertEqual(r['evidence'],'measured_operator');self.assertEqual(len(r['samples_ms']),9)
            self.assertTrue(all(v>0 for v in r['samples_ms']))

if __name__=='__main__':unittest.main()
