import itertools,json,unittest
from pathlib import Path
import numpy as np
from edgedecode.quant import pack,unpack,relative_mse
from edgedecode.backend import CPU,Metal,ROOT
from edgedecode.architecture import cost,sweep
from edgedecode.experiment import forward,native_forward

class FormatTests(unittest.TestCase):
    def test_zero(self):
        for bits in (4,8):
            p=pack(np.zeros((3,17)),bits,32)
            np.testing.assert_array_equal(unpack(p),0)
    def test_exact_endpoints_and_padding(self):
        w=np.array([[-7,0,7,1,-1]],np.float32)
        p=pack(w,4,8)
        np.testing.assert_array_equal(unpack(p),w)
        self.assertEqual(p.nbytes,8) # 4 bytes packed + 4-byte scale
    def test_storage_formula(self):
        for b,g in itertools.product((4,8),(32,64,128)):
            p=pack(np.ones((7,137)),b,g)
            self.assertEqual(p.nbytes,7*p.padded_cols*b//8+7*(p.padded_cols//g)*4)
    def test_error_bound(self):
        rng=np.random.default_rng(1);w=rng.normal(size=(7,139)).astype(np.float32)
        for b in (4,8):
            p=pack(w,b,32);bound=np.repeat(p.scales,32,axis=1)[:,:139]/2
            self.assertTrue(np.all(np.abs(w-unpack(p))<=bound+1e-6))
    def test_invalid(self):
        for w in [[],[1,2],np.ones((0,2)),np.array([[np.nan]]),np.array([[np.inf]])]:
            with self.assertRaises(ValueError):pack(w)
        for b,g in [(3,32),(4,0),(4,3),(4,True)]:
            with self.assertRaises(ValueError):pack(np.ones((2,2)),b,g)
    def test_rmse_shape(self):
        with self.assertRaises(ValueError):relative_mse([1],[1,2])
    def test_no_reserved_codes(self):
        p=pack(np.arange(-16,16).reshape(2,16),4,32)
        self.assertFalse(np.any((p.data&15)==15));self.assertFalse(np.any((p.data>>4)==15))

@unittest.skipUnless((ROOT/'build/cpu.so').exists(),'build CPU before native tests')
class CPUTests(unittest.TestCase):
    def test_quant_against_dequantized_reference(self):
        rng=np.random.default_rng(4)
        for b,g,k,n in itertools.product((4,8),(32,64),(17,137),(1,3)):
            w=rng.normal(size=(7,k)).astype(np.float32);x=rng.normal(size=(n,k)).astype(np.float32);p=pack(w,b,g)
            r=CPU(p,x)
            np.testing.assert_allclose(r.run(),x@unpack(p).T,rtol=2e-5,atol=2e-5)
    def test_float(self):
        rng=np.random.default_rng(5);w=rng.normal(size=(9,31)).astype(np.float32);x=rng.normal(size=(3,31)).astype(np.float32)
        np.testing.assert_allclose(CPU(w,x).run(),x@w.T,rtol=2e-5,atol=2e-5)
    def test_bad_input(self):
        with self.assertRaises(ValueError):CPU(pack(np.ones((3,4))),np.ones((2,5)))
    def test_composition(self):
        rng=np.random.default_rng(8);w=[pack(rng.normal(size=(7,13)),4,32),pack(rng.normal(size=(3,7)),8,32)];x=rng.normal(size=(4,13)).astype(np.float32)
        np.testing.assert_allclose(native_forward(x,w),forward(x,[unpack(p) for p in w]),rtol=1e-4,atol=1e-4)

class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.p={'bandwidth_GB_s':1,'compute_GFLOP_s':1,'decode_Gweight_s':1,'launch_us':0}
        self.e={'external_pJ_byte':1,'mac_pJ':1,'decode_pJ_weight':1}
    def test_bytes_and_units(self):
        r=cost(2,4,1,4,4,self.p,self.e)
        self.assertEqual(r['weight_bytes'],12)
        self.assertAlmostEqual(r['latency_us'],0.036)
        self.assertAlmostEqual(r['energy_uJ'],52e-6)
    def test_decode_acceleration_cannot_slow(self):
        for n in (1,16,128):
            a=cost(64,64,n,4,32,self.p,self.e);b=cost(64,64,n,4,32,self.p,self.e,decode_scale=2)
            self.assertLessEqual(b['latency_us'],a['latency_us'])
    def test_memory_bound_rejects_compute(self):
        p=dict(self.p,bandwidth_GB_s=0.01)
        a=cost(64,64,1,4,32,p,self.e);b=cost(64,64,1,4,32,p,self.e,compute_scale=2)
        self.assertEqual(a['latency_us'],b['latency_us'])
    def test_bad_profile(self):
        with self.assertRaises(ValueError):cost(2,2,1,4,32,dict(self.p,bandwidth_GB_s=0),self.e)
    def test_sweep_provenance(self):
        conf=json.loads((ROOT/'configs/architecture.json').read_text());rows=sweep(conf)
        self.assertEqual(len(rows),108)
        self.assertTrue(all(r['evidence']=='hypothetical_uncalibrated' for r in rows))
        self.assertTrue(all(not r['energy_change_assessed'] for r in rows))

class MetalTests(unittest.TestCase):
    def test_optional_backend(self):
        if not (ROOT/'build/metal.so').exists():self.skipTest('Metal not built')
        rng=np.random.default_rng(12);w=rng.normal(size=(17,65)).astype(np.float32);x=rng.normal(size=(3,65)).astype(np.float32)
        for weights in [w,pack(w,4,32),pack(w,8,64)]:
            try:r=Metal(weights,x)
            except RuntimeError as exc:self.skipTest(str(exc))
            try:
                expected=x@(unpack(weights) if hasattr(weights,'bits') else weights).T
                np.testing.assert_allclose(r.run(),expected,rtol=3e-5,atol=3e-5)
            finally:r.close()
            with self.assertRaises(RuntimeError):r.run()

if __name__=='__main__':unittest.main()
