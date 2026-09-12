"""Native reference dispatch. Timings include Python/ctypes dispatch."""
import ctypes as C
from pathlib import Path
import numpy as np
from .quant import Packed
ROOT=Path(__file__).resolve().parents[1]
F=C.POINTER(C.c_float);U=C.POINTER(C.c_uint8)
def fp(x):return x.ctypes.data_as(F)

class CPU:
    def __init__(self,w,x):
        self.w=w; self.x=np.ascontiguousarray(x,dtype=np.float32)
        self.m,self.k=(w.rows,w.cols) if isinstance(w,Packed) else w.shape
        if self.x.ndim!=2 or self.x.shape[1]!=self.k:raise ValueError('input shape mismatch')
        if not np.isfinite(self.x).all():raise ValueError('nonfinite input')
        self.y=np.empty((len(x),self.m),dtype=np.float32)
        self.lib=C.CDLL(str(ROOT/'build/cpu.so'))
        self.lib.ed_quant.argtypes=[U,F,F,F]+[C.c_int]*5
        self.lib.ed_quant.restype=None
        self.lib.ed_float.argtypes=[F,F,F]+[C.c_int]*3
        self.lib.ed_float.restype=None
        if not isinstance(w,Packed):self.w=np.ascontiguousarray(w,dtype=np.float32)
    def run(self):
        w=self.w
        if isinstance(w,Packed):
            self.lib.ed_quant(w.data.ctypes.data_as(U),fp(w.scales),fp(self.x),fp(self.y),self.m,self.k,len(self.x),w.group,w.bits)
        else:self.lib.ed_float(fp(w),fp(self.x),fp(self.y),self.m,self.k,len(self.x))
        return self.y
    def close(self):pass

class Metal(CPU):
    def __init__(self,w,x):
        # Reuse shape/contiguity checks; CPU build is required for validation.
        super().__init__(w,x)
        self.lib=C.CDLL(str(ROOT/'build/metal.so'))
        self.lib.ed_metal_create.argtypes=[C.c_char_p,C.c_void_p,C.c_size_t,F,C.c_size_t,F]+[C.c_int]*5
        self.lib.ed_metal_create.restype=C.c_void_p
        self.lib.ed_metal_run.argtypes=[C.c_void_p,F];self.lib.ed_metal_run.restype=C.c_int
        self.lib.ed_metal_free.argtypes=[C.c_void_p];self.lib.ed_metal_free.restype=None
        if isinstance(w,Packed):data,scales,g,bits=w.data,w.scales,w.group,w.bits
        else:data,scales,g,bits=self.w,np.ones(1,dtype=np.float32),2,32
        self.handle=self.lib.ed_metal_create((ROOT/'native/linear.metal').read_bytes(),data.ctypes.data,data.nbytes,fp(scales),scales.nbytes,fp(self.x),self.m,self.k,len(self.x),g,bits)
        if not self.handle:raise RuntimeError('Metal device or pipeline unavailable')
    def run(self):
        if not self.handle:raise RuntimeError('closed backend')
        if self.lib.ed_metal_run(self.handle,fp(self.y)):raise RuntimeError('Metal command failed')
        return self.y
    def close(self):
        if self.handle:self.lib.ed_metal_free(self.handle);self.handle=None
