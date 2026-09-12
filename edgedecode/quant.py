"""An explicitly specified row-grouped symmetric format; not GGUF/AWQ."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Packed:
    data: np.ndarray
    scales: np.ndarray
    rows: int
    cols: int
    group: int
    bits: int

    @property
    def padded_cols(self):
        return ((self.cols + self.group - 1) // self.group) * self.group

    @property
    def nbytes(self):
        return self.data.nbytes + self.scales.nbytes


def pack(w, bits=4, group=64):
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or min(w.shape) <= 0 or not np.isfinite(w).all():
        raise ValueError('expected nonempty finite matrix')
    if bits not in (4, 8) or type(group) is not int or group < 2 or group % 2:
        raise ValueError('bits must be 4/8 and group a positive even integer >=2')
    m, k = w.shape
    kp = ((k + group - 1) // group) * group
    blocks = np.pad(w, ((0, 0), (0, kp-k))).reshape(m, -1, group)
    qmax = (1 << (bits-1)) - 1
    maximum = np.max(np.abs(blocks), axis=-1)
    scales = np.where(maximum > 0, maximum / qmax, 1).astype(np.float32)
    signed = np.clip(np.rint(blocks / scales[..., None]), -qmax, qmax)
    codes = (signed.astype(np.int16) + qmax).astype(np.uint8).ravel()
    data = codes if bits == 8 else codes[0::2] | (codes[1::2] << 4)
    return Packed(np.ascontiguousarray(data), np.ascontiguousarray(scales), m, k, group, bits)


def unpack(p):
    if p.bits == 4:
        codes = np.empty(p.data.size*2, dtype=np.uint8)
        codes[0::2], codes[1::2] = p.data & 15, p.data >> 4
    else:
        codes = p.data
    qmax = (1 << (p.bits-1)) - 1
    blocks = codes.astype(np.int16).reshape(p.rows, -1, p.group) - qmax
    return (blocks * p.scales[..., None]).reshape(p.rows, p.padded_cols)[:, :p.cols].astype(np.float32)


def relative_mse(reference, actual):
    ref, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if ref.shape != actual.shape:
        raise ValueError('shape mismatch')
    return float(np.sum((ref-actual)**2) / max(np.sum(ref**2), 1e-30))
