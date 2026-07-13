"""Minimal CPU/GPU examples for intde2_HPC.py."""

import math
import numpy as np
from numba import set_num_threads

from intde2_HPC import (
    cuda_available,
    intdeiini,
    intdeidini,
    make_intdei_cpu,
    make_intdeid_cpu,
    make_intdei_cuda,
    make_intdeid_cuda,
)


# Plain scalar Python functions. The factories compile them for CPU or GPU.
def cauchy_scaled(x, p):
    return 1.0 / (1.0 + p * x * x)


def gaussian_scaled(x, p):
    # Guard against x*x overflow in extreme DE tails.
    if abs(x) > 1.0e154:
        return 0.0
    return math.exp(-p * x * x)


# Precompute tables once and reuse them.
aw_semi = intdeiini(lenaw=8000, tiny=1e-307, eps=1e-12)
aw_whole = intdeidini(lenaw=8000, tiny=1e-307, eps=1e-12)

# ---------------------------------------------------------------------
# CPU: scalar JIT and parallel batch
# ---------------------------------------------------------------------
set_num_threads(8)  # adapt to your CPU

intdei_one, intdei_many = make_intdei_cpu(cauchy_scaled)
value, error = intdei_one(0.0, 1.0, aw_semi)
print("CPU single intdei:", value, error)

n = 10000
a = np.zeros(n)
params = np.linspace(0.5, 2.0, n)
values, errors = intdei_many(a, params, aw_semi)
print("CPU batch first result:", values[0], errors[0])

intdeid_one, intdeid_many = make_intdeid_cpu(gaussian_scaled)
value, error = intdeid_one(1.0, aw_whole)
print("CPU single intdeid:", value, error, "true:", math.sqrt(math.pi))

# ---------------------------------------------------------------------
# NVIDIA GPU: one CUDA thread per independent integral
# ---------------------------------------------------------------------
if cuda_available():
    _, intdei_many_gpu = make_intdei_cuda(cauchy_scaled)
    values_gpu, errors_gpu = intdei_many_gpu(a, params, aw_semi)
    print("GPU batch first result:", values_gpu[0], errors_gpu[0])

    _, intdeid_many_gpu = make_intdeid_cuda(gaussian_scaled)
    values_gpu, errors_gpu = intdeid_many_gpu(params, aw_whole)
    print("GPU whole-line first result:", values_gpu[0], errors_gpu[0])
else:
    print("No CUDA device/runtime detected.")
