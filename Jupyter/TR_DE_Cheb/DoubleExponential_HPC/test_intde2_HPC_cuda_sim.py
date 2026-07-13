import math
import numpy as np
from intde2_HPC import (
    intdeini, intdeiini, intdeidini, intdeoini,
    make_intde_cuda, make_intdei_cuda, make_intdeid_cuda, make_intdeo_cuda,
)


def f_finite(x, p):
    return 1.0 / math.sqrt(x)


def f_semi(x, p):
    return 1.0 / (1.0 + p*x*x)


def f_whole(x, p):
    if abs(x) > 1.0e154:
        return 0.0
    return math.exp(-p*x*x)


def f_osc(x, omega, p):
    if x == 0.0:
        return omega
    return math.sin(omega*x)/x


def main():
    eps=1e-11
    tiny=1e-307

    aw=intdeini(4000,tiny,eps)
    _, run=make_intde_cuda(f_finite)
    v,e=run(np.zeros(2),np.ones(2),np.zeros(2),aw,threads_per_block=2)
    print('finite',v,e)
    assert np.max(np.abs(v-2.0)) < 1e-8

    aw=intdeiini(4000,tiny,eps)
    _, run=make_intdei_cuda(f_semi)
    v,e=run(np.zeros(2),np.ones(2),aw,threads_per_block=2)
    print('semi',v,e)
    assert np.max(np.abs(v-math.pi/2)) < 1e-8

    aw=intdeidini(4000,tiny,eps)
    _, run=make_intdeid_cuda(f_whole)
    v,e=run(np.ones(2),aw,threads_per_block=2)
    print('whole',v,e)
    assert np.max(np.abs(v-math.sqrt(math.pi))) < 1e-8

    aw=intdeoini(4000,tiny,eps)
    _, run=make_intdeo_cuda(f_osc)
    v,e=run(np.zeros(2),np.ones(2),np.zeros(2),aw,threads_per_block=2)
    print('osc',v,e)
    assert np.max(np.abs(v-math.pi/2)) < 1e-8
    print('CUDA simulator tests passed')

if __name__=='__main__':
    main()
