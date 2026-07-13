import math
import numpy as np

from intde2_HPC import (
    intdeini, intdeiini, intdeidini, intdeoini,
    make_intde_cpu, make_intdei_cpu, make_intdeid_cpu, make_intdeo_cpu,
)


def f_finite(x, p):
    return 1.0 / math.sqrt(x)


def f_semi(x, p):
    return 1.0 / (1.0 + p * x * x)


def f_whole(x, p):
    # avoid overflow in x*x in far DE tails
    ax = abs(x)
    if ax > 1.0e154:
        return 0.0
    return math.exp(-p * x * x)


def f_osc(x, omega, p):
    if x == 0.0:
        return omega
    return math.sin(omega * x) / x


def main():
    eps = 1e-13
    tiny = 1e-307

    aw = intdeini(8000, tiny, eps)
    one, batch = make_intde_cpu(f_finite)
    v,e = one(0.0,1.0,0.0,aw)
    print('finite single',v,e)
    assert abs(v-2.0) < 1e-11
    aa=np.zeros(8); bb=np.ones(8); pp=np.zeros(8)
    vv,ee=batch(aa,bb,pp,aw)
    assert np.max(np.abs(vv-2.0)) < 1e-11

    aw = intdeiini(8000, tiny, eps)
    one, batch = make_intdei_cpu(f_semi)
    v,e = one(0.0,1.0,aw)
    print('semi single',v,e)
    assert abs(v-math.pi/2) < 1e-11
    aa=np.zeros(8); pp=np.ones(8)
    vv,ee=batch(aa,pp,aw)
    assert np.max(np.abs(vv-math.pi/2)) < 1e-11

    aw = intdeidini(8000, tiny, eps)
    one, batch = make_intdeid_cpu(f_whole)
    v,e = one(1.0,aw)
    print('whole single',v,e)
    assert abs(v-math.sqrt(math.pi)) < 1e-11
    pp=np.ones(8)
    vv,ee=batch(pp,aw)
    assert np.max(np.abs(vv-math.sqrt(math.pi))) < 1e-11

    aw = intdeoini(8000, tiny, eps)
    one, batch = make_intdeo_cpu(f_osc)
    v,e = one(0.0,1.0,0.0,aw)
    print('osc single',v,e)
    assert abs(v-math.pi/2) < 1e-10
    aa=np.zeros(8); ww=np.ones(8); pp=np.zeros(8)
    vv,ee=batch(aa,ww,pp,aw)
    assert np.max(np.abs(vv-math.pi/2)) < 1e-10

    print('CPU HPC tests passed')

if __name__=='__main__':
    main()
