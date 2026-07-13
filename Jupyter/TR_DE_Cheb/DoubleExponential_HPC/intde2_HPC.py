# Original C files for DE quadrature:
# Takuya Ooura, https://www.kurims.kyoto-u.ac.jp/~ooura/intde.html
#
# 2025 (c) Hanc, Hancova (jozef.hanc@upjs.sk)
# Python conversion, whole-line extension, and HPC implementation
# Ver: Jan-2025
"""
High-performance CPU and NVIDIA-GPU backends for the table-based INTDE2
double-exponential quadrature algorithms.

Supported DE transformations
----------------------------
Finite interval ``(a, b)`` — tanh-sinh::

    x(t) = (a + b)/2
           + (b - a)/2 * tanh((pi/2) * sinh(t))

Nonoscillatory semi-infinite interval ``(a, +infinity)``::

    x(t) = a + exp((pi/2) * sinh(t))

Nonoscillatory whole line ``(-infinity, +infinity)``::

    x(t) = sinh((pi/2) * sinh(t))

Oscillatory semi-infinite interval ``(a, +infinity)``::

    Ooura's Fourier-type DE transformation, represented by paired
    abscissae x_minus and x_plus and weights beta and pi - beta.
    The exact mathematical definitions are documented in ``intde2.py``.

Parallelization design
----------------------
The original INTDE2 algorithms are adaptive and branch-heavy.  The evaluation
of one integral is therefore serial, but compiled by Numba.  Reliable
parallelism is applied across a batch of independent integrals:

* CPU: one ``numba.prange`` iteration per integral, compiled with
  ``@njit(parallel=True)``;
* NVIDIA GPU: one CUDA thread per independent integral.

This layout is intended for parameter sweeps, bootstrap runs, Monte Carlo
studies, simulation experiments, and evaluation of many related integrals.
A single adaptive integral generally does not contain enough independent work
to utilize a GPU efficiently.

The module expects ``intde2.py`` in the same directory.  Its table
initialization routines are reused because table construction is normally
performed only once and is not the computational bottleneck.

Factory interfaces
------------------
CPU factories return ``(single, batch)``.  The supplied scalar integrand is
compiled automatically by Numba::

    intdei_one, intdei_many = make_intdei_cpu(f)

CUDA factories return ``(kernel, batch_host)``.  ``kernel`` is the raw CUDA
kernel for device-array workflows; ``batch_host`` is a convenience launcher
for NumPy host arrays::

    intdei_kernel, intdei_many_gpu = make_intdei_cuda(f)

Integrand signatures
--------------------
Finite, semi-infinite, and whole-line routines::

    f(x, param) -> float

Oscillatory routine::

    f(x, omega, param) -> float

All work arrays and calculations use IEEE float64.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numba import njit, prange

try:
    from numba import cuda
except Exception:  # pragma: no cover - depends on local CUDA installation
    cuda = None

from intde2 import (
    intdeini as _intdeini_reference,
    intdeiini as _intdeiini_reference,
    intdeoini as _intdeoini_reference,
)

__all__ = [
    "intdeini",
    "intdeiini",
    "intdeidini",
    "intdeoini",
    "cuda_available",
    "make_intde_cpu",
    "make_intdei_cpu",
    "make_intdeid_cpu",
    "make_intdeo_cpu",
    "make_intde_cuda",
    "make_intdei_cuda",
    "make_intdeid_cuda",
    "make_intdeo_cuda",
]


def _as_aw(values) -> np.ndarray:
    """Return a contiguous float64 INTDE2 work array."""
    return np.ascontiguousarray(values, dtype=np.float64)


def intdeini(lenaw: int = 8000, tiny: float = 1.0e-307,
             eps: float = 1.0e-15) -> np.ndarray:
    """Initialize the tanh-sinh table for integration over ``(a, b)``.

    DE transform::

        x(t) = (a + b)/2
               + (b - a)/2 * tanh((pi/2) * sinh(t))
    """
    return _as_aw(_intdeini_reference(lenaw, tiny, eps))


def intdeiini(lenaw: int = 8000, tiny: float = 1.0e-307,
              eps: float = 1.0e-15) -> np.ndarray:
    """Initialize the DE table for ``(a, +infinity)``.

    DE transform::

        x(t) = a + exp((pi/2) * sinh(t))
    """
    return _as_aw(_intdeiini_reference(lenaw, tiny, eps))


def intdeoini(lenaw: int = 8000, tiny: float = 1.0e-307,
              eps: float = 1.0e-15) -> np.ndarray:
    """Initialize Ooura's Fourier-type DE table for oscillatory tails.

    The table represents paired abscissae

        x_minus = a + alpha(tau)/abs(omega)
        x_plus  = a + (alpha(tau) + pi*tau)/abs(omega)

    with complementary internal weights ``beta(tau)`` and
    ``pi - beta(tau)``.  See ``intde2.py`` for the full formulas.
    """
    return _as_aw(_intdeoini_reference(lenaw, tiny, eps))


def intdeidini(lenaw: int = 8000, tiny: float = 1.0e-307,
               eps: float = 1.0e-15) -> np.ndarray:
    """Initialize the DE table for ``(-infinity, +infinity)``.

    DE transform::

        x(t) = sinh((pi/2) * sinh(t))

        dx/dt = (pi/2) * cosh(t) * cosh((pi/2) * sinh(t))

    Each table record contains four values:

        tail indicator, positive node x, auxiliary weight, quadrature weight.

    The integration routines evaluate both ``f(-x, param)`` and
    ``f(x, param)``.
    """
    if lenaw <= 1000:
        raise ValueError("lenaw must be greater than 1000")
    if not (0.0 < tiny < 1.0):
        raise ValueError("tiny must satisfy 0 < tiny < 1")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must satisfy 0 < eps < 1")

    efs = 0.1
    hoff = 11.0

    aw = np.zeros(lenaw, dtype=np.float64)

    tinyln = -math.log(tiny)
    epsln = 1.0 - math.log(efs * eps)
    h0 = hoff / epsln
    ehp = math.exp(h0)
    ehm = 1.0 / ehp

    # Prevent overflow of the increasing Jacobian weights.
    logmax = math.log(np.finfo(np.float64).max)
    log2 = math.log(2.0)

    aw[2] = eps
    aw[3] = math.exp(-ehm * epsln)
    aw[4] = math.sqrt(efs * eps)

    noff = 5
    aw[noff] = 0.0
    aw[noff + 1] = 4.0 * h0
    aw[noff + 2] = 0.5 * math.pi * h0

    h = 2.0
    nk = 0
    k = noff + 4

    while True:
        t = 0.5 * h

        while True:
            em = math.exp(h0 * t)
            ep = 0.25 * math.pi * em
            em = 0.25 * math.pi / em
            j = k

            while True:
                s = ep - em
                # log(cosh(s)), stable for large positive s.
                log_chs = s + math.log1p(math.exp(-2.0 * s)) - log2
                log_aux = log_chs + math.log(4.0 * h0)
                log_quad = log_chs + math.log((ep + em) * h0)

                if not (
                    ep < tinyln
                    and log_aux < logmax
                    and log_quad < logmax
                    and j <= lenaw - 4
                ):
                    break

                u = math.exp(s)
                v = 1.0 / u
                x = 0.5 * (u - v)
                chs = 0.5 * (u + v)

                aw[j] = v
                aw[j + 1] = x
                aw[j + 2] = chs * (4.0 * h0)
                aw[j + 3] = chs * ((ep + em) * h0)

                ep *= ehp
                em *= ehm
                j += 4

            t += h
            k += nk
            if not (t < 1.0):
                break

        h *= 0.5

        if nk == 0:
            if j > lenaw - 8:
                j -= 4
            nk = j - noff
            k += nk
            aw[1] = float(nk)

        if not (2 * k - noff - 4 <= lenaw):
            break

    aw[0] = float(k - 4)
    return aw


def cuda_available() -> bool:
    """Return whether a usable CUDA runtime/device is visible."""
    return bool(cuda is not None and cuda.is_available())


# ---------------------------------------------------------------------------
# CPU factories: one compiled scalar integrator + a prange batch wrapper
# ---------------------------------------------------------------------------


def make_intde_cpu(f: Callable, *, fastmath: bool = False):
    """Build compiled CPU integrators for ``(a, b)``.

    DE transform::

        x(t) = (a + b)/2
               + (b - a)/2 * tanh((pi/2) * sinh(t))

    ``f`` must have signature ``f(x, param)``.
    Returns ``(single, batch)``; the batch routine uses ``prange``.
    """
    f_jit = njit(inline="always", fastmath=fastmath)(f)

    @njit(fastmath=fastmath)
    def single(a, b, param, aw):
        noff = 5
        lenawm = int(aw[0] + 0.5)
        nk = int(aw[1] + 0.5)
        epsh = aw[4]
        ba = b - a

        i = f_jit((a + b) * aw[noff], param)
        ir = i * aw[noff + 1]
        i *= aw[noff + 2]
        err = abs(i)

        k = nk + noff
        j = noff

        while True:
            j += 3
            xa = ba * aw[j]
            fa = f_jit(a + xa, param)
            fb = f_jit(b - xa, param)
            ir += (fa + fb) * aw[j + 1]
            fa *= aw[j + 2]
            fb *= aw[j + 2]
            i += fa + fb
            err += abs(fa) + abs(fb)
            if not (aw[j] > epsh and j < k):
                break

        errt = err * aw[3]
        errh = err * epsh
        errd = 1.0 + 2.0 * errh
        jtmp = j

        while abs(fa) > errt and j < k:
            j += 3
            fa = f_jit(a + ba * aw[j], param)
            ir += fa * aw[j + 1]
            fa *= aw[j + 2]
            i += fa

        jm = j
        j = jtmp

        while abs(fb) > errt and j < k:
            j += 3
            fb = f_jit(b - ba * aw[j], param)
            ir += fb * aw[j + 1]
            fb *= aw[j + 2]
            i += fb

        if j < jm:
            jm = j

        jm -= noff + 3
        h = 1.0
        m = 1
        klim = k + nk

        while errd > errh and klim <= lenawm:
            iback = i
            irback = ir

            while True:
                jtmp = k + jm
                j = k + 3
                while j <= jtmp:
                    xa = ba * aw[j]
                    fa = f_jit(a + xa, param)
                    fb = f_jit(b - xa, param)
                    ir += (fa + fb) * aw[j + 1]
                    i += (fa + fb) * aw[j + 2]
                    j += 3

                k += nk
                j = jtmp

                while True:
                    j += 3
                    fa = f_jit(a + ba * aw[j], param)
                    ir += fa * aw[j + 1]
                    fa *= aw[j + 2]
                    i += fa
                    if not (abs(fa) > errt and j < k):
                        break

                j = jtmp
                while True:
                    j += 3
                    fb = f_jit(b - ba * aw[j], param)
                    ir += fb * aw[j + 1]
                    fb *= aw[j + 2]
                    i += fb
                    if not (abs(fb) > errt and j < k):
                        break

                if not (k < klim):
                    break

            errd = h * (abs(i - 2.0 * iback) + abs(ir - 2.0 * irback))
            h *= 0.5
            m *= 2
            klim = 2 * klim - noff

        i *= h * ba
        if errd > errh:
            err = -errd * (m * abs(ba))
        else:
            err = err * aw[2] * (m * abs(ba))
        return i, err

    @njit(parallel=True, fastmath=fastmath)
    def batch(a, b, params, aw):
        n = a.size
        values = np.empty(n, dtype=np.float64)
        errors = np.empty(n, dtype=np.float64)
        for q in prange(n):
            values[q], errors[q] = single(a[q], b[q], params[q], aw)
        return values, errors

    return single, batch


def make_intdei_cpu(f: Callable, *, fastmath: bool = False):
    """Build compiled CPU integrators for ``(a, +infinity)``.

    DE transform::

        x(t) = a + exp((pi/2) * sinh(t))

    ``f`` must have signature ``f(x, param)``.
    Returns ``(single, batch)``; the batch routine uses ``prange``.
    """
    f_jit = njit(inline="always", fastmath=fastmath)(f)

    @njit(fastmath=fastmath)
    def single(a, param, aw):
        noff = 5
        lenawm = int(aw[0] + 0.5)
        nk = int(aw[1] + 0.5)
        epsh = aw[4]

        i = f_jit(a + aw[noff], param)
        ir = i * aw[noff + 1]
        i *= aw[noff + 2]
        err = abs(i)

        k = nk + noff
        j = noff

        while True:
            j += 6
            fm = f_jit(a + aw[j], param)
            fp = f_jit(a + aw[j + 1], param)
            ir += fm * aw[j + 2] + fp * aw[j + 3]
            fm *= aw[j + 4]
            fp *= aw[j + 5]
            i += fm + fp
            err += abs(fm) + abs(fp)
            if not (aw[j] > epsh and j < k):
                break

        errt = err * aw[3]
        errh = err * epsh
        errd = 1.0 + 2.0 * errh
        jtmp = j

        while abs(fm) > errt and j < k:
            j += 6
            fm = f_jit(a + aw[j], param)
            ir += fm * aw[j + 2]
            fm *= aw[j + 4]
            i += fm

        jm = j
        j = jtmp

        while abs(fp) > errt and j < k:
            j += 6
            fp = f_jit(a + aw[j + 1], param)
            ir += fp * aw[j + 3]
            fp *= aw[j + 5]
            i += fp

        if j < jm:
            jm = j

        jm -= noff + 6
        h = 1.0
        m = 1
        klim = k + nk

        while errd > errh and klim <= lenawm:
            iback = i
            irback = ir

            while True:
                jtmp = k + jm
                j = k + 6
                while j <= jtmp:
                    fm = f_jit(a + aw[j], param)
                    fp = f_jit(a + aw[j + 1], param)
                    ir += fm * aw[j + 2] + fp * aw[j + 3]
                    i += fm * aw[j + 4] + fp * aw[j + 5]
                    j += 6

                k += nk
                j = jtmp

                while True:
                    j += 6
                    fm = f_jit(a + aw[j], param)
                    ir += fm * aw[j + 2]
                    fm *= aw[j + 4]
                    i += fm
                    if not (abs(fm) > errt and j < k):
                        break

                j = jtmp
                while True:
                    j += 6
                    fp = f_jit(a + aw[j + 1], param)
                    ir += fp * aw[j + 3]
                    fp *= aw[j + 5]
                    i += fp
                    if not (abs(fp) > errt and j < k):
                        break

                if not (k < klim):
                    break

            errd = h * (abs(i - 2.0 * iback) + abs(ir - 2.0 * irback))
            h *= 0.5
            m *= 2
            klim = 2 * klim - noff

        i *= h
        if errd > errh:
            err = -errd * m
        else:
            err *= aw[2] * m
        return i, err

    @njit(parallel=True, fastmath=fastmath)
    def batch(a, params, aw):
        n = a.size
        values = np.empty(n, dtype=np.float64)
        errors = np.empty(n, dtype=np.float64)
        for q in prange(n):
            values[q], errors[q] = single(a[q], params[q], aw)
        return values, errors

    return single, batch


def make_intdeid_cpu(f: Callable, *, fastmath: bool = False):
    """Build compiled CPU integrators for the whole real line.

    DE transform::

        x(t) = sinh((pi/2) * sinh(t))

    ``f`` must have signature ``f(x, param)``.
    Returns ``(single, batch)``; the batch routine uses ``prange``.
    """
    f_jit = njit(inline="always", fastmath=fastmath)(f)

    @njit(fastmath=fastmath)
    def single(param, aw):
        noff = 5
        lenawm = int(aw[0] + 0.5)
        nk = int(aw[1] + 0.5)
        epsh = aw[4]

        i = f_jit(aw[noff], param)
        ir = i * aw[noff + 1]
        i *= aw[noff + 2]
        err = abs(i)

        k = nk + noff
        j = noff

        while True:
            j += 4
            x = aw[j + 1]
            fm = f_jit(-x, param)
            fp = f_jit(x, param)
            ir += (fm + fp) * aw[j + 2]
            fm *= aw[j + 3]
            fp *= aw[j + 3]
            i += fm + fp
            err += abs(fm) + abs(fp)
            if not (aw[j] > epsh and j < k):
                break

        errt = err * aw[3]
        errh = err * epsh
        errd = 1.0 + 2.0 * errh
        jtmp = j

        while abs(fm) > errt and j < k:
            j += 4
            fm = f_jit(-aw[j + 1], param)
            ir += fm * aw[j + 2]
            fm *= aw[j + 3]
            i += fm

        jm = j
        j = jtmp

        while abs(fp) > errt and j < k:
            j += 4
            fp = f_jit(aw[j + 1], param)
            ir += fp * aw[j + 2]
            fp *= aw[j + 3]
            i += fp

        if j < jm:
            jm = j

        jm -= noff + 4
        h = 1.0
        m = 1
        klim = k + nk

        while errd > errh and klim <= lenawm:
            iback = i
            irback = ir

            while True:
                jtmp = k + jm
                j = k + 4
                while j <= jtmp:
                    x = aw[j + 1]
                    fm = f_jit(-x, param)
                    fp = f_jit(x, param)
                    ir += (fm + fp) * aw[j + 2]
                    i += (fm + fp) * aw[j + 3]
                    j += 4

                k += nk
                j = jtmp

                while True:
                    j += 4
                    fm = f_jit(-aw[j + 1], param)
                    ir += fm * aw[j + 2]
                    fm *= aw[j + 3]
                    i += fm
                    if not (abs(fm) > errt and j < k):
                        break

                j = jtmp
                while True:
                    j += 4
                    fp = f_jit(aw[j + 1], param)
                    ir += fp * aw[j + 2]
                    fp *= aw[j + 3]
                    i += fp
                    if not (abs(fp) > errt and j < k):
                        break

                if not (k < klim):
                    break

            errd = h * (abs(i - 2.0 * iback) + abs(ir - 2.0 * irback))
            h *= 0.5
            m *= 2
            klim = 2 * klim - noff

        i *= h
        if errd > errh:
            err = -errd * m
        else:
            err *= aw[2] * m
        return i, err

    @njit(parallel=True, fastmath=fastmath)
    def batch(params, aw):
        n = params.size
        values = np.empty(n, dtype=np.float64)
        errors = np.empty(n, dtype=np.float64)
        for q in prange(n):
            values[q], errors[q] = single(params[q], aw)
        return values, errors

    return single, batch


def make_intdeo_cpu(f: Callable, *, fastmath: bool = False):
    """Build compiled CPU integrators for oscillatory semi-infinite tails.

    Uses Ooura's Fourier-type DE transformation with paired nodes
    ``x_minus`` and ``x_plus`` and complementary weights ``beta`` and
    ``pi - beta``.

    ``f`` must have signature ``f(x, omega, param)``.
    Returns ``(single, batch)``; the batch routine uses ``prange``.
    """
    f_jit = njit(inline="always", fastmath=fastmath)(f)

    @njit(fastmath=fastmath)
    def single(a, omega, param, aw):
        if omega == 0.0:
            return math.nan, math.nan

        lenawm = int(aw[0] + 0.5)
        nk0 = int(aw[1] + 0.5)
        noff0 = 6
        nk = int(aw[2] + 0.5)
        noff = 2 * nk0 + noff0
        lmax = int(aw[3] + 0.5)
        eps = aw[4]

        per = 1.0 / abs(omega)
        w02 = 2.0 * aw[noff + 2]
        perw = per * w02

        i = f_jit(a + aw[noff] * per, omega, param)
        ir = i * aw[noff + 1]
        i *= aw[noff + 2]
        err = abs(i)

        h = 2.0
        m = 1
        k = noff
        errh = 0.0

        while True:
            iback = i
            irback = ir
            t = 0.5 * h

            while True:
                if k == noff:
                    tk = 1.0
                    k += nk
                    j = noff

                    while True:
                        j += 3
                        xa = per * aw[j]
                        fm = f_jit(a + xa, omega, param)
                        fp = f_jit(a + xa + perw * tk, omega, param)
                        ir += (fm + fp) * aw[j + 1]
                        fm *= aw[j + 2]
                        fp *= w02 - aw[j + 2]
                        i += fm + fp
                        err += abs(fm) + abs(fp)
                        tk += 1.0
                        if not (aw[j] > eps and j < k):
                            break

                    errh = err * aw[5]
                    err *= eps
                    jm = j - noff
                else:
                    tk = t
                    j = k + 3
                    while j <= k + jm:
                        xa = per * aw[j]
                        fm = f_jit(a + xa, omega, param)
                        fp = f_jit(a + xa + perw * tk, omega, param)
                        ir += (fm + fp) * aw[j + 1]
                        fm *= aw[j + 2]
                        fp *= w02 - aw[j + 2]
                        i += fm + fp
                        tk += 1.0
                        j += 3

                    j = k + jm
                    k += nk

                while abs(fm) > err and j < k:
                    j += 3
                    fm = f_jit(a + per * aw[j], omega, param)
                    ir += fm * aw[j + 1]
                    fm *= aw[j + 2]
                    i += fm

                fm = f_jit(a + perw * tk, omega, param)
                s2 = w02 * fm
                i += s2

                if abs(fp) > err or abs(s2) > err:
                    l = 0
                    while True:
                        l += 1
                        s0 = 0.0
                        s1 = 0.0
                        s2 = fm * aw[noff0 + 1]

                        j = noff0 + 2
                        while j <= noff - 2:
                            tk += 1.0
                            fm = f_jit(a + perw * tk, omega, param)
                            s0 += fm
                            s1 += fm * aw[j]
                            s2 += fm * aw[j + 1]
                            j += 2

                        if s2 <= err or l >= lmax:
                            break
                        i += w02 * s0

                    i += s1
                    if s2 > err:
                        err = s2

                t += h
                if not (t < 1.0):
                    break

            if m == 1:
                errd = 1.0 + 2.0 * errh
            else:
                errd = h * (abs(i - 2.0 * iback) + abs(ir - 2.0 * irback))

            h *= 0.5
            m *= 2
            if not (errd > errh and 2 * k - noff <= lenawm):
                break

        i *= h * per
        if errd > errh:
            err = -errd * per
        else:
            err *= per * m * 0.5
        return i, err

    @njit(parallel=True, fastmath=fastmath)
    def batch(a, omega, params, aw):
        n = a.size
        values = np.empty(n, dtype=np.float64)
        errors = np.empty(n, dtype=np.float64)
        for q in prange(n):
            values[q], errors[q] = single(a[q], omega[q], params[q], aw)
        return values, errors

    return single, batch


# ---------------------------------------------------------------------------
# CUDA factories: one GPU thread evaluates one independent integral.
# ---------------------------------------------------------------------------


def _require_cuda():
    if cuda is None:
        raise RuntimeError(
            "CUDA support is unavailable. Install NVIDIA numba-cuda and a "
            "compatible NVIDIA driver/toolkit."
        )


def _host_cuda_launcher(kernel, arrays, aw, threads_per_block=128, stream=0):
    """Common host launcher used by all CUDA factories."""
    _require_cuda()
    host_arrays = [np.ascontiguousarray(x, dtype=np.float64) for x in arrays]
    n = host_arrays[0].size
    if any(x.size != n for x in host_arrays):
        raise ValueError("all batch input arrays must have the same length")

    d_inputs = [cuda.to_device(x, stream=stream) for x in host_arrays]
    d_aw = cuda.to_device(_as_aw(aw), stream=stream)
    d_values = cuda.device_array(n, dtype=np.float64, stream=stream)
    d_errors = cuda.device_array(n, dtype=np.float64, stream=stream)

    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block, stream](*d_inputs, d_aw, d_values, d_errors)

    values = d_values.copy_to_host(stream=stream)
    errors = d_errors.copy_to_host(stream=stream)
    if stream != 0:
        stream.synchronize()
    return values, errors


def make_intde_cuda(f: Callable, *, fastmath: bool = False):
    """Build an NVIDIA-CUDA batch integrator for ``(a, b)``.

    DE transform::

        x(t) = (a + b)/2
               + (b - a)/2 * tanh((pi/2) * sinh(t))

    One CUDA thread evaluates one complete adaptive integral.
    ``f`` must have signature ``f(x, param)`` and use CUDA-supported Python.
    Returns ``(kernel, batch_host)``.
    """
    _require_cuda()
    f_device = cuda.jit(device=True, inline=True, fastmath=fastmath)(f)

    @cuda.jit(device=True, inline=True)
    def one(a, b, param, aw):
        noff = 5
        lenawm = int(aw[0] + 0.5)
        nk = int(aw[1] + 0.5)
        epsh = aw[4]
        ba = b - a

        i = f_device((a + b) * aw[noff], param)
        ir = i * aw[noff + 1]
        i *= aw[noff + 2]
        err = abs(i)
        k = nk + noff
        j = noff

        while True:
            j += 3
            xa = ba * aw[j]
            fa = f_device(a + xa, param)
            fb = f_device(b - xa, param)
            ir += (fa + fb) * aw[j + 1]
            fa *= aw[j + 2]
            fb *= aw[j + 2]
            i += fa + fb
            err += abs(fa) + abs(fb)
            if not (aw[j] > epsh and j < k):
                break

        errt = err * aw[3]
        errh = err * epsh
        errd = 1.0 + 2.0 * errh
        jtmp = j

        while abs(fa) > errt and j < k:
            j += 3
            fa = f_device(a + ba * aw[j], param)
            ir += fa * aw[j + 1]
            fa *= aw[j + 2]
            i += fa

        jm = j
        j = jtmp
        while abs(fb) > errt and j < k:
            j += 3
            fb = f_device(b - ba * aw[j], param)
            ir += fb * aw[j + 1]
            fb *= aw[j + 2]
            i += fb

        if j < jm:
            jm = j
        jm -= noff + 3
        h = 1.0
        m = 1
        klim = k + nk

        while errd > errh and klim <= lenawm:
            iback = i
            irback = ir
            while True:
                jtmp = k + jm
                j = k + 3
                while j <= jtmp:
                    xa = ba * aw[j]
                    fa = f_device(a + xa, param)
                    fb = f_device(b - xa, param)
                    ir += (fa + fb) * aw[j + 1]
                    i += (fa + fb) * aw[j + 2]
                    j += 3
                k += nk
                j = jtmp
                while True:
                    j += 3
                    fa = f_device(a + ba * aw[j], param)
                    ir += fa * aw[j + 1]
                    fa *= aw[j + 2]
                    i += fa
                    if not (abs(fa) > errt and j < k):
                        break
                j = jtmp
                while True:
                    j += 3
                    fb = f_device(b - ba * aw[j], param)
                    ir += fb * aw[j + 1]
                    fb *= aw[j + 2]
                    i += fb
                    if not (abs(fb) > errt and j < k):
                        break
                if not (k < klim):
                    break
            errd = h * (abs(i - 2.0 * iback) + abs(ir - 2.0 * irback))
            h *= 0.5
            m *= 2
            klim = 2 * klim - noff

        i *= h * ba
        if errd > errh:
            err = -errd * (m * abs(ba))
        else:
            err = err * aw[2] * (m * abs(ba))
        return i, err

    @cuda.jit(fastmath=fastmath)
    def kernel(a, b, params, aw, values, errors):
        q = cuda.grid(1)
        if q < a.size:
            values[q], errors[q] = one(a[q], b[q], params[q], aw)

    def batch_host(a, b, params, aw, threads_per_block=128, stream=0):
        return _host_cuda_launcher(
            kernel, (a, b, params), aw, threads_per_block, stream
        )

    return kernel, batch_host


def make_intdei_cuda(f: Callable, *, fastmath: bool = False):
    """Build an NVIDIA-CUDA batch integrator for ``(a, +infinity)``.

    DE transform::

        x(t) = a + exp((pi/2) * sinh(t))

    One CUDA thread evaluates one complete adaptive integral.
    """
    _require_cuda()
    f_device = cuda.jit(device=True, inline=True, fastmath=fastmath)(f)

    @cuda.jit(device=True, inline=True)
    def one(a, param, aw):
        noff = 5
        lenawm = int(aw[0] + 0.5)
        nk = int(aw[1] + 0.5)
        epsh = aw[4]

        i = f_device(a + aw[noff], param)
        ir = i * aw[noff + 1]
        i *= aw[noff + 2]
        err = abs(i)
        k = nk + noff
        j = noff

        while True:
            j += 6
            fm = f_device(a + aw[j], param)
            fp = f_device(a + aw[j + 1], param)
            ir += fm * aw[j + 2] + fp * aw[j + 3]
            fm *= aw[j + 4]
            fp *= aw[j + 5]
            i += fm + fp
            err += abs(fm) + abs(fp)
            if not (aw[j] > epsh and j < k):
                break

        errt = err * aw[3]
        errh = err * epsh
        errd = 1.0 + 2.0 * errh
        jtmp = j

        while abs(fm) > errt and j < k:
            j += 6
            fm = f_device(a + aw[j], param)
            ir += fm * aw[j + 2]
            fm *= aw[j + 4]
            i += fm

        jm = j
        j = jtmp
        while abs(fp) > errt and j < k:
            j += 6
            fp = f_device(a + aw[j + 1], param)
            ir += fp * aw[j + 3]
            fp *= aw[j + 5]
            i += fp

        if j < jm:
            jm = j
        jm -= noff + 6
        h = 1.0
        m = 1
        klim = k + nk

        while errd > errh and klim <= lenawm:
            iback = i
            irback = ir
            while True:
                jtmp = k + jm
                j = k + 6
                while j <= jtmp:
                    fm = f_device(a + aw[j], param)
                    fp = f_device(a + aw[j + 1], param)
                    ir += fm * aw[j + 2] + fp * aw[j + 3]
                    i += fm * aw[j + 4] + fp * aw[j + 5]
                    j += 6
                k += nk
                j = jtmp
                while True:
                    j += 6
                    fm = f_device(a + aw[j], param)
                    ir += fm * aw[j + 2]
                    fm *= aw[j + 4]
                    i += fm
                    if not (abs(fm) > errt and j < k):
                        break
                j = jtmp
                while True:
                    j += 6
                    fp = f_device(a + aw[j + 1], param)
                    ir += fp * aw[j + 3]
                    fp *= aw[j + 5]
                    i += fp
                    if not (abs(fp) > errt and j < k):
                        break
                if not (k < klim):
                    break
            errd = h * (abs(i - 2.0 * iback) + abs(ir - 2.0 * irback))
            h *= 0.5
            m *= 2
            klim = 2 * klim - noff

        i *= h
        if errd > errh:
            err = -errd * m
        else:
            err *= aw[2] * m
        return i, err

    @cuda.jit(fastmath=fastmath)
    def kernel(a, params, aw, values, errors):
        q = cuda.grid(1)
        if q < a.size:
            values[q], errors[q] = one(a[q], params[q], aw)

    def batch_host(a, params, aw, threads_per_block=128, stream=0):
        return _host_cuda_launcher(
            kernel, (a, params), aw, threads_per_block, stream
        )

    return kernel, batch_host


def make_intdeid_cuda(f: Callable, *, fastmath: bool = False):
    """Build an NVIDIA-CUDA batch integrator for the whole real line.

    DE transform::

        x(t) = sinh((pi/2) * sinh(t))

    One CUDA thread evaluates one complete adaptive integral.
    """
    _require_cuda()
    f_device = cuda.jit(device=True, inline=True, fastmath=fastmath)(f)

    @cuda.jit(device=True, inline=True)
    def one(param, aw):
        noff = 5
        lenawm = int(aw[0] + 0.5)
        nk = int(aw[1] + 0.5)
        epsh = aw[4]

        i = f_device(aw[noff], param)
        ir = i * aw[noff + 1]
        i *= aw[noff + 2]
        err = abs(i)
        k = nk + noff
        j = noff

        while True:
            j += 4
            x = aw[j + 1]
            fm = f_device(-x, param)
            fp = f_device(x, param)
            ir += (fm + fp) * aw[j + 2]
            fm *= aw[j + 3]
            fp *= aw[j + 3]
            i += fm + fp
            err += abs(fm) + abs(fp)
            if not (aw[j] > epsh and j < k):
                break

        errt = err * aw[3]
        errh = err * epsh
        errd = 1.0 + 2.0 * errh
        jtmp = j

        while abs(fm) > errt and j < k:
            j += 4
            fm = f_device(-aw[j + 1], param)
            ir += fm * aw[j + 2]
            fm *= aw[j + 3]
            i += fm

        jm = j
        j = jtmp
        while abs(fp) > errt and j < k:
            j += 4
            fp = f_device(aw[j + 1], param)
            ir += fp * aw[j + 2]
            fp *= aw[j + 3]
            i += fp

        if j < jm:
            jm = j
        jm -= noff + 4
        h = 1.0
        m = 1
        klim = k + nk

        while errd > errh and klim <= lenawm:
            iback = i
            irback = ir
            while True:
                jtmp = k + jm
                j = k + 4
                while j <= jtmp:
                    x = aw[j + 1]
                    fm = f_device(-x, param)
                    fp = f_device(x, param)
                    ir += (fm + fp) * aw[j + 2]
                    i += (fm + fp) * aw[j + 3]
                    j += 4
                k += nk
                j = jtmp
                while True:
                    j += 4
                    fm = f_device(-aw[j + 1], param)
                    ir += fm * aw[j + 2]
                    fm *= aw[j + 3]
                    i += fm
                    if not (abs(fm) > errt and j < k):
                        break
                j = jtmp
                while True:
                    j += 4
                    fp = f_device(aw[j + 1], param)
                    ir += fp * aw[j + 2]
                    fp *= aw[j + 3]
                    i += fp
                    if not (abs(fp) > errt and j < k):
                        break
                if not (k < klim):
                    break
            errd = h * (abs(i - 2.0 * iback) + abs(ir - 2.0 * irback))
            h *= 0.5
            m *= 2
            klim = 2 * klim - noff

        i *= h
        if errd > errh:
            err = -errd * m
        else:
            err *= aw[2] * m
        return i, err

    @cuda.jit(fastmath=fastmath)
    def kernel(params, aw, values, errors):
        q = cuda.grid(1)
        if q < params.size:
            values[q], errors[q] = one(params[q], aw)

    def batch_host(params, aw, threads_per_block=128, stream=0):
        return _host_cuda_launcher(
            kernel, (params,), aw, threads_per_block, stream
        )

    return kernel, batch_host


def make_intdeo_cuda(f: Callable, *, fastmath: bool = False):
    """Build an NVIDIA-CUDA batch integrator for oscillatory tails.

    Uses Ooura's paired Fourier-type DE abscissae and weights.
    One CUDA thread evaluates one complete adaptive integral.
    """
    _require_cuda()
    f_device = cuda.jit(device=True, inline=True, fastmath=fastmath)(f)

    @cuda.jit(device=True, inline=True)
    def one(a, omega, param, aw):
        if omega == 0.0:
            return math.nan, math.nan

        lenawm = int(aw[0] + 0.5)
        nk0 = int(aw[1] + 0.5)
        noff0 = 6
        nk = int(aw[2] + 0.5)
        noff = 2 * nk0 + noff0
        lmax = int(aw[3] + 0.5)
        eps = aw[4]
        per = 1.0 / abs(omega)
        w02 = 2.0 * aw[noff + 2]
        perw = per * w02

        i = f_device(a + aw[noff] * per, omega, param)
        ir = i * aw[noff + 1]
        i *= aw[noff + 2]
        err = abs(i)
        h = 2.0
        m = 1
        k = noff
        errh = 0.0

        while True:
            iback = i
            irback = ir
            t = 0.5 * h
            while True:
                if k == noff:
                    tk = 1.0
                    k += nk
                    j = noff
                    while True:
                        j += 3
                        xa = per * aw[j]
                        fm = f_device(a + xa, omega, param)
                        fp = f_device(a + xa + perw * tk, omega, param)
                        ir += (fm + fp) * aw[j + 1]
                        fm *= aw[j + 2]
                        fp *= w02 - aw[j + 2]
                        i += fm + fp
                        err += abs(fm) + abs(fp)
                        tk += 1.0
                        if not (aw[j] > eps and j < k):
                            break
                    errh = err * aw[5]
                    err *= eps
                    jm = j - noff
                else:
                    tk = t
                    j = k + 3
                    while j <= k + jm:
                        xa = per * aw[j]
                        fm = f_device(a + xa, omega, param)
                        fp = f_device(a + xa + perw * tk, omega, param)
                        ir += (fm + fp) * aw[j + 1]
                        fm *= aw[j + 2]
                        fp *= w02 - aw[j + 2]
                        i += fm + fp
                        tk += 1.0
                        j += 3
                    j = k + jm
                    k += nk

                while abs(fm) > err and j < k:
                    j += 3
                    fm = f_device(a + per * aw[j], omega, param)
                    ir += fm * aw[j + 1]
                    fm *= aw[j + 2]
                    i += fm

                fm = f_device(a + perw * tk, omega, param)
                s2 = w02 * fm
                i += s2

                if abs(fp) > err or abs(s2) > err:
                    l = 0
                    while True:
                        l += 1
                        s0 = 0.0
                        s1 = 0.0
                        s2 = fm * aw[noff0 + 1]
                        j = noff0 + 2
                        while j <= noff - 2:
                            tk += 1.0
                            fm = f_device(a + perw * tk, omega, param)
                            s0 += fm
                            s1 += fm * aw[j]
                            s2 += fm * aw[j + 1]
                            j += 2
                        if s2 <= err or l >= lmax:
                            break
                        i += w02 * s0
                    i += s1
                    if s2 > err:
                        err = s2

                t += h
                if not (t < 1.0):
                    break

            if m == 1:
                errd = 1.0 + 2.0 * errh
            else:
                errd = h * (abs(i - 2.0 * iback) + abs(ir - 2.0 * irback))
            h *= 0.5
            m *= 2
            if not (errd > errh and 2 * k - noff <= lenawm):
                break

        i *= h * per
        if errd > errh:
            err = -errd * per
        else:
            err *= per * m * 0.5
        return i, err

    @cuda.jit(fastmath=fastmath)
    def kernel(a, omega, params, aw, values, errors):
        q = cuda.grid(1)
        if q < a.size:
            values[q], errors[q] = one(a[q], omega[q], params[q], aw)

    def batch_host(a, omega, params, aw, threads_per_block=128, stream=0):
        return _host_cuda_launcher(
            kernel, (a, omega, params), aw, threads_per_block, stream
        )

    return kernel, batch_host
