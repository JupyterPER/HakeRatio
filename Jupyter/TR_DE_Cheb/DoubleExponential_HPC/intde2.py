# Original C files for DE quadrature:
# Takuya Ooura, https://www.kurims.kyoto-u.ac.jp/~ooura/intde.html
#
# 2025 (c) Hanc, Hancova (jozef.hanc@upjs.sk)
# Python conversion, whole-line extension, and HPC implementation
# Ver: Jan-2025
"""
Reference Python translation of Takuya Ooura's INTDE2 fast
double-exponential (DE) quadrature package (table-based), extended with a whole-line
nonoscillatory DE rule.

Functions
---------
intdeini  : initialize nodes/weights for finite-interval integration
intde     : integrate f over (a, b)

intdeiini : initialize nodes/weights for nonoscillatory semi-infinite integration
intdei    : integrate nonoscillatory f over (a, +infinity)

intdeidini: initialize nodes/weights for nonoscillatory whole-line integration
intdeid   : integrate nonoscillatory f over (-infinity, +infinity)

intdeoini : initialize nodes/weights for oscillatory semi-infinite integration
intdeo    : integrate oscillatory f over (a, +infinity)

DE transformations
------------------
1. Finite interval ``(a, b)`` — tanh-sinh transformation::

       x(t) = (a + b)/2
              + (b - a)/2 * tanh((pi/2) * sinh(t))

       dx/dt = (b - a) * (pi/4) * cosh(t)
               / cosh((pi/2) * sinh(t))**2

2. Nonoscillatory semi-infinite interval ``(a, +infinity)``::

       x(t) = a + exp((pi/2) * sinh(t))

       dx/dt = (pi/2) * cosh(t) * exp((pi/2) * sinh(t))

3. Nonoscillatory whole line ``(-infinity, +infinity)``::

       x(t) = sinh((pi/2) * sinh(t))

       dx/dt = (pi/2) * cosh(t) * cosh((pi/2) * sinh(t))

   This whole-line rule is an extension of Ooura's INTDE2 package.

4. Oscillatory semi-infinite interval ``(a, +infinity)`` — Ooura's
   Fourier-type DE transformation.

   For an internal table variable ``tau``, define::

       q = pqoff / epsln
       p = ppoff - log(q**2 * 2/pi)

       chi(tau) = exp(p - (pi/2) * cosh(2*q*tau))
       r(tau)   = sqrt((2/pi) * chi(tau) + tau**2)
       alpha(tau) = chi(tau) / (tau + r(tau))

       beta(tau) = (
           q * chi(tau) * (pi/2) * sinh(2*q*tau)
           + alpha(tau)
       ) / r(tau)

   The paired physical abscissae are::

       x_minus = a + alpha(tau) / abs(omega)
       x_plus  = a + (alpha(tau) + pi*tau) / abs(omega)

   and the corresponding internal weights are ``beta(tau)`` and
   ``pi - beta(tau)``.  The final trapezoidal scaling is ``h/abs(omega)``.

Implementation notes
--------------------
The implementation deliberately preserves the original scalar control flow,
adaptive error logic, and layout of the work array ``aw``.  It is therefore a
reference translation, not a vectorized rewrite.

For all integration routines, a negative returned error indicates that the
requested accuracy was not reached before the precomputed table was exhausted.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence

__all__ = [
    "intdeini",
    "intde",
    "intdeiini",
    "intdei",
    "intdeidini",
    "intdeid",
    "intdeoini",
    "intdeo",
]


ScalarFunction = Callable[[float], float]


def intdeini(
    lenaw: int = 8000,
    tiny: float = 1.0e-307,
    eps: float = 1.0e-15,
) -> list[float]:
    """Initialize ``aw`` for :func:`intde` on a finite interval.

    DE transform (tanh-sinh)::

        x(t) = (a + b)/2
               + (b - a)/2 * tanh((pi/2) * sinh(t))

    The stored table is independent of the particular interval ``(a, b)``;
    affine scaling is applied later by :func:`intde`.
    """

    if lenaw <= 1000:
        raise ValueError("lenaw must be greater than 1000")

    # ---- adjustable parameter ----
    efs = 0.1
    hoff = 8.5
    # ------------------------------

    aw = [0.0] * lenaw

    tinyln = -math.log(tiny)
    epsln = 1.0 - math.log(efs * eps)
    h0 = hoff / epsln
    ehp = math.exp(h0)
    ehm = 1.0 / ehp

    aw[2] = eps
    aw[3] = math.exp(-ehm * epsln)
    aw[4] = math.sqrt(efs * eps)

    noff = 5

    aw[noff] = 0.5
    aw[noff + 1] = h0
    aw[noff + 2] = (math.pi / 4.0) * h0

    h = 2.0
    nk = 0
    k = noff + 3

    while True:
        t = h * 0.5

        while True:
            em = math.exp(h0 * t)
            ep = (math.pi / 2.0) * em
            em = (math.pi / 2.0) / em
            j = k

            while True:
                xw = 1.0 / (1.0 + math.exp(ep - em))
                wg = xw * (1.0 - xw) * h0

                aw[j] = xw
                aw[j + 1] = wg * 4.0
                aw[j + 2] = wg * (ep + em)

                ep *= ehp
                em *= ehm
                j += 3

                if not (ep < tinyln and j <= lenaw - 3):
                    break

            t += h
            k += nk

            if not (t < 1.0):
                break

        h *= 0.5

        if nk == 0:
            if j > lenaw - 6:
                j -= 3

            nk = j - noff
            k += nk
            aw[1] = float(nk)

        if not (2 * k - noff - 3 <= lenaw):
            break

    aw[0] = float(k - 3)
    return aw


def intde(
    f: ScalarFunction,
    a: float,
    b: float,
    aw: Sequence[float],
) -> tuple[float, float]:
    """Integrate ``f`` over ``(a, b)`` using the tanh-sinh DE transform.

    Transformation::

        x(t) = (a + b)/2
               + (b - a)/2 * tanh((pi/2) * sinh(t))

    ``aw`` must be produced by :func:`intdeini`.
    """

    noff = 5

    lenawm = int(aw[0] + 0.5)
    nk = int(aw[1] + 0.5)
    epsh = aw[4]
    ba = b - a

    i = f((a + b) * aw[noff])
    ir = i * aw[noff + 1]
    i *= aw[noff + 2]
    err = abs(i)

    k = nk + noff
    j = noff

    while True:
        j += 3
        xa = ba * aw[j]
        fa = f(a + xa)
        fb = f(b - xa)

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
        fa = f(a + ba * aw[j])
        ir += fa * aw[j + 1]
        fa *= aw[j + 2]
        i += fa

    jm = j
    j = jtmp

    while abs(fb) > errt and j < k:
        j += 3
        fb = f(b - ba * aw[j])
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
                fa = f(a + xa)
                fb = f(b - xa)

                ir += (fa + fb) * aw[j + 1]
                i += (fa + fb) * aw[j + 2]
                j += 3

            k += nk
            j = jtmp

            while True:
                j += 3
                fa = f(a + ba * aw[j])
                ir += fa * aw[j + 1]
                fa *= aw[j + 2]
                i += fa

                if not (abs(fa) > errt and j < k):
                    break

            j = jtmp

            while True:
                j += 3
                fb = f(b - ba * aw[j])
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


def intdeiini(
    lenaw: int = 8000,
    tiny: float = 1.0e-307,
    eps: float = 1.0e-15,
) -> list[float]:
    """Initialize ``aw`` for :func:`intdei` on ``(a, +infinity)``.

    DE transform::

        x(t) = a + exp((pi/2) * sinh(t))

        dx/dt = (pi/2) * cosh(t) * exp((pi/2) * sinh(t))

    The table stores the transformed positive and reciprocal node pairs and
    their auxiliary and quadrature weights.
    """

    if lenaw <= 1000:
        raise ValueError("lenaw must be greater than 1000")

    # ---- adjustable parameter ----
    efs = 0.1
    hoff = 11.0
    # ------------------------------

    aw = [0.0] * lenaw

    tinyln = -math.log(tiny)
    epsln = 1.0 - math.log(efs * eps)
    h0 = hoff / epsln
    ehp = math.exp(h0)
    ehm = 1.0 / ehp

    aw[2] = eps
    aw[3] = math.exp(-ehm * epsln)
    aw[4] = math.sqrt(efs * eps)

    noff = 5

    aw[noff] = 1.0
    aw[noff + 1] = 4.0 * h0
    aw[noff + 2] = (math.pi / 2.0) * h0

    h = 2.0
    nk = 0
    k = noff + 6

    while True:
        t = h * 0.5

        while True:
            em = math.exp(h0 * t)
            ep = (math.pi / 4.0) * em
            em = (math.pi / 4.0) / em
            j = k

            while True:
                xp = math.exp(ep - em)
                xm = 1.0 / xp
                wp = xp * ((ep + em) * h0)
                wm = xm * ((ep + em) * h0)

                aw[j] = xm
                aw[j + 1] = xp
                aw[j + 2] = xm * (4.0 * h0)
                aw[j + 3] = xp * (4.0 * h0)
                aw[j + 4] = wm
                aw[j + 5] = wp

                ep *= ehp
                em *= ehm
                j += 6

                if not (ep < tinyln and j <= lenaw - 6):
                    break

            t += h
            k += nk

            if not (t < 1.0):
                break

        h *= 0.5

        if nk == 0:
            if j > lenaw - 12:
                j -= 6

            nk = j - noff
            k += nk
            aw[1] = float(nk)

        if not (2 * k - noff - 6 <= lenaw):
            break

    aw[0] = float(k - 6)
    return aw


def intdei(
    f: ScalarFunction,
    a: float,
    aw: Sequence[float],
) -> tuple[float, float]:
    """Integrate nonoscillatory ``f`` over ``(a, +infinity)``.

    DE transform::

        x(t) = a + exp((pi/2) * sinh(t))

    ``aw`` must be produced by :func:`intdeiini`.
    """

    noff = 5

    lenawm = int(aw[0] + 0.5)
    nk = int(aw[1] + 0.5)
    epsh = aw[4]

    i = f(a + aw[noff])
    ir = i * aw[noff + 1]
    i *= aw[noff + 2]
    err = abs(i)

    k = nk + noff
    j = noff

    while True:
        j += 6
        fm = f(a + aw[j])
        fp = f(a + aw[j + 1])

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
        fm = f(a + aw[j])
        ir += fm * aw[j + 2]
        fm *= aw[j + 4]
        i += fm

    jm = j
    j = jtmp

    while abs(fp) > errt and j < k:
        j += 6
        fp = f(a + aw[j + 1])
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
                fm = f(a + aw[j])
                fp = f(a + aw[j + 1])

                ir += fm * aw[j + 2] + fp * aw[j + 3]
                i += fm * aw[j + 4] + fp * aw[j + 5]
                j += 6

            k += nk
            j = jtmp

            while True:
                j += 6
                fm = f(a + aw[j])
                ir += fm * aw[j + 2]
                fm *= aw[j + 4]
                i += fm

                if not (abs(fm) > errt and j < k):
                    break

            j = jtmp

            while True:
                j += 6
                fp = f(a + aw[j + 1])
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



def intdeidini(
    lenaw: int = 8000,
    tiny: float = 1.0e-307,
    eps: float = 1.0e-15,
) -> list[float]:
    """Initialize ``aw`` for :func:`intdeid` on the whole real line.

    DE transform::

        x(t) = sinh((pi/2) * sinh(t))

        dx/dt = (pi/2) * cosh(t) * cosh((pi/2) * sinh(t))

    This whole-line rule is an extension of Ooura's original INTDE2 package.
    The table stores positive nodes; :func:`intdeid` evaluates the symmetric
    pair ``f(-x)`` and ``f(x)``.
    """
    if lenaw <= 1000:
        raise ValueError("lenaw must be greater than 1000")

    efs = 0.1
    hoff = 11.0
    aw = [0.0] * lenaw

    tinyln = -math.log(tiny)
    epsln = 1.0 - math.log(efs * eps)
    h0 = hoff / epsln
    ehp = math.exp(h0)
    ehm = 1.0 / ehp
    logmax = math.log(sys.float_info.max)
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
                ss = ep - em
                log_chs = ss + math.log1p(math.exp(-2.0 * ss)) - log2
                if not (
                    ep < tinyln
                    and log_chs + math.log(4.0 * h0) < logmax
                    and log_chs + math.log((ep + em) * h0) < logmax
                    and j <= lenaw - 4
                ):
                    break
                u = math.exp(ss)
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


def intdeid(
    f: ScalarFunction,
    aw: Sequence[float],
) -> tuple[float, float]:
    """Integrate nonoscillatory ``f`` over ``(-infinity, +infinity)``.

    DE transform::

        x(t) = sinh((pi/2) * sinh(t))

    The derivative is even, so the algorithm evaluates paired contributions
    from ``f(-x)`` and ``f(x)``.  ``aw`` must be produced by
    :func:`intdeidini`.
    """
    noff = 5
    lenawm = int(aw[0] + 0.5)
    nk = int(aw[1] + 0.5)
    epsh = aw[4]

    i = f(aw[noff])
    ir = i * aw[noff + 1]
    i *= aw[noff + 2]
    err = abs(i)
    k = nk + noff
    j = noff

    while True:
        j += 4
        x = aw[j + 1]
        fm = f(-x)
        fp = f(x)
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
        fm = f(-aw[j + 1])
        ir += fm * aw[j + 2]
        fm *= aw[j + 3]
        i += fm
    jm = j
    j = jtmp
    while abs(fp) > errt and j < k:
        j += 4
        fp = f(aw[j + 1])
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
                fm = f(-x)
                fp = f(x)
                ir += (fm + fp) * aw[j + 2]
                i += (fm + fp) * aw[j + 3]
                j += 4
            k += nk
            j = jtmp
            while True:
                j += 4
                fm = f(-aw[j + 1])
                ir += fm * aw[j + 2]
                fm *= aw[j + 3]
                i += fm
                if not (abs(fm) > errt and j < k):
                    break
            j = jtmp
            while True:
                j += 4
                fp = f(aw[j + 1])
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

def intdeoini(
    lenaw: int = 8000,
    tiny: float = 1.0e-307,
    eps: float = 1.0e-15,
) -> list[float]:
    """Initialize ``aw`` for oscillatory :func:`intdeo`.

    This is Ooura's Fourier-type DE transformation.  With internal variable
    ``tau``::

        chi = exp(p - (pi/2) * cosh(2*q*tau))
        r = sqrt((2/pi) * chi + tau**2)
        alpha = chi / (tau + r)
        beta = (q*chi*(pi/2)*sinh(2*q*tau) + alpha) / r

    The physical paired nodes used by :func:`intdeo` are::

        x_minus = a + alpha / abs(omega)
        x_plus  = a + (alpha + pi*tau) / abs(omega)

    with internal weights ``beta`` and ``pi - beta``.
    """

    if lenaw <= 1000:
        raise ValueError("lenaw must be greater than 1000")

    # ---- adjustable parameter ----
    lmax = 5
    efs = 0.1
    enoff = 0.40
    pqoff = 2.9
    ppoff = -0.72
    # ------------------------------

    aw = [0.0] * lenaw

    tinyln = -math.log(tiny)
    epsln = 1.0 - math.log(efs * eps)
    frq4 = 2.0 / math.pi
    per2 = math.pi
    pq = pqoff / epsln
    pp = ppoff - math.log(pq * pq * frq4)
    ehp = math.exp(2.0 * pq)
    ehm = 1.0 / ehp

    aw[3] = float(lmax)
    aw[4] = eps
    aw[5] = math.sqrt(efs * eps)

    noff0 = 6
    nk0 = 1 + int(enoff * epsln)
    aw[1] = float(nk0)
    noff = 2 * nk0 + noff0

    wg = 0.0
    xw = 1.0

    for k in range(1, nk0 + 1):
        wg += xw
        aw[noff - 2 * k] = wg
        aw[noff - 2 * k + 1] = xw
        xw = xw * (nk0 - k) / k

    wg = per2 / wg

    for k in range(noff0, noff - 1, 2):
        aw[k] *= wg
        aw[k + 1] *= wg

    xw = math.exp(pp - math.pi / 2.0)
    aw[noff] = math.sqrt(xw * (per2 * 0.5))
    aw[noff + 1] = xw * pq
    aw[noff + 2] = per2 * 0.5

    h = 2.0
    nk = 0
    k = noff + 3

    while True:
        t = h * 0.5

        while True:
            em = math.exp(2.0 * pq * t)
            ep = (math.pi / 4.0) * em
            em = (math.pi / 4.0) / em
            tk = t
            j = k

            while True:
                xw = math.exp(pp - ep - em)
                wg = math.sqrt(frq4 * xw + tk * tk)
                xa = xw / (tk + wg)
                wg = (pq * xw * (ep - em) + xa) / wg

                aw[j] = xa
                aw[j + 1] = xw * pq
                aw[j + 2] = wg

                ep *= ehp
                em *= ehm
                tk += 1.0
                j += 3

                if not (ep < tinyln and j <= lenaw - 3):
                    break

            t += h
            k += nk

            if not (t < 1.0):
                break

        h *= 0.5

        if nk == 0:
            if j > lenaw - 6:
                j -= 3

            nk = j - noff
            k += nk
            aw[2] = float(nk)

        if not (2 * k - noff - 3 <= lenaw):
            break

    aw[0] = float(k - 3)
    return aw


def intdeo(
    f: ScalarFunction,
    a: float,
    omega: float,
    aw: Sequence[float],
) -> tuple[float, float]:
    """Integrate oscillatory ``f`` over ``(a, +infinity)``.

    The function is sampled at Ooura's paired Fourier-DE abscissae::

        x_minus = a + alpha(tau) / abs(omega)
        x_plus  = a + (alpha(tau) + pi*tau) / abs(omega)

    with complementary internal weights ``beta(tau)`` and
    ``pi - beta(tau)``.  ``aw`` must be produced by :func:`intdeoini`.
    """

    if omega == 0.0:
        raise ValueError("omega must be nonzero")

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

    i = f(a + aw[noff] * per)
    ir = i * aw[noff + 1]
    i *= aw[noff + 2]
    err = abs(i)

    h = 2.0
    m = 1
    k = noff

    while True:
        iback = i
        irback = ir
        t = h * 0.5

        while True:
            if k == noff:
                tk = 1.0
                k += nk
                j = noff

                while True:
                    j += 3
                    xa = per * aw[j]
                    fm = f(a + xa)
                    fp = f(a + xa + perw * tk)

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
                    fm = f(a + xa)
                    fp = f(a + xa + perw * tk)

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
                fm = f(a + per * aw[j])
                ir += fm * aw[j + 1]
                fm *= aw[j + 2]
                i += fm

            fm = f(a + perw * tk)
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
                        fm = f(a + perw * tk)

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
