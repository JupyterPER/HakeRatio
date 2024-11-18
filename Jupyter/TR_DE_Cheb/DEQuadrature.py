"""
Original C file  for DE quadrature: Takuya Ooura, https://www.kurims.kyoto-u.ac.jp/~ooura/index.html
Python conversion and modification: Andrej Gajdos, Jozef Hanc, https://github.com/fdslrm
"""

import math
import numpy as np

def intde(f, a, b, eps_rel = 1e-10):
    '''
    I = integral of f(x) over (a,b)
    '''
    
    mmax = 256
    efs = 0.1 
    hoff = 8.5
    
    pi2 = 2 * math.atan(1)
    epsln = 1 - math.log(efs * eps_rel)
    epsh = math.sqrt(efs * eps_rel)
    h0 = hoff / epsln
    ehp = math.exp(h0)
    ehm = 1 / ehp
    epst = math.exp(-ehm * epsln)
    ba = b - a
    ir = f((a + b) * 0.5) * (ba * 0.25)
    i = ir * (2 * pi2)
    err = abs(i) * epst
    h = 2 * h0
    m = 1
    
    while True:
        
        iback = i
        irback = ir
        t = h * 0.5
        
        while True:
            
            em = math.exp(t)
            ep = pi2 * em
            em = pi2 / em
            
            while True:
                
                xw = 1 / (1 + math.exp(ep - em))
                xa = ba * xw
                wg = xa * (1 - xw)
                fa = f(a + xa) * wg
                fb = f(b - xa) * wg
                ir += fa + fb
                i += (fa + fb) * (ep + em)
                errt = (abs(fa) + abs(fb)) * (ep + em)
                
                if (m == 1): 
                    err += errt * epst
                
                ep *= ehp
                em *= ehm
                
                if not(errt > err or xw > epsh):
                
                    break
                
            t += h
            
            if not(t < h0):
            
                break
        
        if m == 1:
            errh = (err / epst) * epsh * h0
            errd = 1 + 2 * errh
            
        else: 
            errd = h * (abs(i - 2 * iback) + 4 * abs(ir - 2 * irback))
        
        h *= 0.5;
        m *= 2;
        
        if not(errd > errh and m < mmax):
        
            break
    
    i *= h
    
    if errd > errh:
        err = -errd * m
    else: 
        err = errh * epsh * m / (2 * efs)
    
    result = [ i, err, m ]
    
    return result    


def intdei(f, a, eps_rel = 1e-10):
    '''
    I = integral of nonoscillatory f(x) over (a,oo)
    '''
    
    mmax = 256
    efs = 0.1
    hoff = 11
    
    pi4 = math.atan(1)
    epsln = 1 - math.log(efs * eps_rel)
    epsh = math.sqrt(efs * eps_rel)
    h0 = hoff / epsln
    ehp = math.exp(h0)
    ehm = 1 / ehp
    epst = math.exp(-ehm * epsln)
    ir = f(a + 1)
    i = ir * (2 * pi4)
    err = abs(i) * epst
    h = 2 * h0
    m = 1
    
    while True:
        
        iback = i
        irback = ir
        t = h * 0.5
        
        while True:
            
            em = math.exp(t)
            ep = pi4 * em
            em = pi4 / em
            
            while True:
                
                xp = math.exp(ep - em)
                xm = 1 / xp
                fp = f(a + xp) * xp
                fm = f(a + xm) * xm
                ir += fp + fm
                i += (fp + fm) * (ep + em)
                errt = (abs(fp) + abs(fm)) * (ep + em)
                
                if m == 1: 
                    err += errt * epst
                    
                ep *= ehp
                em *= ehm
                
                if not(errt > err or xm > epsh):
                    
                    break
            
            t += h
            
            if not(t < h0):
                
                break
            
        if m == 1:
            errh = (err / epst) * epsh * h0
            errd = 1 + 2 * errh
        else: 
            errd = h * (abs(i - 2 * iback) + 4 * abs(ir - 2 * irback))
        
        h *= 0.5
        m *= 2
        
        if not(errd > errh and m < mmax):
            
            break
    
    i *= h
    
    if errd > errh:
        err = -errd * m
    else:
        err = errh * epsh * m / (2 * efs)
    
    result = [i, err, m]
    
    return result

            
def intdeo(f, a, omega, eps_rel = 1e-10):
    '''
    I = integral of f(x) over (a,oo)
        f(x) has an oscillatory component sin(omega*x+theta)
    '''    
       
    mmax = 256
    lmax = 5
    efs = 0.1 
    enoff = 0.4
    pqoff = 2.9
    ppoff = -0.72
    
    pi4 = math.atan(1)
    epsln = 1 - math.log(efs * eps_rel)
    epsh = math.sqrt(efs * eps_rel)
    n = int(enoff * epsln)
    frq4 = abs(omega) / (2 * pi4)
    per2 = 4 * pi4 / abs(omega)
    pq = pqoff / epsln
    pp = ppoff - math.log(pq * pq * frq4)
    ehp = math.exp(2 * pq)
    ehm = 1 / ehp
    xw = math.exp(pp - 2 * pi4)
    i = f(a + math.sqrt(xw * (per2 * 0.5)))
    ir = i * xw
    i *= per2 * 0.5
    err = abs(i)
    h = 2
    m = 1
    
    while True:
        
        iback = i
        irback = ir
        t = h * 0.5
        
        while True:
            
            em = math.exp(2 * pq * t)
            ep = pi4 * em
            em = pi4 / em
            tk = t
            
            while True:
                
                xw = math.exp(pp - ep - em)
                wg = math.sqrt(frq4 * xw + tk * tk)
                xa = xw / (tk + wg)
                wg = (pq * xw * (ep - em) + xa) / wg
                fm = f(a + xa)
                fp = f(a + xa + per2 * tk)
                ir += (fp + fm) * xw
                fm *= wg
                fp *= per2 - wg
                i += fp + fm
            
                if m == 1: 
                    err += abs(fp) + abs(fm)
                    
                ep *= ehp
                em *= ehm
                tk += 1
                
                if not(ep < epsln):
                    
                    break
            
            if m == 1:
                errh = err * epsh
                err *= eps_rel

            tn = tk
            
            while abs(fm) > err: 
                xw = math.exp(pp - ep - em)
                xa = xw / tk * 0.5
                wg = xa * (1 / tk + 2 * pq * (ep - em))
                fm = f(a + xa)
                ir += fm * xw
                fm *= wg
                i += fm
                ep *= ehp
                em *= ehm
                tk += 1
            
            fm = f(a + per2 * tn)
            em = per2 * fm
            i += em
            
            if abs(fp) > err or abs(em) > err: 
                l = 0
                
                while True:
                    l = l + 1
                    tn += n
                    em = fm
                    fm = f(a + per2 * tn)
                    xa = fm
                    ep = fm
                    em += fm
                    xw = 1
                    wg = 1
                    
                    for k in range(1, n):
                        xw = xw * (n + 1 - k) / k
                        wg += xw
                        fp = f(a + per2 * (tn - k))
                        xa += fp
                        ep += fp * wg
                        em += fp * xw
                    
                    wg = per2 * n / (wg * n + xw)
                    em = wg * abs(em)
                    
                    if em <= err or l >= lmax: 
                        
                        break
                    
                    i += per2 * xa
                        
                i += wg * ep
                
                if em > err:
                    err = em
            
            t += h
            
            if not(t < 1):
                
                break
        
        if m == 1: 
            errd = 1 + 2 * errh
        else:  
            errd = h * (abs(i - 2 * iback) + pq * abs(ir - 2 * irback))
            
        h *= 0.5
        m *= 2
        
        if not(errd > errh and m < mmax):
            
            break
        
    i *= h

    if errd > errh: 
        err = -errd
    else: 
        err *= m * 0.5
    
    result = [i, err, m]
    
    return result