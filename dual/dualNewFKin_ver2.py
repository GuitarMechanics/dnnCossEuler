import numpy as np
import sympy as sp

class dualNewFKin():
    def __init__(self, rad, plen, totlen, alpha = 1.8):
        self.r = rad
        self.plen = plen
        self.tlen = totlen
        self.alp = alpha
        thp, thd, tp, td, lp, ld, alp, r, s = sp.symbols(r'\theta_p \theta_d t_p t_d l_p l_d \alpha r s', real = True)
        kda = td / (ld * r)
        kd1 = alp * kda
        kd = 2 * (kda - kd1) * s / ld + kd1
        self.tpeff = sp.Symbol(r'\Delta t_{p,eff}', real=True)
        self.tdeff = sp.Symbol(r'\Delta t_{d,eff}', real=True)
        phip = sp.Symbol(r'\phi_{p,eff}', real=True)
        phid = sp.Symbol(r'\phi_{d,eff}', real=True)
        self.cossym = sp.Symbol(r'c_{\phi}')
        self.lcst = lp + self.tpeff * self.cossym
        self.lcstsym = sp.Symbol(r'l_{cst}')
        
        kda2 = self.tdeff / (ld * r)
        kd12 = kda2 * alp
        kd2 = 2 * (kda2 - kd12) * s / ld + kd12
        with sp.assuming(sp.Q.nonzero(ld)):
            self.dpnew = sp.integrate(r * kd2 / 2, (s, 0, self.lcst))
            dpnew_lcstsym = sp.integrate(r * kd2 / 2, (s, 0, self.lcstsym))
            self.proxcompress = self.dpnew.subs([(lp,self.plen),(ld,self.tlen),(r, self.r),(alp, self.alp)]) / 2
        tpeqn2 = sp.Eq(tp, self.tpeff - self.dpnew)
        tdeqn2 = sp.Eq(td, self.tpeff + self.tdeff + self.dpnew)
        sol2 = sp.solve([tpeqn2, tdeqn2],(tp,td))        
        self.tpsol = sol2[tp].subs([(lp,self.plen),(ld,self.tlen),(r, self.r),(alp, self.alp)])
        self.tdsol = sol2[td].subs([(lp,self.plen),(ld,self.tlen),(r, self.r),(alp, self.alp)])

    def getDP(self, td):
        return (self.plen * td) * (self.alp*(self.tlen - self.plen)+self.plen) / (self.tlen**2) / 2
    
    # def getEffTDL(self, tp, td):
    #     dp = self.getDP(td)
    #     tpeff = tp
    #     tdeff = td - tpeff - np.abs(dp / 2)*np.sign(td - tpeff)
    #     return tpeff, tdeff
    def getEffTDL(self, pa, da, degrees=False):
        if degrees:
            pang = np.deg2rad(pa)
            dang = np.deg2rad(da)
        else:
            pang = pa
            dang = da
        peff = pang * self.r
        deff = dang * self.r
        return peff, deff
    
    def getTrueTDL(self, pa, da, degrees = False):
        peff, deff = self.getEffTDL(pa, da, degrees)
        cosval = np.sign(peff) * np.sign(deff)
        lcst = self.plen + peff * cosval
        tpval = self.tpsol.subs([(self.tpeff, peff),(self.tdeff,deff),(self.lcstsym,lcst),(self.cossym,cosval)])
        tdval = self.tdsol.subs([(self.tpeff, peff),(self.tdeff,deff),(self.lcstsym,lcst),(self.cossym,cosval)])

        return tpval, tdval
    
    def retrieveEffTDL(self, tp, td):
        peff, deff = sp.symbols(r't_peff t_deff')
        cosval = sp.sign(peff) * sp.sign(deff)
        lcst = self.plen + peff * cosval
        tpval = self.tpsol.subs([(self.tpeff, peff),(self.tdeff,deff),(self.lcstsym,lcst),(self.cossym,cosval)])
        tdval = self.tdsol.subs([(self.tpeff, peff),(self.tdeff,deff),(self.lcstsym,lcst),(self.cossym,cosval)])
        tpeqn = sp.Eq(tp, tpval)
        tdeqn = sp.Eq(td, tdval)
        sol = sp.nsolve((tpeqn,tdeqn),(peff,deff),(tp, td-tp))
        return sol[0], sol[1]

    def getPos(self, pa, da, degrees = False, resolution = 200):
        peff, deff = self.getEffTDL(pa, da, degrees)
        cosval = np.sign(peff) * np.sign(deff)
        lcst = self.plen + peff * cosval
        proxcompress = float(self.proxcompress.subs([(self.tpeff, peff),
                                               (self.tdeff, deff),
                                               (self.cossym,cosval),
                                               (self.lcstsym, lcst)]))
        newplen = self.plen - np.abs(proxcompress)
        dlen = self.tlen - self.plen
        if degrees:
            pang = np.deg2rad(pa)
            dang = np.deg2rad(da)
        else:
            pang = pa
            dang = da

        psegs = np.linspace(0, newplen, resolution)
        dsegs = np.linspace(0, dlen, resolution)

        kpavg = pang / newplen
        kp1 = self.alp * kpavg
        kdavg = dang / dlen
        kd1 = self.alp * kdavg

        thp = (kpavg - kp1) * psegs **2 / newplen + kp1 * psegs
        thd = (kdavg - kd1) * dsegs **2 / dlen + kd1 * dsegs
        px = np.trapezoid(np.sin(thp), psegs)
        py = np.trapezoid(np.cos(thp), psegs)
        dxraw = np.trapezoid(np.sin(thd), dsegs)
        dyraw = np.trapezoid(np.cos(thd), dsegs)
        prox_rotmat = np.array([[np.cos(-pang), -np.sin(-pang)],
                                [np.sin(-pang),np.cos(-pang)]])
        dpos_raw = np.array([dxraw, dyraw])
        dpos_rot = prox_rotmat @ dpos_raw
        dx_rot = dpos_rot[0]
        dy_rot = dpos_rot[1]
        dx = dx_rot + px
        dy = dy_rot + py
        return px, py, dx, dy