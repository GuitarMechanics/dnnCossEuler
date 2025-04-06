import sympy as sp
import numpy as np

class dualGaraKRNew():
    def __init__(self, r, plen, totlen, layout = 'upright'):
        self.r = r
        self.lp = plen
        self.lt = totlen
        self.alp = 1.82
        thp, thd, tp, td, lp, ld, alp, r, s = sp.symbols(r'\theta_p \theta_d t_p t_d l_p l_d \alpha r s')
        kda = td / (ld * r)
        kd1 = alp * kda
        kd = 2 * (kda - kd1) * s / ld + kd1

        kpa = tp / (lp * r)
        kp1 = alp * kpa
        kp = 2 * (kpa - kp1) * s / lp + kp1
        thp_d = sp.integrate(kd, (s, 0, lp))
        # dp = thp_d * r
        # dd = sp.integrate(kd, (s, lp, ld)) * r
        dp = sp.integrate(r * kd, (s, 0, lp))
        dd = sp.integrate(r * kd, (s, lp, ld))
        tp_add = sp.symbols('t_a')
        phi_s = sp.symbols('\phi_s')
        tp_eff = sp.Rational(1,2) * tp_add + sp.cos(phi_s) * dp
        td_eff = dd - sp.Rational(1,2) * tp_add - sp.cos(phi_s) * tp_add

        pang_eqn = sp.Eq(thp, tp_eff / r)
        dang_eqn = sp.Eq(thd, td_eff / r)
        sol = sp.solve([pang_eqn, dang_eqn], (tp_add, td))
        self.tpadd = sol[tp_add]
        self.tdsol = sol[td]
        self.tpsol = self.tpadd + dp.subs({td: self.tdsol})
        
        self.tpval = self.tpsol.subs({'l_d': self.lt,
                                                   'r'  : self.r,
                                                   'l_p': self.lp,
                                                   r'\alpha':self.alp})
        self.tdval = self.tdsol.subs({'l_d': self.lt,
                                                   'r'  : self.r,
                                                   'l_p': self.lp,
                                                   r'\alpha':self.alp})

    def getDistReqTDL(self, distang):
        retval = self.tdsol.subs({r'\theta_d':distang})
        return retval
    
    def getProxTDL(self, proxang, distang):
        dreq = self.getDistReqTDL(distang)
        ptdl = self.tpsol.subs({r'\theta_p':proxang,
                                't_d': dreq})
        return ptdl
    
    def getTDLs_past(self, proxang, distang):
        ptdl = self.getProxTDL(proxang, distang)
        dtdl = self.getDistReqTDL(distang) + ptdl
        return ptdl, dtdl
    
    def getPhiVal(self, proxang, distang):
        if proxang * distang < 0:
            phi_val = sp.pi
        else:
            phi_val = 0
        return phi_val

    def getTDLs(self, proxang, distang):
        tpval = self.tpval.subs({r'\theta_p':proxang,
                                 r'\theta_d':distang,
                                 '\phi_s':self.getPhiVal(proxang, distang)})
        tdval = self.tdval.subs({r'\theta_p':proxang,
                                 r'\theta_d':distang,
                                 '\phi_s':self.getPhiVal(proxang, distang)})
        
        return tpval, tdval