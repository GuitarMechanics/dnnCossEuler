import sympy as sp
import numpy as np

class dualGaraKRNew():
    def __init__(self, r, plen, totlen, layout = 'upright'):
        self.r = r
        self.lp = plen
        self.lt = totlen
        self.alp = 1.82
        thp, thd, tp, td, lp, ld, alp, r, s = sp.symbols(r'\theta_p \theta_d t_p t_d l_p l_d \alpha r s')

        phi_s = sp.symbols('\phi_s')
        kda = td / (ld * r)
        kd1 = alp * kda
        kd = 2 * (kda - kd1) * s / ld + kd1

        # kpa = tp / (lp * r)
        # kp1 = alp * kpa
        # kp = 2 * (kpa - kp1) * s / lp + kp1

        # ksum = kpa + kda
        # pang_int = sp.integrate(ksum, (s, 0, lp))
        # pang_eqn = sp.Eq(pang_int, thp)
        # tot_ang = (td + sp.cos(phi_s) * (tp * r*thp)) / r
        # thd_eqn = sp.Eq(td, r * thd - sp.cos(phi_s) * (tp - 2 * r * thp))
        # sol = sp.solve([pang_eqn, thd_eqn], (tp, td))
        dp = sp.integrate(r * kd, (s, 0, lp)) * sp.Rational(1,2)
        thd_eqn = sp.Eq(thd, (td - dp)/r)
        thp_eqn = sp.Eq(thp, (tp + dp)/r)

        sol = sp.solve([thd_eqn, thp_eqn], (tp, td))
        
        self.tpval = sol[tp].subs({'l_d': self.lt,
                                                   'r'  : self.r,
                                                   'l_p': self.lp,
                                                   r'\alpha':self.alp})
        self.tdval = sol[td].subs({'l_d': self.lt,
                                                   'r'  : self.r,
                                                   'l_p': self.lp,
                                                   r'\alpha':self.alp}) + self.tpval

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
    
    def getPosSympy(self, proxang, distang):
        s = sp.symbols('s')
        kpa = proxang / (self.lp + sp.cos(self.getPhiVal(proxang, distang)))
        kp1 = self.alp * kpa
        kp = 2 * (kpa - kp1) * s / (self.lp + sp.cos(self.getPhiVal(proxang, distang))) + kp1
        kda = distang / (self.lt - self.lp)
        kd1 = self.alp * kda
        kd = 2 * (kda - kd1) / (self.lt - self.lp) + kd1

        pang = sp.integrate(kp, s)
        dang = sp.integrate(kd, s) + proxang

        prox_x = sp.integrate(sp.sin(pang), (s, 0, self.lp + sp.cos(self.getPhiVal(proxang, distang))))
        prox_y = sp.integrate(sp.cos(pang), (s, 0, self.lp + sp.cos(self.getPhiVal(proxang, distang))))

        dist_x = sp.integrate(sp.sin(dang), (s, 0, self.lt - self.lp))
        dist_y = sp.integrate(sp.cos(dang), (s, 0, self.lt - self.lp))
        return prox_x, prox_y, dist_x + prox_x, dist_y + prox_y
