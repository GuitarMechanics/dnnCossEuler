import sympy as sp
import numpy as np

class dualGaraKR():
    def __init__(self, r, plen, totlen, layout = 'upright'):
        self.r = r
        self.lp = plen
        self.lt = totlen
        self.alp = 1.82
        thp, thd, tp, td, lp, ld, alp, r, s = sp.symbols(r'\theta_p \theta_d t_p t_d l_p l_d \alpha r s')
        kda = td / (ld * r)
        kd1 = alp * kda
        kd = 2 * (kda - kd1) * s / ld + kd1
        thdint = sp.integrate(kd, (s, lp, ld))
        # self.thdeqn = thdint - thd
        self.thdeqn = r * thd - td + tp
        self.tdsol = sp.solve(self.thdeqn, td)[0].subs({'l_d': self.lt,
                                                   'r'  : self.r,
                                                   'l_p': self.lp,
                                                   r'\alpha':self.alp})

        kpa = tp / (lp * r)
        kp1 = alp * kpa
        kp = 2 * (kpa - kp1) * s / lp + kp1
        kptot = kd + kp
        kpint = sp.integrate(kptot, (s, 0, lp))
        self.thpeqn = kpint - thp
        self.tpsol = sp.solve(self.thpeqn, tp)[0].subs({'l_d': self.lt,
                                                   'r'  : self.r,
                                                   'l_p': self.lp,
                                                   r'\alpha':self.alp})
        
        self.tdsolsnew = sp.solve((self.thdeqn, self.thpeqn), ('t_p','t_d'))
        self.tpval = self.tdsolsnew[tp].subs({'l_d': self.lt,
                                                   'r'  : self.r,
                                                   'l_p': self.lp,
                                                   r'\alpha':self.alp})
        self.tdval = self.tdsolsnew[td].subs({'l_d': self.lt,
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
    
    def getTDLs(self, proxang, distang):
        tpval = self.tpval.subs({r'\theta_p':proxang,
                                 r'\theta_d':distang})
        tdval = self.tdval.subs({r'\theta_p':proxang,
                                 r'\theta_d':distang})
        
        return tpval, tdval