import numpy as np
import pandas as pd
import scipy.integrate as itg

class NewGaraEuler():
    def __init__(self, len, rad, res = 1000):
        self.len = len
        self.rad = rad
        self.seg = np.linspace(0, self.len, res)
    
    def getKR(self, g_cond):
        if g_cond == -1:
            return 1.18
        elif g_cond == 0:
            return 1.5
        elif g_cond == 1:
            return 1.82
        else:
            raise Exception('Invalid ground angle condition')
        
    def getTipPos(self, g_cond, tdl):
        ka = tdl / (self.len * self.rad)
        ki = ka * self.getKR(g_cond)
        theta = (ka - ki) / self.len * self.seg ** 2 + ki * self.seg
        x = np.trapz(np.sin(theta), self.seg)
        y = np.trapz(np.cos(theta), self.seg)
        return x, y