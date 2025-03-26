import numpy as np
from scipy.integrate import quad

class GaraCossEuler():
    def __init__(self, len, rad):
        self.len = len
        self.rad = rad

    def getTipAngle(self, tdl):
        return tdl / self.rad
    
    def getAvgCurv(self, tdl):
        return tdl / (self.len * self.rad)
    
    def getInitCurv(self, tdl):
        lr = self.len / self.rad
        th = self.getTipAngle(tdl)
        ka = self.getAvgCurv(tdl)
        kr = (-0.04 * lr ** 2 + 3.6 * lr + 4 * lr * th + 8 * th - 2.5 * th**2) * 0.001 + 1

        return kr * ka
    
    def getTipPos(self, tdl, res = 200):
        seg = np.linspace(0, self.len, res)
        ki = self.getInitCurv(tdl)
        ka = self.getAvgCurv(tdl)
        th = (ka - ki) / self.len * seg ** 2 + ki * seg
        ux = np.sin(th)
        uy = np.cos(th)

        x = np.trapz(ux, seg)
        y = np.trapz(uy, seg)

        return x, y
        
