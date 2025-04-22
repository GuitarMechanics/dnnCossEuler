import numpy as np

class dualNewFKin():
    def __init__(self, r, plen, totlen, alp = 1.8):
        self.r = r
        self.plen = plen
        self.tlen = totlen
        self.alp = alp

    def getDP(self, td):
        return (self.plen * td) * (self.alp*(self.tlen - self.plen)+self.plen) / (self.tlen**2) / 2
    
    def getEffTDL(self, tp, td):
        dp = self.getDP(td)
        tpeff = tp + dp / 2
        tdeff = td - tpeff - np.abs(dp / 2)*np.sign(td - tpeff)
        return tpeff, tdeff
    
    def getAngs(self, tp, td, degrees = False):
        tpeff, tdeff = self.getEffTDL(tp, td)
        pang = tpeff / self.r
        tang = tdeff / self.r

        if degrees:
            return np.rad2deg(pang), np.rad2deg(tang)
        else:
            return pang, tang

    def getPos(self, tp, td, resolution = 200):
        pang, tang = self.getAngs(tp, td)
        dp = self.getDP(td)

        pseglen = self.plen - np.abs(dp / 2)
        pseg = np.linspace(0,pseglen, resolution)
        pka = pang / pseglen
        pk1 = self.alp * pka
        ptheta = (pka - pk1) / pseglen * pseg ** 2 + pk1 * pseg
        px = np.trapezoid(np.sin(ptheta), pseg)
        py = np.trapezoid(np.cos(ptheta), pseg)

        dseglen = self.tlen - self.plen
        dseg = np.linspace(0,dseglen, resolution)
        dka = tang / dseglen
        dk1 = self.alp * dka
        dtheta = (dka - dk1) / dseglen * dseg**2 + dk1 * dseg
        dx_raw = np.trapezoid(np.sin(dtheta), dseg)
        dy_raw = np.trapezoid(np.cos(dtheta), dseg)

        prox_rotmat = np.array([[np.cos(-pang), -np.sin(-pang)],
                                [np.sin(-pang),np.cos(-pang)]])
        dpos_raw = np.array([dx_raw, dy_raw])
        dpos_rot = prox_rotmat @ dpos_raw
        dx_rot = dpos_rot[0]
        dy_rot = dpos_rot[1]
        dx = dx_rot + px
        dy = dy_rot + py
        # dx = dx_raw + px
        # dy = dy_raw + py

        return px, py, dx, dy