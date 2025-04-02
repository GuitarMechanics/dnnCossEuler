import pandas as pd
import numpy as np
import scipy.optimize as opt
import scipy.integrate as itg

len = 62
rad = 3.4

def theta(s, tdl, ki):
    ka = tdl / (len * rad)
    return (ka - ki) / len * s ** 2 + ki * s

def safe_fsolve(func, guess):
    try:
        sol = opt.fsolve(func, guess, xtol=1e-6)
        if np.isfinite(sol[0]):
            return sol[0]
        else:
            raise ValueError
    except:
        # fallback to root_scalar with a bracket if fsolve fails
        try:
            result = opt.root_scalar(func, bracket=[guess * 0.5, guess * 1.5], method='brentq')
            if result.converged:
                return result.root
        except:
            pass
    return np.nan  # return NaN if everything fails
writer = pd.ExcelWriter('KR_obtained_revnoop_kavg_w_ang.xlsx',mode='w')

sheet_name = ['std_ess', 'rev_ess', 'nup_ess']
for sheet in sheet_name:
    df = pd.read_excel('revnoopdong_essentials.xlsx',
                       sheet_name = sheet)
    dftdl = []
    dfka = []
    dfkix = []
    dfkiy = []
    dfKRx = []
    dfKRy = []
    dfkixy= []
    dfKRxy= []
    for _ , row in df.iterrows():
        tdl = row['TDL']
        x = row['hor']
        y = row['ver']
        ang = row['ang']
        dftdl.append(tdl)
        # ka = tdl / (len * rad)
        ka = ang / len
        dfka.append(ka)

        def equation_x(ki):
            result, _ = itg.quad(lambda s: np.sin(theta(s, tdl, ki)),0,len)
            return result + x
        
        def equation_y(ki):
            result, _ = itg.quad(lambda s: np.cos(theta(s, tdl, ki)),0,len)
            return result - y
        
        def equation_tot(ki):
            resx, _ = itg.quad(lambda s: np.sin(theta(s, tdl, ki)),0,len)
            resy, _ = itg.quad(lambda s: np.cos(theta(s, tdl, ki)),0,len)

            ret = np.sqrt((resx + x) ** 2 + (resy - y)**2)
            return ret

        solx = safe_fsolve(equation_x, ka * 1.5)
        kix = solx
        dfkix.append(kix)
        krx = kix / ka
        dfKRx.append(krx)

        soly = safe_fsolve(equation_y, ka * 1.5)
        kiy = soly
        dfkiy.append(kiy)
        kry = kiy / ka
        dfKRy.append(kry)

        solxy = safe_fsolve(equation_tot, ka * 1.5)
        kixy = solxy
        dfkixy.append(kixy)
        krxy = kixy / ka
        dfKRxy.append(krxy)

    newdf = pd.DataFrame({'TDL':dftdl,
                          'ka': dfka,
                          'kix': dfkix,
                          'KRx': dfKRx,
                          'kiy': dfkiy,
                          'KRy': dfKRy,
                          'kixy': dfkixy,
                          'KRxy': dfKRxy})
    print(newdf)
    pd.DataFrame(newdf).to_excel(writer,index=False,sheet_name=sheet)

writer.close()