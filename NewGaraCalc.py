import NewGaraEuler as nge
import pandas as pd
import numpy as np
import os

os.system('cls') if os.name == 'nt' else os.system('clear')

sheet_name = ['std_ess', 'rev_ess', 'nup_ess']
def getGcondFromSheet(sheet):
    if sheet == 'std_ess':
        return 1
    elif sheet == 'rev_ess':
        return -1
    elif sheet == 'nup_ess':
        return 0
    else:
        raise Exception('Invalid sheet name')
    
bb = nge.NewGaraEuler(62, 3.4)
writer = pd.ExcelWriter('garaKR_errors.xlsx',mode='w')
for sheet in sheet_name:
    ordf = pd.read_excel('revnoopdong_essentials.xlsx',
                        sheet_name = sheet)
    dftdl = []
    dfexpx= []
    dfexpy= []
    dfmodx= []
    dfmody= []
    dfxerr= []
    dfyerr= []
    dfperr= []
    dfaang= []
    dfeang= []
    dfaerr= []
    for i, row in ordf.iterrows():
        tdl = row['TDL']
        x = row['hor']
        y = row['ver']
        ang = row['ang']
        g_cond = getGcondFromSheet(sheet)
        mx, my = bb.getTipPos(g_cond, tdl)

        dftdl.append(tdl)
        dfexpx.append(x)
        dfexpy.append(y)
        dfmodx.append(-mx)
        dfmody.append(my)
        dfxerr.append(x + mx)
        dfyerr.append(y - my)
        dfperr.append(np.sqrt((x + mx) ** 2 + (y - my) ** 2))
        dfaang.append(np.rad2deg(tdl / 3.4))
        dfeang.append(ang)
        dfaerr.append(ang - dfaang[-1])
    newdf = pd.DataFrame({'TDL':dftdl,
                          'ExpX': dfexpx,
                          'ExpY': dfexpy,
                          'ModX': dfmodx,
                          'ModY': dfmody,
                          'Xerr': dfxerr,
                          'Yerr': dfyerr,
                          'Perr': dfperr,
                          'AnalAng': dfaang,
                          'ModAng': dfeang,
                          'Aerr': dfaerr})
    print(newdf)
    newdf.to_excel(writer, sheet_name = sheet, index = False)
writer.close()