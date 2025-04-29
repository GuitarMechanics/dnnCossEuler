import numpy as np
import pandas as pd
from dualNewFKin_ver2 import dualNewFKin
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.system('cls' if os.name == 'nt' else 'clear')

db = dualNewFKin(3.4, 33, 61)
df = pd.read_csv('dual/dualdeep.csv')

ptarg = []
dtarg = []
ptdl = []
dtdl = []
pa = []
ta = []
retptdl = []
retdtdl = []
retpang = []
retdang = []
px = []
py = []
tx = []
ty = []

for i, row in df.iterrows():
    ptdl.append(row['proxtdl'])
    dtdl.append(row['disttdl'])
    ptarg.append(row['proxtarg'])
    dtarg.append(row['disttarg'])

    tpret, tdret = db.getTrueTDL(ptarg[-1],dtarg[-1],degrees=True)
    retptdl.append(tpret)
    retdtdl.append(tdret)
    pa.append(row['prox_ang'])
    ta.append(row['dist_ang'])

    # retpa, retda = db.retreiveAngs(ptdl[-1],dtdl[-1])
    # retpang.append(retpa)
    # retdang.append(retda)
    pxpos, pypos, txpos, typos = db.getPos(ptarg[-1], dtarg[-1], degrees=True)
    px.append(pxpos)
    py.append(pypos)
    tx.append(txpos)
    ty.append(typos)

newdf = pd.DataFrame({
    'proxtarg':ptarg,
    'disttarg':dtarg,
    'ptdl':ptdl,
    'dtdl':dtdl,
    'exproxang':pa,
    'exdistang':ta,
    # 'ret_pa':retpang,
    # 'ret_da':retdang,
    'ret_tp':retptdl,
    'ret_td':retdtdl,
    'prox_x':px,
    'prox_y':py,
    'dist_x':tx,
    'dist_y':ty
})
newdf.to_csv('dual/newfkin_datas_ver2_withpos.csv',index=None)
expdf = pd.read_csv('dual/dualexpdata.csv')
plt.figure()
plt.scatter(data = newdf, x = 'prox_x', y = 'prox_y', label = 'NEWFkinProx')
plt.scatter(data = newdf, x = 'dist_x', y = 'dist_y', label = 'NEWFkinDist')
plt.scatter(data = expdf, x = 'phor_rev', y = 'EXP_Prox', label = 'EXPProx')
plt.scatter(data = expdf, x = 'dhor_rev', y = 'EXP_Dist', label = 'EXPDist')
plt.legend()
plt.show()