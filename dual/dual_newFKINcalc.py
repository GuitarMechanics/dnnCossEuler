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
px = []
py = []
tx = []
ty = []
dp = []
tpeff = []
tdeff = []

for i, row in df.iterrows():
    ptdl.append(row['proxtdl'])
    dtdl.append(row['disttdl'])
    ptarg.append(row['proxtarg'])
    dtarg.append(row['disttarg'])
    
    dpval = db.getDP(row['disttdl'])
    dp.append(dpval)

    paval, taval = db.getAngs(ptdl[-1], dtdl[-1],degrees=True)
    pa.append(paval)
    ta.append(taval)

    tpeffval, tdeffval = db.getEffTDL(ptdl[-1],dtdl[-1])
    tpeff.append(tpeffval)
    tdeff.append(tdeffval)

    pxpos, pypos, txpos, typos = db.getPos(ptdl[-1], dtdl[-1])
    px.append(pxpos)
    py.append(pypos)
    tx.append(txpos)
    ty.append(typos)

newdf = pd.DataFrame({
    'proxtarg':ptarg,
    'disttarg':dtarg,
    'ptdl':ptdl,
    'dtdl':dtdl,
    'proxang':pa,
    'distang':ta,
    'prox_x':px,
    'prox_y':py,
    'dist_x':tx,
    'dist_y':ty,
    'dp':dp,
    'tpeff':tpeff,
    'tdeff':tdeff
})
newdf.to_csv('dual/newfkin_datas.csv',index=None)
expdf = pd.read_csv('dual/dualexpdata.csv')
plt.figure()
plt.scatter(data = newdf, x = 'prox_x', y = 'prox_y', label = 'NEWFkinProx')
plt.scatter(data = newdf, x = 'dist_x', y = 'dist_y', label = 'NEWFkinDist')
plt.scatter(data = expdf, x = 'phor_rev', y = 'EXP_Prox', label = 'EXPProx')
plt.scatter(data = expdf, x = 'dhor_rev', y = 'EXP_Dist', label = 'EXPDist')
plt.legend()
plt.show()