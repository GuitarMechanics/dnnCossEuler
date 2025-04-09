import dualGaraKRNew as dgk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.system('cls' if os.name == 'nt' else 'clear')

df = pd.read_csv('dual/prev_dnn_datas.csv')
ptarg = []
dtarg = []
tpecavc = []
tdecavc = []
tpdnn = []
tddnn = []
tpnew = []
tdnew = []

tdcls = dgk.dualGaraKRNew(3.4,33,61)

for i, row in df.iterrows():
    ptarg.append(int(row['proxtarg']))
    dtarg.append(int(row['disttarg']))
    tpecavc.append(float(row['ecprox']))
    tdecavc.append(float(row['ecdist']))
    tpdnn.append(float(row['dnprox']))
    tddnn.append(float(row['dndist']))
    pnew, dnew = tdcls.getTDLs(np.deg2rad(int(row['proxtarg'])),np.deg2rad(int(row['disttarg'])))
    tpnew.append(pnew)
    tdnew.append(dnew)

newdf = pd.DataFrame({'ProxTarget' : ptarg,
                      'DistTarget' : dtarg,
                      'ECAVCProx' : tpecavc,
                      'ECAVCDist' : tdecavc,
                      'DNNProx' : tpdnn,
                      'DNNDist' : tddnn,
                      'NewProx' : tpnew,
                      'NewDist' : tdnew})
newdf.to_csv('dual/garaKRdual.csv')

meltdfProx = newdf.melt(id_vars=['ProxTarget'], value_vars=['ECAVCProx',
                                                            'DNNProx',
                                                            'NewProx',
                                                        'ECAVCDist',
                                                        'DNNDist',
                                                        'NewDist'],
                                                        var_name='Method',
                                                        value_name='TDL')
meltdfDist = newdf.melt(id_vars=['DistTarget'], value_vars=['ECAVCProx',
                                                            'DNNProx',
                                                            'NewProx',
                                                        'ECAVCDist',
                                                        'DNNDist',
                                                        'NewDist'],
                                                        var_name='Method',
                                                        value_name='TDL')

meltdfProx.to_csv('dual/meltdfProx.csv')
meltdfDist.to_csv('dual/meltdfDist.csv')