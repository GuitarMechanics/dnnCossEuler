import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dualGaraKRNew as dgk

tdcls = dgk.dualGaraKRNew(3.4,33,61)

df = pd.read_csv('dual/garaKRdual.csv')

px = []
py = []
dx = []
dy = []

for i, row in df.iterrows():
    proxang = row['ProxTarget']
    distang = row['DistTarget']
    prox_x, prox_y, dist_x, dist_y = tdcls.getPos(proxang, distang)
    px.append(prox_x)
    py.append(prox_y)
    dx.append(dist_x)
    dy.append(dist_y)


newdf = pd.DataFrame({
    'prox_x': px,
    'prox_y': py,
    'dist_x': dx,
    'dist_y': dy
})

newdf.to_csv('dual/dualGaraKRPos.csv', index=False)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(newdf['prox_x'], newdf['prox_y'], label='Proximal', color='blue')
plt.plot(newdf['dist_x'], newdf['dist_y'], label='Distal', color='orange')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Proximal and Distal Positions')
plt.legend()
plt.grid()
plt.show()