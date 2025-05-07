from NewGaraEuler import NewGaraEuler
import pandas as pd
import numpy as np


sb = NewGaraEuler(62,3.4)
tdls = np.linspace(0,8.5,1000)

tdl = []
xpos = []
ypos = []
displacement = [0]
totaldisplacement = [0]

for t in tdls:
    tdl.append(t)
    x, y = sb.getTipPos(g_cond = 0, tdl = t)
    xpos.append(x)
    ypos.append(y)

for i in range(len(xpos)-1):
    displacement.append(np.sqrt((xpos[i+1]-xpos[i])**2 + (ypos[i+1]-ypos[i])**2))
    totaldisplacement.append(totaldisplacement[i] + displacement[i+1])

df = pd.DataFrame({'tdl': tdl,
                   'xpos': xpos, 
                   'ypos': ypos, 
                   'displacement': displacement,
                   'totaldisplacement': totaldisplacement})
df.to_csv('single/singletipdisplacement_nup.csv', index=False)