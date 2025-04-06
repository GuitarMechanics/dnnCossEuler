import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

exec(open('dual/dualGaraKRCalc.py').read())

meltdfDist = pd.read_csv('dual/meltdfDist.csv')
meltdfProx = pd.read_csv('dual/meltdfProx.csv')

plt.figure(figsize = (15,9))
plt.subplot(2,1,1)
sns.boxplot(data = meltdfProx,
            x = 'ProxTarget',
            y = 'TDL',
            hue = 'Method',
            palette='Set2')
plt.subplot(2,1,2)
sns.boxplot(data = meltdfDist,
            x = 'DistTarget',
            y = 'TDL',
            hue = 'Method',
            palette='Set2')
plt.show()