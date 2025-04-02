import seaborn as sns
import matplotlib.pyplot as plt
import keras
import numpy as np
import os

os.system('cls') if os.name == 'nt' else os.system('clear')

model = keras.models.load_model('250324_KRmodel.keras')

angles = np.linspace(0,2,11)
ratios = np.linspace(0,50,21)

for ang in angles:
    predict = []
    for rat in ratios:
        predict.append(model.predict(np.array([[rat,ang]]))[0][0])
    plt.plot(ratios,predict,label=f'angle={np.round(ang)}')
plt.legend()
plt.show()   