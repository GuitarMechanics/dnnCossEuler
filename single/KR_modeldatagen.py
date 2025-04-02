import keras
import pandas as pd
import numpy as np
import os
os.system('cls') if os.name == 'nt' else os.system('clear')

KRmodel = keras.models.load_model('250324_KRmodel.keras')

data = pd.read_csv('curvature_reginfos.csv')
LR_ratio = []
tip_angle = []
KR_cosserat = []
KR_predict = []

for i, rows in data.iterrows():
    LR_ratio.append(rows['LR_ratio'])
    tip_angle.append(rows['tipang'])
    KR_cosserat.append(rows['Kratio'])
    KR_predict.append(KRmodel.predict(np.array([[rows['LR_ratio'],rows['tipang']]]))[0][0])

pd.DataFrame({'LR_ratio' : LR_ratio,
              'tip_angle' : tip_angle,
              'KR_cosserat' : KR_cosserat,
              'KR_predict' : KR_predict}).to_csv('KR_predict.csv',index=False)