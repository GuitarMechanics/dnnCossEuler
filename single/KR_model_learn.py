import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import os

data = pd.read_csv('curvature_reginfos.csv')
x = []
z = []
for i, rows in data.iterrows():
    x.append([rows['Kratio']])
    z.append([rows['LR_ratio'],rows['tipang']])
xnd = np.array(x)
znd = np.array(z)
kfdata = KFold(21,shuffle=True)


def appendKFoldLoss(model, x, xnd, ynd, text, val_loss_record):
    validation_accuracy = []
    validation_loss = []
    for train, test in kfdata.split(x):
        evaldata = model.evaluate(xnd[test],ynd[test])
        validation_accuracy.append(evaldata[1])
        validation_loss.append(evaldata[0])
    avgacc = 0
    avglos = 0
    for vals in validation_accuracy:
        avgacc += vals
    avgacc /= len(validation_accuracy)
    for vals in validation_loss:
        avglos += vals
    avglos /= len(validation_loss)
    val_loss_record.append([text,avglos])

layers = 2
inits = keras.initializers.glorot_normal(seed=None)
opt = keras.optimizers.Adam()
KRmodel = keras.Sequential()
KRmodel.add(keras.layers.Dense(units=2,activation='linear',input_dim=2))
for i in range(layers):
    #ikinmodel.add(keras.layers.BatchNormalization())
    KRmodel.add(keras.layers.Dense(units=192,activation='sigmoid',use_bias=True))
    KRmodel.add(keras.layers.Dropout(0.3))
KRmodel.add(keras.layers.BatchNormalization())
KRmodel.add(keras.layers.Dense(1,activation='linear'))

val_loss_record = [] ## log of avglos
KRmodel.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])

loopcount = 5000
end_stack = 0

try:
    for i in range (loopcount):
        print(f'---------------train loop: {i} / {loopcount}------------')
        validation_mse = []
        validation_loss = []
        for train, test in kfdata.split(z):
            history = KRmodel.fit(znd[train], xnd[train], epochs=300,
                                validation_data = (znd[test],xnd[test]),
                                callbacks=keras.callbacks.EarlyStopping(
                                    patience=50,
                                    restore_best_weights=True,
                                ))
            evaldata = KRmodel.evaluate(znd[test],xnd[test])
            validation_mse.append(evaldata[1])
            validation_loss.append(evaldata[0])
        avgmse = 0
        avglos = 0
        for vals in validation_mse:
            avgmse += vals
        avgmse /= len(validation_mse)
        for vals in validation_loss:
            avglos += vals
        avglos /= len(validation_loss)

        if avgmse <= 5e-6:
            end_stack += 1
        else:
            end_stack = 0

        val_loss_record.append([i+1,avglos,avgmse])
        if(end_stack >= 3):
            break
except(KeyboardInterrupt):
    print("stopped by keyboard input")

# print("post-train")
# model2.fit(znd, xnd, epochs=50) ##post-train after k-fold
appendKFoldLoss(KRmodel, z, znd, xnd, 'final', val_loss_record)

#model.save(f'dualbend/models/dual_sigmoids_kfold_lowep_pos_layers{hiddenlayers}.keras')
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = 'dualbend/models/dual_sigmoids_kfold_history.csv'
# with open(hist_csv_file, mode ='w') as f:
#     hist_df.to_csv(f)
pd.DataFrame(val_loss_record).to_csv(f'KRmodel_avgmse_forlearningcurve.csv')

KRmodel.save(f'250324_KRmodel.keras')

print("git uploading")
os.system('git add .')
os.system("git commit -a -m'autocommit_KRmodelLearn'")
os.system("git push origin master")