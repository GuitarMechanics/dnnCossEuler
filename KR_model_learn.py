import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import os

data = pd.read_csv('dualbend/dnntraindatas_pos.csv')
x = []
x = []
for i, rows in data.iterrows():
    x.append([rows['phor'],rows['pver'],rows['thor'],rows['tver']])
xnd = np.array(x)
kfdata = KFold(9,shuffle=True)

z = []
for i, rows in data.iterrows():
    z.append([rows['proxtdl'],rows['disttdl']])
znd = np.array(z)
print(znd)
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

layers = 3
inits = keras.initializers.glorot_normal(seed=None)
opt = keras.optimizers.Adam()
fkinmodel = keras.Sequential()
fkinmodel.add(keras.layers.Dense(units=2,activation='linear',input_dim=2))
for i in range(layers):
    #ikinmodel.add(keras.layers.BatchNormalization())
    fkinmodel.add(keras.layers.Dense(units=192,activation='sigmoid',use_bias=True))
    fkinmodel.add(keras.layers.Dropout(0.3))
fkinmodel.add(keras.layers.BatchNormalization())
fkinmodel.add(keras.layers.Dense(4,activation='linear'))

val_loss_record = [] ## log of avglos
fkinmodel.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

loopcount = 5000
end_stack = 0

try:
    for i in range (loopcount):
        print(f'---------------train loop: {i} / {loopcount}------------')
        validation_accuracy = []
        validation_loss = []
        for train, test in kfdata.split(z):
            history = fkinmodel.fit(znd[train], xnd[train], epochs=300,
                                validation_data = (znd[test],xnd[test]),
                                callbacks=keras.callbacks.EarlyStopping(
                                    patience=50,
                                    restore_best_weights=True,
                                ))
            evaldata = fkinmodel.evaluate(znd[test],xnd[test])
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

        if avgacc >= 0.99 and avglos < 0.2:
            end_stack += 1
        else:
            end_stack = 0

        val_loss_record.append([i+1,avglos,avgacc])
        if(end_stack >= 3):
            break
except(KeyboardInterrupt):
    print("stopped by keyboard input")

# print("post-train")
# model2.fit(znd, xnd, epochs=50) ##post-train after k-fold
appendKFoldLoss(fkinmodel, z, znd, xnd, 'final', val_loss_record)

#model.save(f'dualbend/models/dual_sigmoids_kfold_lowep_pos_layers{hiddenlayers}.keras')
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = 'dualbend/models/dual_sigmoids_kfold_history.csv'
# with open(hist_csv_file, mode ='w') as f:
#     hist_df.to_csv(f)
pd.DataFrame(val_loss_record).to_csv(f'dualbend/models/fkin241225_dataappend_avglos_rec_forlearningcurve.csv')

fkinmodel.save(f'dualbend/models/fkin241225_dataappend.keras')

## Fkin POS chart generation
tdldata = pd.read_csv('ikin241112_dataappend_datas.csv')
targetangs = []
tdls = []
for i, rows in tdldata.iterrows():
    targetangs.append([rows['proxtarg'],rows['disttarg']])
    tdls.append([rows['proxtdl'],rows['disttdl']])
targetNdarr = np.array(targetangs)
tdlsNdarr = np.array(tdls)

posPredict = np.array(fkinmodel.predict(tdlsNdarr).tolist())
outNdArr = np.hstack((targetNdarr,posPredict))
print(outNdArr)
pd.DataFrame(outNdArr).to_csv('dualbend/fkin241225_dataappend_datas.csv',\
                              header=['proxtarg','disttarg','proxh','proxv','tiph','tipv'],index=None)

print("git uploading")
os.system('git add .')
os.system("git commit -a -m'autocommit_fkin241225_dataappend'")
os.system("git push origin master")

print('shutdown scheduled')
os.system('shutdown -h 1') ## auto-shutdown when done