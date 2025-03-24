import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import os

auto_shutdown = False
auto_git = True

print('program setups : ')
print('auto_git = ',auto_git,'\nauto_shutdown = ',auto_shutdown)
input('continue?')

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
    val_loss_record.append([text,avglos,avgacc])
## get loss curve
hiddenlayers = [1,2,3,4,5,6,7,8]
layernodes = [32,64,96,128,160,192]

writer = pd.ExcelWriter('KRmodel_lcurvedatas.xlsx',mode='w')
sheetHeader = ['loop','kfoldloss','kfoldaccuracy']

for nodes in layernodes:
    for layers in hiddenlayers:
        print(f'================ ++ node / layer: {nodes} / {layers} ++ ================')
        inits = keras.initializers.he_normal(seed=None)
        opt = keras.optimizers.Adam()
        sgd = keras.optimizers.SGD()
        model2 = keras.Sequential()
        model2.add(keras.layers.Dense(units=2,activation='linear',input_dim=2))
        for i in range(layers):
            model2.add(keras.layers.Dense(units=nodes,activation='sigmoid',use_bias=True))
            model2.add(keras.layers.Dropout(0.3))
        model2.add(keras.layers.Dense(1,activation='linear'))

        val_loss_record2 = [] ## log of avglos
        model2.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
        # model2.fit(znd, xnd, epochs=50) ##pre-train before k-fold

        # appendKFoldLoss(model2,z, znd, xnd, 'pre-train', val_loss_record2) 

        loopcount = 5
        try:
            for i in range (loopcount):
                print(f'================ ++ train loop: {i} / {loopcount} ++ ================')
                validation_accuracy = []
                validation_loss = []
                for train, test in kfdata.split(z):##300/50 // 100/30
                    history = model2.fit(znd[train], xnd[train], epochs=200,
                                        validation_data = (znd[test],xnd[test]),
                                        callbacks=keras.callbacks.EarlyStopping(
                                            patience=30,
                                            restore_best_weights=True
                                        ))
                    evaldata = model2.evaluate(znd[test],xnd[test])
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

                # if avgacc >= 0.999 and avglos < 0.1:
                #     break
                val_loss_record2.append([i+1,avglos,avgacc])
        except(KeyboardInterrupt):
            print("stopped by keyboard input")

        # print("post-train")
        # model2.fit(znd, xnd, epochs=50) ##post-train after k-fold
        appendKFoldLoss(model2, z, znd, xnd, 'final', val_loss_record2)

        #model.save(f'dualbend/models/dual_sigmoids_kfold_lowep_pos_layers{hiddenlayers}.keras')
        # hist_df = pd.DataFrame(history.history)
        # hist_csv_file = 'dualbend/models/dual_sigmoids_kfold_history.csv'
        # with open(hist_csv_file, mode ='w') as f:
        #     hist_df.to_csv(f)
        pd.DataFrame(val_loss_record2).to_excel(writer,header=sheetHeader,index=False,sheet_name=f'nodes{nodes}_layers{layers}')

writer.close()
outlist = []
layerheader = ['nodes','layers','val_loss','val_acc']
for nodes in layernodes:
    for layers in hiddenlayers:
        df = pd.read_excel('KRmodel_lcurvedatas.xlsx',sheet_name=f'nodes{nodes}_layers{layers}').to_numpy()
        print(f"node={nodes} layers={layers}",df[-1,1:2])
        outlist.append([nodes,layers,df[-1,1],df[-1,2]])
pd.DataFrame(outlist).to_csv('KRmodel_lcurvesummary.csv',header=layerheader,index=False)


if(auto_git):
    os.system('git add .')
    os.system('git commit -a -m"autocommit by KR_model_findbeststruct.py"')
    os.system('git push origin master')

if(auto_shutdown):
    print('shutdown scheduled')
    os.system('shutdown -h 1') ## auto-shutdown when done