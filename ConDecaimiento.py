import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import RandomUniform as RU
from keras import regularizers

train = pd.read_csv('tra2.pat', header = None, sep = ' ')
val = pd.read_csv('val2.pat', header = None, sep = ' ')

train = train.to_numpy()
val = val.to_numpy()

def build_model(inner_neurons = 51, INTERVAL = 1):
    ru = RU(minval = -INTERVAL, maxval = INTERVAL)
    model = Sequential()
    model.add(Dense(inner_neurons, input_dim = 100, activation = 'relu', kernel_initializer = ru,
                kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation = 'sigmoid',
                kernel_regularizer=regularizers.l2(0.01)))
    return model


EPOCHS = 500
RUNS = 10
INTERVAL = 1
NEURONS = [50, 40, 30, 20, 10, 2]
prefix = 'dism_1_bwd'

loss_train_dict = {}
loss_test_dict = {}
acc_train_dict = {}
acc_test_dict = {}

sgd = SGD(lr = 0.01)
ru = RU(minval = -INTERVAL, maxval = INTERVAL)


for j in range(len(NEURONS)):
    
    loss_train = np.zeros((RUNS,EPOCHS))
    acc_train = np.zeros((RUNS,EPOCHS))
    loss_test = np.zeros((RUNS,EPOCHS))
    acc_test = np.zeros((RUNS,EPOCHS))
    
    for i in range (RUNS):
        model = build_model(inner_neurons = NEURONS[j], INTERVAL = INTERVAL)
        model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
        history = model.fit(X, y, epochs = EPOCHS, batch_size = 10, verbose = 0, validation_data=(X_test, y_test))


        loss_train[i,:] = history.history['loss']
        loss_test[i,:] = history.history['val_loss']

        acc_train[i,:] = history.history['acc']
        acc_test[i,:] = history.history['val_acc']
        
        
        print(f'{i} run. {NEURONS[j]} neuronas.')
    loss_train_dict[NEURONS[j]] = loss_train
    loss_test_dict[NEURONS[j]] = loss_test
    acc_train_dict[NEURONS[j]] = acc_train
    acc_test_dict[NEURONS[j]] = acc_test
    
print(history.history['acc'][-1])

import pickle

with open(f'{prefix}_4diccionarios.pkl', 'wb') as f:
    pickle.dump([loss_train_dict, loss_test_dict, acc_train_dict, acc_test_dict], f)

def dict_fin(dictionary_final):
    loss_train_dict_fin = {}
    for key in dictionary_final:
        loss_train_dict_fin[key] = dictionary_final[key][:,-1]
    return loss_train_dict_fin
    
def dism_to_df(RUNS, dictionary, name):
    dict_final = dict_fin(dictionary)
    df = pd.DataFrame()
    df_aux = pd.DataFrame()

    for i in dict_final:
        df_aux['i'] = list(range(RUNS))
        df_aux[name] = dict_final[i]
        df_aux['Neurons'] = i

        df = df.append(df_aux, ignore_index = True)
    return df
def dism_df_definitivo(RUNS, dict_train, dict_test, name):
    df_train = dism_to_df(RUNS, dict_train, name)
    df_train['Fase'] = 'Train'
    
    df_test = dism_to_df(RUNS, dict_test, name)
    df_test['Fase'] = 'Test'
    
    df = df_train.append(df_test, ignore_index = True)
    return df

df_dism_loss = dism_df_definitivo(RUNS, loss_train_dict, loss_test_dict, 'Loss')
df_dism_acc = dism_df_definitivo(RUNS, acc_train_dict, acc_test_dict, 'Accuracy')

ax = sns.lineplot(x = 'Neurons', y = 'Loss', data = df_dism_loss, hue = 'Fase')

ax.get_figure().savefig(f'Figuras/{prefix}_loss.png')

ax = sns.lineplot(x = 'Neurons', y = 'Accuracy', data = df_dism_acc, hue = 'Fase')
ax.get_figure().savefig(f'Figuras/{prefix}_acc.png')