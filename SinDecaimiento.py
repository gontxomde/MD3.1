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

y = [1 if line[100] == 1 else 0 for line in train]
y_test  = [1 if line[100] == 1 else 0 for line in val]
X = train[:,:100]
X_test = val[:,:100]

def build_model(inner_neurons = 51, INTERVAL = 1):
    ru = RU(minval = -INTERVAL, maxval = INTERVAL)
    model = Sequential()
    model.add(Dense(inner_neurons, input_dim = 100, activation = 'relu', kernel_initializer = ru, 
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    return model

EPOCHS = 1000
RUNS = 10
INTERVAL = 1
prefix = '7_1_bwd'

loss_train = np.zeros((RUNS,EPOCHS))
acc_train = np.zeros((RUNS,EPOCHS))
loss_test = np.zeros((RUNS,EPOCHS))
acc_test = np.zeros((RUNS,EPOCHS))

sgd = SGD(lr = 0.01)
ru = RU(minval = -INTERVAL, maxval = INTERVAL)


    
for i in range (RUNS):
    model = build_model(inner_neurons = 52, INTERVAL = INTERVAL)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    history = model.fit(X, y, epochs = EPOCHS, batch_size = 10, verbose = 0, validation_data=(X_test, y_test))


    loss_train[i,:] = history.history['loss']
    loss_test[i,:] = history.history['val_loss']

    acc_train[i,:] = history.history['acc']
    acc_test[i,:] = history.history['val_acc']
    print(i)
        
print(history.history['acc'][-1])

for execution in loss_train:
    sns.lineplot(x = list(range(EPOCHS)), y = execution)

def dataframe_from_np(EPOCHS, np_array, columns):
    aux = np.empty((EPOCHS,2))
    aux[:,0] = range(EPOCHS)
    aux[:,1] = np_array[0]
    
    aux_junto = np.copy(aux)
    
    for exe in np_array[1:]:
        aux[:,1] = exe
        aux_junto = np.append(aux_junto,aux, axis = 0)
    df = pd.DataFrame(data = aux_junto, columns = columns)
    return df

def df_completo(EPOCHS, numpy_train, numpy_test, columns):
    train_df = dataframe_from_np(EPOCHS, numpy_train, columns[:2])
    test_df = dataframe_from_np(EPOCHS, numpy_test, columns[:2])
    
    train_df['Fase'] = 'Train'
    test_df['Fase'] = 'Test'
    
    df = train_df.append(test_df)
    return df

accuracy = df_completo(EPOCHS, acc_train, acc_test, ['Epoch', 'Accuracy'])
loss     = df_completo(EPOCHS, loss_train, loss_test, ['Epoch', 'Loss'])

accuracy.to_csv(f'{prefix}_acc')
loss.to_csv(f'{prefix}_loss')

ax = sns.lineplot(x = 'Epoch', y = 'Accuracy', data = accuracy, hue = 'Fase')
ax.get_figure().savefig(f'Figuras/{prefix}_acc.png')

ax = sns.lineplot(x = 'Epoch', y = 'Loss', data = loss, hue = 'Fase')
ax.get_figure().savefig(f'Figuras/{prefix}_loss.png')