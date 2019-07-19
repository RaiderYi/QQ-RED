#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras import optimizers
data = pd.read_csv('haiyaochuli.csv',header=None,encoding='ISO-8859-1')
data.columns = ['序列','金额']
data=data.fillna(0)
data_use=data[data['序列']<=int(8)]
t=data_use['金额'].shift(-1)
data_use['目标']=t

data_p1 = data_use[int(0)<data_use['序列']]
data_processed =data_p1[data_use['序列']<int(8)]
data_processed=data_processed.fillna(0)
X= np.array(data_processed[['序列','金额']])
y= data_processed['目标']
plt.figure()
plt.plot(data_processed['金额'])
plt.figure()
plt.scatter(data_processed['序列'],data_processed['金额'])
plt.show()
#EX_x='1.33	1.54	1.28	0.8	0.05	1.6	1.29	1.61	0.01	1.96	0.5	0.74	1.89	1.29	0.29	1.6	0.84	1.49	0.19	1.54	1.27	0.58	0.54	1.31	1.02	1.18	1.19	0.1	0.34	1.55	1.68	1.08	1.34	0.74	0.91	0.13	0.66	1.23	1.26	0.55	0.21'

#EX_y='1.54	1.28	0.8	0.05	1.6	1.29	1.61	0.01	1.96	0.5	0.74	1.89	1.29	0.29	1.6	0.84	1.49	0.19	1.54	1.27	0.58	0.54	1.31	1.02	1.18	1.19	0.1	0.34	1.55	1.68	1.08	1.34	0.74	0.91	0.13	0.66	1.23	1.26	0.55	0.21	1.29'
'''
m=EX_x.split()
m_1= list(map(float, m))
m_2=[]
n=EX_y.split()
n_1 = list(map(float, n))
n_2=[]
for i in range(0,len(m_1)):
    m_2.append((m_1[i]*100)%10)
for i in range(0,len(n_1)):
    n_2.append((n_1[i]*100)%10)
'''
#X=np.array(pd.DataFrame(list(map(float, m_2))))
#y=pd.DataFrame(list(map(float, n_2)))
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#X_train = np.reshape(X_train, (X_train.shape[0],1, 2))
#X_test = np.reshape(X_test, (X_tes t.shape[0], 1, 2))
#X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
#X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#X = np.reshape(X, (X.shape[0],1, 1))
# create model
'''model = Sequential()
#model.add(Dense(1, input_dim=1, init='uniform', activation='relu')) #input layer
model.add(Dense(50, input_shape=(1,),activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(10,activation='relu'))
model.add(Dense(1, activation='relu'))#output layer
# Optimizers
#RMSProp = optimizers.RMSprop(lr=0.1,rho=0.9,epsilon=None,decay=0)
# Compile model
model.compile(optimizer='sgd',loss='logcosh', )

# Fit the model
history=model.fit(X[:-4], y[:-4],validation_split=0,nb_epoch=40, batch_size=1)
# evaluate the model
# 随机数参数
# evaluate the model
# 随机数参数
pred_test_y = model.predict(X[-4:-1])
from keras.utils import plot_model
#plot_model(model, to_file='model.png')


#history = model.fit(X, y, validation_split=0.25, epochs=20, batch_size=16, verbose=1)





# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
#lt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.figure(figsize=(8, 4), dpi=80)
plt.plot(range(len(X[-4:-1])), y[-4:-1], ls='-.',lw=2,c='r',label='True_Value')
plt.plot(range(len(X[-4:-1])), pred_test_y, ls='-',lw=2,c='b',label='Predict_Value')
plt.legend()
plt.show()'''
