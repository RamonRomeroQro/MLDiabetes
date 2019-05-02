from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

width = 5
model = Sequential()
model.add(Dense(width, input_shape=width))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['accuracy'])

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

import pandas as pd
import numpy as np
df=pd.read_csv('test.csv')
#df = df.values

given = df.iloc[:,:5]
expected = df.iloc[:,-1]
print(given.values)
print(expected.values)

model.fit(given, expected,
          epochs=20,
          batch_size=128)
s= np.array([1  , 1,  0 , 3 ,25 ])

t=pd.read_csv('ok.csv')

print(model.predict(t.values))
# score = model.evaluate(x_test, y_test, batch_size=128)