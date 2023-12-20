import tensorflow
from tensorflow import keras
import numpy

model = keras.Sequential([
  keras.layers.Dense(units=1,
input_shape=[1])
])

model.compile(optimizer = 'sgd',
loss ='mean_squared_error')

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0],
dtype=float)
y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0],
dtype=float)

model.fit(x_train, y_train, epochs=500)

print(model.predict([6.0]))