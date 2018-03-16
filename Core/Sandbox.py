import math as m
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts

l1=1
l2=1
x_vals=np.array()
y_vals=np.array()
t2_vals=np.array()
t1_vals=np.array()
count=0
rails = np.linspace(0,10,num=100)

for val in rails:
    x=val
    for val2 in rails:
        y=val2
        magic = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if 1 >= magic >= -1:
            t2_vals.append(m.acos(magic))
            x_vals.append(x)
            y_vals.append(y)
            count+=1

for t2 in t2_vals:

    t1 = ((-(l2*m.sin(t2))*x+(l1+l2*m.cos(t2))*y)/(l2*m.sin(t2)*y+(l1+l2*m.cos(t2))*x))
    t1_vals.append(t1)

x_train, x_test, y_train, y_test = tts(x_vals, y_vals, test_size = 0.33)

print(x_train.shape)

# train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x":(x_train)},
#                                                     y = {"y":(y_train)},
#                                                     batch_size=10,
#                                                     num_epochs = None,
#                                                     shuffle = True)
#
# test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":np.asarray(x_test)},
#                                                    num_epochs = 1,
#                                                    shuffle = False)
#
# feature_x = [tf.feature_column.numeric_column("x",shape=[1])]
#
# estimator = tf.estimator.LinearRegressor(feature_columns=feature_x)
#
# estimator.train(input_fn=train_input_fn,steps=1000)