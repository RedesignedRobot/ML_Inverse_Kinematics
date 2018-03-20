import math as m
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler as mms

#Creating initial lists

x_vals_raw = []
y_vals_raw = []
t1_vals_raw = []
t2_vals_raw = []
fx_raw=[]
fy_raw=[]
delta_X_raw=[]
delta_Y_raw=[]

#Init required vars

l1=1
l2=1
count=0
rails = np.linspace(0,100,num=1000)
tf.logging.set_verbosity(tf.logging.INFO)

#Normalizing and generating inverse dataset

for val in rails:
    x=val
    for val2 in rails:
        y=val2
        magic = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if 1 >= magic >= -1:
            t2_vals_raw.append(m.acos(magic))
            x_vals_raw.append(x)
            y_vals_raw.append(y)
            count+=1

for t2 in t2_vals_raw:

    t1 = ((-(l2*m.sin(t2))*x+(l1+l2*m.cos(t2))*y)/(l2*m.sin(t2)*y+(l1+l2*m.cos(t2))*x))
    t1_vals_raw.append(t1)

#Converting list to numpy array

x_vals = np.array(x_vals_raw)
y_vals = np.array(y_vals_raw)
t1_vals = np.array(t1_vals_raw)
t2_vals = np.array(t2_vals_raw)

#checking with forward kinematics

for i in range(len(t1_vals)):
    fx_raw.append((l1*m.cos(t1_vals[i]))+(l2*m.cos(t1_vals[i]+t2_vals[i])))
    fy_raw.append((l1*m.sin(t1_vals[i]))+(l2*m.sin(t1_vals[i]+t2_vals[i])))

fx=np.array(fx_raw)
fy=np.array(fy_raw)

#Calculating delta

for i in range(len(t1_vals)):
    delta_X_raw.append(fx[i]-x_vals[i])
    delta_Y_raw.append(fy[i]-y_vals[i])

delta_X = np.array(delta_X_raw)
delta_Y = np.array(delta_Y_raw)

#Converting Rad to Deg

for i in range(len(t1_vals)):
    t1_vals[i]=m.degrees(t1_vals[i])
    t2_vals[i]=m.degrees(t2_vals[i])

#Creating pandas dataframe

data_total = {'Inverse X':x_vals,
        'Inverse Y':y_vals,
        'Forward X':fx,
        'Forward Y':fy,
        'Theta 1 °':t1_vals,
        'Theta 2 °':t2_vals}

x_data_cols = {'Inverse_X':x_vals,
               'Inverse_Y':y_vals}

y_data_cols = {'Theta_1':t1_vals}

# delta = {'Delta X':delta_X,
#         'Delta Y':delta_Y,}

# delta_dataframe=pd.DataFrame(data=delta)

dh_data_x = pd.DataFrame(data=x_data_cols)
dh_data_y = pd.DataFrame(data=y_data_cols)
grand_set = pd.DataFrame(data=data_total)

grand_set.to_csv("grand_set",sep=',')
dh_data_x.to_csv("dh_data_x",sep='\t',index=False,index_label=False)
dh_data_y.to_csv("dh_data_y",sep='\t',index_label=False,index=False)

#Spliting & Preprocessing Data

X_train, X_test, y_train, y_test = tts(dh_data_x, dh_data_y, test_size=0.33, random_state=101)
scaler = mms()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(data=scaler.transform(X_train),
                        columns=X_train.columns,
                        index=X_train.index)

X_test_scaled = pd.DataFrame(data=scaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index)


#Creating feature columns

feature_cols = [tf.feature_column.numeric_column('Inverse_X'),
                tf.feature_column.numeric_column('Inverse_Y')]

print(feature_cols)

#Building DNN Regressor

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train_scaled,
                                                y=y_train,
                                                batch_size=20,
                                                num_epochs=100000,
                                                shuffle=True)

model = tf.estimator.DNNRegressor(hidden_units=[10,10,10],feature_columns=feature_cols)

model.train(input_fn=input_func,steps=30000)
#Loss for final step: 0.0008938401

#Testing the model

predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test_scaled,
                                                          num_epochs=1,
                                                          shuffle=False)

predictions_gen = model.predict(predict_input_func)

print(predictions_gen[0])

for p in enumerate(predictions_gen):
    print(p)