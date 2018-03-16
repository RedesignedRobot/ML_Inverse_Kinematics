import math as m

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split as tts

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

data = {'Inverse X':x_vals,
        'Inverse Y':y_vals,
        'Forward X':fx,
        'Forward Y':fy,
        'Theta 1 °':t1_vals,
        'Theta 2 °':t2_vals}

delta = {'Delta X':delta_X,
        'Delta Y':delta_Y,}

dataframe = pd.DataFrame(data=data)
delta_dataframe=pd.DataFrame(data=delta)

print(dataframe)
print(delta_dataframe)