import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import copy

# ---------Preparing Data------------
df = pd.read_csv('Indian_IPO_Market_Data.csv')

df['Listing_Gains_Profit'] = np.where(df['Listing_Gains_Percent'] > 0, 1, 0)
df['Listing_Gains_Profit'].value_counts(normalize=True)

df = df.drop(['Date', 'IPOName','Listing_Gains_Percent'], axis=1)
 
# Neural Network is susceptible to outliers, so we need to clip them
for col in ['Issue_Size', 'Subscription_QIB', 'Subscription_HNI', 'Subscription_RII', 'Subscription_Total']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1
    lower = (q1 - 1.5 * iqr)
    upper = (q3 + 1.5 *iqr)

    df[col] = df[col].clip(lower, upper)

target = 'Listing_Gains_Profit'
predictors = df.drop(columns=[target]).columns

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[predictors])

X_scaled_df = pd.DataFrame(X_scaled, columns=predictors)

X = X_scaled_df.values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# ----------Neural Network------------
tf.random.set_seed(42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Training Model
model.fit(X_train, y_train, epochs=250)

# ----------Comparing neuronal network with Regression and Random Forest Classifier------------

# 1- Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Logistic Regression: {accuracy_score(y_test, y_pred):.4f}")

# 2- Randoml Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Random Forest: {accuracy_score(y_test, y_pred):.4f}")