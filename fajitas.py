import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from model import create_model
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


df = pd.read_csv('BTC-USD.csv')
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(data.values) * 0.9)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len]
x_train, y_train = [], []

for i in range(60, training_data_len):
  x_train.append(training_data[i-60:i, 0])
  y_train.append(training_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

model = create_model(60)
model.load_weights('model_weights.keras')
model.compile(optimizer='adam',loss='mean_squared_error')



testing_data = scaled_data[0:,:]
x_test = []
y_test=dataset[60:,:]
for i in range(60,len(testing_data)):
  x_test.append(testing_data[i-60:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
rmse=np.sqrt(np.mean((predictions - y_test)**2))

train=data[:training_data_len]
valid=data[60:]
valid['Predictions']=predictions


closing = valid['Close'].values
pred = valid['Predictions'].values

total_error = 0
for i in range(len(closing)):
  difference = closing[i] - pred[i]
  total_error += (difference / closing[i]) * 100
avg_percent_error = total_error / len(pred)


plt.figure(figsize=(16,8))
plt.title('Fajita Target vs BTC')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closing Price in USD($)',fontsize=18)
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()