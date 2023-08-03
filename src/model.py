import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt

scaler = MinMaxScaler(feature_range=(0,1))

class Model:
    def __init__(self, x_shape):
        self.x_shape = x_shape
        self.model = load_model('./src/model_weights.keras')
        
    def prediction(self, x_data):
        size = len(x_data)
        x_data = x_data.reshape(-1, 1)  # Ensure 2D shape for scaling
        scaled_data = scaler.fit_transform(x_data)
        scaled_data = scaled_data[size - self.x_shape:]
        scaled_data = scaled_data.reshape((1, self.x_shape, 1)) # Reshape to 3D for LSTM
        val = self.model.predict(scaled_data)
        return scaler.inverse_transform(val)[0][0]
    
    def extrapolate(self, data, length):
        current_x_data = data
        predictions = []
        for i in range(length):
            prediction = self.prediction(current_x_data)
            predictions.append(prediction)
            current_x_data = np.append(current_x_data, prediction) 
        return predictions
    
model = Model(60)
df = pd.read_csv('./stocks/AMZN.csv')
data = df.filter(['Close'])
dataset = data.values
predictions = model.extrapolate(dataset, 10)
data = dataset[len(dataset)-100:]
full_data = np.concatenate((data, np.array(predictions).reshape(-1, 1)), axis=0) # Concatenate the data and predictions

print(full_data)

plt.figure(figsize=(16,8))
plt.title('Fajita Target vs AMZN')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closing Price in USD($)',fontsize=18)
plt.plot(full_data)
plt.show()
