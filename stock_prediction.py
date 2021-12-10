import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras import regularizers
from plotly import graph_objs as go

#CSV file

filename= "/content/drive/MyDrive/CSV file/ASIANPAINT.csv"
df= pd.read_csv(filename)
print(df.info())

#drop the unnecessary data
df['Date']= pd.to_datetime(df['Date'])
df.set_axis(df['Date'], inplace= False)
df= df.dropna()
df.drop(columns=['Adj Close'], inplace=False)

cl_data = df['Close'].values
cl_data = cl_data.reshape((len(cl_data), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(cl_data)

print('Min: %f, Max: %f' %(scaler.data_min_, scaler.data_max_))
close_data = scaler.transform(cl_data)
print(close_data)

#split train and test data
split_percent = 0.50
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

data_train = df['Date'][:split]
data_test = df['Date'][split:]

print(len(close_train))
print(len(close_test))

close_train = close_train.reshape((len(close_train), 1))
close_test = close_test.reshape((len(close_test), 1))

look_back = 5

train_generator = TimeseriesGenerator(close_train,close_train, length= look_back, batch_size=3)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=2)

#ML Model
model = Sequential()
model.add(
        LSTM(80, kernel_regularizer=regularizers.l2(0.001), return_sequences=True, activation='relu',
        input_shape=(look_back,1)))
model.add(
        LSTM(40, kernel_regularizer=regularizers.l2(0.001), return_sequences=True, activation='relu'))
model.add(
        LSTM(4, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

num_epochs= 30
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test =close_test.reshape((-1))
prediction = prediction.reshape((-1))

def predict(num_predict, model):
  prediction_list = close_data[-look_back:]

  for _ in range(num_prediction):
    x=prediction_list[-look_back:]
    x= x.reshape((1, look_back, 1))
    out= model.predict(x)[0][0]
    prediction_list= np.append(prediction_list, out)
    prediction_list = prediction_list[look_back - 1:]

    return prediction_list

def predict_dates(num_prediction):
  last_date = df['Date'].values[-1]
  prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
  return prediction_dates

num_prediction = 365
forecast= predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

#Create a graph
trace1 = go.Scatter(
    x = data_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = data_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = data_test,
    y = close_test,
    mode = 'lines',
    name = 'Ground Truth'
)
trace4 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode = 'lines',
    name = 'Forecast'
)
layout = go.Layout(
    title = "Predicting for Asian Paints of 1 Year",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "close"}
)
fig = go.Figure(data=[trace1,trace2,trace3,trace4], layout=layout)
fig.show()
