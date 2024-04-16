import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import datetime

st.title('Stock trend prediction')
ticker_symbol = st.text_input('Enter Stock ticker','BTC-USD')
start = st.date_input("Start Date", datetime.date(2021, 1, 1))
end = st.date_input("End Date", datetime.date(2024, 1, 1))
tS = float(st.text_input('Enter Sell Threshold',"0.1"))
tB = float(st.text_input('Enter Buy Threshold',"0.9"))

data = yf.download(ticker_symbol, start=start, end=end)
data = data.reset_index()

st.subheader("Data from {start} to {end}".format(start=start, end=end))
st.write(data)

st.subheader("Closing Price VS Time Chart")

ema_periods = [5, 25, 50, 99, 200]
for period in ema_periods:
    column_name = f'EMA_{period}'
    data[column_name] = data['Adj Close'].ewm(span=period, adjust=False).mean()

for i in ['5', '25', '50', '99', '200']:
    data[f'distance_price_ema{i}'] = ((data['Adj Close'] - data[f'EMA_{i}']) / data['Adj Close'])*100

fig = plt.figure(figsize=(12, 6))

#Plot Bitcoin price
plt.plot(data.index, data['Adj Close'], label='Bitcoin Price', color='blue')

#Plot EMAs
ema_periods = [5, 25, 50, 99, 200]
colors = ['red', 'green', 'orange', 'purple', 'brown']
for i, period in enumerate(ema_periods):
    column_name = f'EMA_{period}'
    plt.plot(data.index, data[column_name], label=f'EMA {period}', color=colors[i])

plt.title('Bitcoin Price with Exponential Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(fig)

#Calculate daily price movement as a percentage
data['Price_Movement'] = ((data['Close'] - data['Open']) / data['Open']) * 100

#Normalize the price movement between 0 and 1
data['Normalized_Price_Movement'] = (data['Close'] - data['Low'].min()) / (data['High'].max() - data['Low'].min())
#Calculate Relative Strength Index (RSI)
def calculate_rsi(data, period=14):
    close_prices = data['Adj Close']
    price_diff = close_prices.diff(1)

    gain = price_diff.where(price_diff > 0, 0)
    loss = -price_diff.where(price_diff < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi

#Adding RSI as a new column
data['RSI'] = calculate_rsi(data)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Adj Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Adj Close'].ewm(span=long_window, adjust=False).mean()

    macd = (short_ema - long_ema)
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    return macd, signal

#Add MACD and Signal as new columns
data['MACD'], data['Signal'] = calculate_macd(data)
#Calculate Stochastic RSI (StochRSI)
def calculate_stochrsi(data, period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    rsi = calculate_rsi(data, period)

    stochrsi = (rsi - rsi.rolling(window=stoch_period).min()) / (rsi.rolling(window=stoch_period).max() - rsi.rolling(window=stoch_period).min())

    stoch_k = stochrsi.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d).mean()

    return stoch_k, stoch_d

#Add StochRSI and StochRSI Signal as new columns
data['StochRSI_K'], data['StochRSI_D'] = calculate_stochrsi(data)
data.dropna(inplace=True)

def find_local_extremas(Date, Normalized_Price_Movement, Price, RSI, MACD, Signal,
                        distance_price_ema5, distance_price_ema25, distance_price_ema50,
                        distance_price_ema99, distance_price_ema200, window_size=3):
    n = len(Price)

    local_extremas = []
    #Find local maximum
    for i in range(window_size, n - window_size):
        if all(Normalized_Price_Movement[i - j] < Normalized_Price_Movement[i] for j in range(1, window_size + 1)) and \
                all(Normalized_Price_Movement[i] > Normalized_Price_Movement[i + k] for k in range(1, window_size + 1)):
            local_extremas.append((Date[i], Normalized_Price_Movement[i], Price[i], RSI[i], MACD[i], Signal[i],
                                   distance_price_ema5[i], distance_price_ema25[i], distance_price_ema50[i],
                                   distance_price_ema99[i], distance_price_ema200[i], 1))

    #Find local minimum
    for i in range(window_size, n - window_size):
        if all(Normalized_Price_Movement[i - j] > Normalized_Price_Movement[i] for j in range(1, window_size + 1)) and \
                all(Normalized_Price_Movement[i] < Normalized_Price_Movement[i + k] for k in range(1, window_size + 1)):
            local_extremas.append((Date[i], Normalized_Price_Movement[i], Price[i], RSI[i], MACD[i], Signal[i],
                                   distance_price_ema5[i], distance_price_ema25[i], distance_price_ema50[i],
                                   distance_price_ema99[i], distance_price_ema200[i], 0))

    return local_extremas
#Converting columns into list
Date = data.index.tolist()
Price = data['Adj Close'].tolist()
RSI = data['RSI'].tolist()
MACD = data['MACD'].tolist()
Signal = data['Signal'].tolist()
Normalized_Price_Movement = data['Normalized_Price_Movement'].tolist()
distance_price_ema5 = data['distance_price_ema5'].tolist()
distance_price_ema25 = data['distance_price_ema25'].tolist()
distance_price_ema50 = data['distance_price_ema50'].tolist()
distance_price_ema99 = data['distance_price_ema99'].tolist()
distance_price_ema200 = data['distance_price_ema200'].tolist()

bitcoin_extremas = find_local_extremas(Date, Normalized_Price_Movement, Price, RSI, MACD, Signal,
                                       distance_price_ema5, distance_price_ema25, distance_price_ema50,
                                       distance_price_ema99, distance_price_ema200)

extremas_df = pd.DataFrame(bitcoin_extremas, columns=['Date', 'Normalized_Price_Movement', 'Price', 'RSI', 'MACD', 'Signal',
                                                      'Distance_Price_EMA5', 'Distance_Price_EMA25', 'Distance_Price_EMA50',
                                                      'Distance_Price_EMA99', 'Distance_Price_EMA200', 'Type'])

extremas_df.set_index('Date', inplace=True)


X = extremas_df[['Normalized_Price_Movement', 'Price', 'RSI', 'MACD', 'Signal', 'Distance_Price_EMA5',	'Distance_Price_EMA25',	'Distance_Price_EMA50',	'Distance_Price_EMA99', 'Distance_Price_EMA200']]
#Target
y = extremas_df['Type']
#Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = pickle.load(open('Butcher_Model.sav', 'rb'))
model.fit(X_train, y_train)

#Predicting probabilities for targets
y_proba = model.predict_proba(X_test)

#Setting thresholds for sell and buy predictions

thresholds = {'Sell': tS, 'Buy': tB}

#Create a dictionary to store results
results = {}

for label, threshold in thresholds.items():
    #Predict classes based on thresholds
    y_pred = (y_proba[:, 1] > threshold).astype(int)

    #Calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test.values, y_pred)

    #Store results
    results[label] = {'confusion_matrix': cm, 'classification_report': report}

#Printing results
st.subheader("Classification Report")
for label, info in results.items():
    st.write(f"\nResults for {label}:")
    # print(f"Confusion Matrix:\n{info['confusion_matrix']}")
    fig = plt.figure(figsize=(3,2))
    sns.set(font_scale=0.5)
    sns.heatmap(info['confusion_matrix'], annot=True, annot_kws={"size": 7}, fmt='g', cmap="Blues") # font size
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    print(f"\nClassification Report:\n{info['classification_report']}")
    st.write(f"\nClassification Report:\n{info['classification_report']}")
    st.pyplot(fig)

st.subheader("Prediction Chart")

y_pred_labels = []
for proba in y_proba:
    if proba[1] >= thresholds['Buy']:
        y_pred_labels.append('Buy')
    elif proba[1] <= thresholds['Sell']:
        y_pred_labels.append('Sell')
    else:
        y_pred_labels.append('Hold')  # Optional: for everything in-between

# Step 2: Prepare the Data for Plotting
# Assuming 'Price' is what you want on the y-axis and the index for the x-axis.
prices = X_test['Price']
indices = X_test.index

# Step 3: Plot the Graph
fig = plt.figure(figsize=(60, 15))
plt.plot(data.index, data['Adj Close'])

for i, label in enumerate(y_pred_labels):
    if label == 'Buy':
        plt.scatter(indices[i], prices.iloc[i], color='red', label='Sell' if i == 0 else "")
    elif label == 'Sell':
        plt.scatter(indices[i], prices.iloc[i], color='green', label='Buy' if i == 0 else "")

# Optional: Improve plot readability and aesthetics
plt.title('Prediction Results')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
