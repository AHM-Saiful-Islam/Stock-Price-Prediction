# StockPricePredictor class with LSTM Model documentation.
# usecase 
'''

# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class StockPricePredictor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.closing_prices = None
        self.X_train = None
        self.y_train = None

    def fetch_data(self):
        """Fetch historical stock data for the given ticker and date range."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.closing_prices = data['Close'].values.reshape(-1, 1)
        return data

    def preprocess_data(self):
        """Preprocess the data by scaling and preparing training data."""
        scaled_data = self.scaler.fit_transform(self.closing_prices)
        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        
        self.X_train = np.array(X_train).reshape(-1, 60, 1)
        self.y_train = np.array(y_train)
        return self.X_train, self.y_train

    def build_model(self):
        """Build the LSTM model."""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, epochs=50, batch_size=32):
        """Train the LSTM model."""
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def predict_future(self, days=60):
        """Make future predictions."""
        last_60_days = self.scaler.transform(self.closing_prices[-60:])
        X_test = last_60_days.reshape(1, -1, 1)

        predicted_prices = []
        for _ in range(days):
            pred_price = self.model.predict(X_test)
            predicted_prices.append(pred_price[0, 0])
            new_input = np.append(X_test.flatten()[1:], pred_price[0, 0]).reshape(1, -1, 1)
            X_test = new_input

        self.predicted_prices = self.scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        return self.predicted_prices

    def evaluate_model(self, real_data):
        """Evaluate the model against real stock data."""
        mse = mean_squared_error(real_data, self.predicted_prices[:len(real_data)])
        r2 = r2_score(real_data, self.predicted_prices[:len(real_data)])
        return mse, r2

    def plot_predictions(self, real_data, predicted_prices):
        """Plot both predicted and real data."""
        plt.figure(figsize=(14, 7))
        plt.plot(pd.date_range(start='2024-09-01', periods=len(real_data), freq='B'), real_data, label='Real Prices')
        plt.plot(pd.date_range(start='2024-09-01', periods=len(predicted_prices), freq='B'), predicted_prices, label='Predicted Prices')
        plt.title('Comparison of Predicted and Real Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.legend()
        plt.show()
'''

## usecase
'''
# Example usage for Meta:
predictor = StockPricePredictor(ticker='META', start_date='2019-08-01', end_date='2024-08-31') # chang ticker for dif company
predictor.fetch_data()
predictor.preprocess_data()
predictor.build_model()
predictor.train_model()
predicted_prices = predictor.predict_future()

# plot both real_data, predicted_prices
real_data = yf.download('META', start='2024-09-01', end='2024-10-31')['Close'].values  # chang ticker for dif company
predictor.plot_predictions(real_data, predicted_prices)

# evaluation
mse, r2 = predictor.evaluate_model(real_data)

print(f'Mean Squared Error: {mse}') # 0 worst, 1 best. big value is worst.
print(f'R-squared: {r2}')           # tends to 100 good. minus value is worst
'''