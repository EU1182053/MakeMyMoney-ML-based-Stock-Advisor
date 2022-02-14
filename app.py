# import required packages
import base64
import datetime
import math
from io import BytesIO

from dateutil.relativedelta import relativedelta
from flask import Flask, render_template, request
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import yfinance as yf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

rcParams['figure.figsize'] = 8, 6
plt.rc('font', size=14)

scaler = MinMaxScaler(feature_range=(0, 1))

plt.rc('font', size=14)
app = Flask(__name__)

"""The default page will route to the form.html page where user can input
necessary variables for machine learning"""

trend = ""


def obtain_data(ticker):
    # Enter the start and end dates using the method date(yyyy,m,dd)
    # stock = get_history(symbol=ticker, start=start, end=end, index=True)

    stock = yf.download(ticker, '2021-02-13', datetime.date.today() - relativedelta(days=2))
    df = stock.copy()
    df = df.reset_index()

    df.index = df.Date
    return df


def detect_trend(data):
    global trend
    close = data['Close']
    check = close[0]
    up, down = 0, 0
    for i in close:
        if check > i:
            down += 1
        else:
            up += 1
    if up > down:
        trend = "Up"
        return trend
    else:
        trend = "Down"
        return trend


def trade_1(levels, last_closing):

    global trend
    if trend == "Up":
        levels_price = []
        for i in range(len(levels)):
            levels_price.append(levels[i][1])
        levels_price = sorted(levels_price)

        if last_closing > max(levels_price):
            if last_closing > (levels_price[-1] * 0.005):
                stop_loss = levels_price[-1]
                target_values = []
                msg = "Targets are Open." + "Buy"
            else:
                stop_loss = 0
                target_values = []
                msg = "Avoid Trade."
        else:

            stop_loss, target_values, msg = 0, 0, ""
            for i in range(len(levels_price) - 1):
                if levels_price[i] <= last_closing <= levels_price[i + 1]:
                    stop_loss = levels_price[i]
                    target_values = levels_price[i + 1:i+3]
                    msg = "Buy"
                    break
        #
        return "stop_loss", stop_loss, "target_values", target_values, "msg", msg
    else:
        levels_price = []
        for i in range(len(levels)):
            levels_price.append(levels[i][1])
        levels_price = sorted(levels_price)
        if last_closing > max(levels_price):
            if last_closing > (levels_price[-1] * 0.005):
                stop_loss = levels_price[-1]
                target_values = []
                msg = "Targets are Open." + "Sell"
            else:
                stop_loss = 0
                target_values = []
                msg = "Avoid Trade."
        else:

            stop_loss, target_values, msg = 0, 0, ""
            for i in range(len(levels_price) - 1):
                if levels_price[i] <= last_closing <= levels_price[i + 1]:
                    stop_loss = levels_price[i]
                    target_values = levels_price[i + 1:i+3]
                    msg = "Sell"
                    break
        #
        return "stop_loss", stop_loss, "target_values",target_values, "msg", msg


def trade_2(levels, last_closing):
    global trend
    levels_price = []
    if trend == "Up":
        for i in range(len(levels)):
            levels_price.append(levels[i][1])
        levels_price = sorted(levels_price)
        if last_closing > max(levels_price):
            if last_closing > (levels_price[-1] * 0.005):
                stop_loss = levels_price[-1]
                target_values = []
                msg = "Targets are Open." + "Buy"
            else:
                stop_loss = 0
                target_values = []
                msg = "Avoid Trade."
        else:

            stop_loss, target_values, msg = 0, 0, ""
            for i in range(len(levels_price) - 1):
                if levels_price[i] <= last_closing <= levels_price[i + 1]:
                    stop_loss = levels_price[i + 1]
                    target_values = levels_price[:i - 1]
                    msg = "Buy"
                    break
        #
        return "stop_loss", stop_loss, "target_values", target_values, "msg", msg
    else:
        for i in range(len(levels)):
            levels_price.append(levels[i][1])
        levels_price = sorted(levels_price)
        if last_closing > max(levels_price):
            if last_closing > (levels_price[-1] * 0.005):
                stop_loss = levels_price[-1]
                target_values = []
                msg = "Targets are Open." + "Sell"
            else:
                stop_loss = 0
                target_values = []
                msg = "Avoid Trade."
        else:

            stop_loss, target_values, msg = 0, 0, ""
            for i in range(len(levels_price) - 1):
                if levels_price[i] <= last_closing <= levels_price[i + 1]:
                    stop_loss = levels_price[i + 1]
                    target_values = levels_price[:i - 1]
                    msg = "Sell"
                    break
        #
        return "stop_loss", stop_loss, "target_values", target_values, "msg", msg


@app.route('/')
def form():
    return render_template('form.html')


@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/news')
def show_news():
    return render_template('news.html')


@app.route('/future', methods=['POST'])
def future():
    if request.method == "POST":

        current_userinput = request.form.get("stock", None)
        df = obtain_data(current_userinput)
        print(detect_trend(df))

        df['Date'] = df['Date'].apply(mpl_dates.date2num)

        df["Date"] = pd.to_datetime(df.Date)
        df.index = df['Date']

        train_value = math.floor(len(df) * 0.9)
        remain_value = math.floor(len(df) - train_value)


        # close data

        close_data = df.sort_index(ascending=True, axis=0)
        new_close_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', "Close"])

        for i in range(0, len(close_data)):
            new_close_dataset["Date"][i] = close_data['Date'][i]
            new_close_dataset["Close"][i] = close_data["Close"][i]

        new_close_dataset.index = new_close_dataset.Date
        new_close_dataset.drop("Date", axis=1, inplace=True)

        final_close_dataset = new_close_dataset.values

        train_close_data = final_close_dataset[0:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close_data = scaler.fit_transform(final_close_dataset)

        x_train_close_data, y_train_close_data = [], []

        for i in range(60, len(train_close_data)):
            x_train_close_data.append(scaled_close_data[i - 60:i, 0])
            y_train_close_data.append(scaled_close_data[i, 0])

        x_train_close_data, y_train_close_data = np.array(x_train_close_data), np.array(y_train_close_data)

        x_train_close_data = np.reshape(x_train_close_data,
                                        (x_train_close_data.shape[0], x_train_close_data.shape[1], 1))


        # close
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_close_data.shape[1], 1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(1))

        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(x_train_close_data, y_train_close_data, epochs=1, batch_size=1, verbose=2)

        inputs_close_data = new_close_dataset[len(new_close_dataset) - remain_value - 60:].values
        inputs_close_data = inputs_close_data.reshape(-1, 1)
        inputs_close_data = scaler.transform(inputs_close_data)


        # close
        X_close_test = []
        for i in range(60, inputs_close_data.shape[0]):
            X_close_test.append(inputs_close_data[i - 60:i, 0])
        X_close_test = np.array(X_close_test)

        X_close_test = np.reshape(X_close_test, (X_close_test.shape[0], X_close_test.shape[1], 1))
        prediction_closing = lstm_model.predict(X_close_test)
        prediction_closing = scaler.inverse_transform(prediction_closing)

        valid_close_data = pd.DataFrame(index=range(0, len(prediction_closing)), columns=["Date", "Predictions"])

        # for i in range(0,len(prediction_opening)):
        valid_close_data["Predictions"] = prediction_closing

        # valid_open_data["Predictions"].dtypes

        plt.plot(valid_close_data[["Predictions"]])

        df = obtain_data(current_userinput)
        df['Date'] = pd.to_datetime(df.index)
        df['Date'] = df['Date'].apply(mpl_dates.date2num)
        last_closing = df['Close'].iloc[-1]

        def isSupport(df, i):
            support = df['Low'][i] < df['Low'][i - 1] < df['Low'][i - 2] and df['Low'][i] < df['Low'][i + 1] < \
                      df['Low'][i + 2]
            return support

        def isResistance(df, i):
            resistance = df['High'][i] > df['High'][i - 1] > df['High'][i - 2] and df['High'][i] > \
                         df['High'][i + 1] > df['High'][i + 2]
            return resistance

        # df = df.loc[:,['Date','Open', 'High', 'Low', 'Close']]
        levels = []
        levels = []
        for i in range(2, df.shape[0] - 2):
            if isSupport(df, i):
                levels.append(df['Low'][i])
                levels.append((i, df['Low'][i]))
            elif isResistance(df, i):
                levels.append(df['High'][i])
                levels.append((i, df['High'][i]))

        def plot_all():
            figr, ax = plt.subplots()
            plt.title(f'{current_userinput} Prices')
            candlestick_ohlc(ax, df.values, width=1, colorup='green', colordown='red', alpha=1)
            date_format = mpl_dates.DateFormatter('%d %b %Y')
            ax.grid(True)
            ax.xaxis.set_major_formatter(date_format)
            figr.autofmt_xdate()
            figr.tight_layout()

            for level in levels:
                plt.hlines(level[1], xmin=df['Date'][level[0]], xmax=max(df['Date']), colors='blue')

        s = np.mean(df['High'] - df['Low'])

        def isFarFromLevel(l):
            return np.sum([abs(l - x) < s for x in levels]) == 0

        levels = []
        for i in range(2, df.shape[0] - 2):
            if isSupport(df, i):
                l = df['Low'][i]

                if isFarFromLevel(l):
                    levels.append((i, l))
            elif isResistance(df, i):
                l = df['High'][i]

                if isFarFromLevel(l):
                    levels.append((i, l))
        plot_all()
        # show the plot
        msg = trade_1(levels, last_closing)
        # print(trade_2(levels, last_closing))
        STOCK = BytesIO()
        plt.savefig(STOCK, format="png")
        STOCK.seek(0)
        sr_level_url = base64.b64encode(STOCK.getvalue()).decode('utf8')

        current_userinput = request.form.get("stock", None)

        df = obtain_data(current_userinput)

        df['Date'] = pd.to_datetime(df.index)

        df['Date'] = df['Date'].apply(mpl_dates.date2num)

        df["Date"] = pd.to_datetime(df.Date)

        df.index = df['Date']

        train_value = math.floor(len(df) * 0.9)

        remain_value = math.floor(len(df) - train_value)
        # close data

        close_data = df.sort_index(ascending=True, axis=0)
        new_close_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', "Close"])

        base = datetime.date.today() - relativedelta(days=2)
        for x in range(0, remain_value):
            valid_close_data['Date'][x] = (base + datetime.timedelta(days=x))
        valid_close_data.index = valid_close_data.Date

        # creating Subplots

        fig, ax = plt.subplots(figsize=(8, 6))

        # allow grid

        ax.grid(True)

        # Setting labels

        ax.set_ylabel('Price')

        # setting title
        plt.title(f'{current_userinput} Prices')
        # Setting labels

        ax.set_xlabel('Date')

        ax.set_ylabel('Price')

        # setting title
        plt.title(f'{current_userinput} Prices')
        # Formatting Date

        date_format = mpdates.DateFormatter('%d-%m-%Y')

        ax.xaxis.set_major_formatter(date_format)

        fig.autofmt_xdate()

        # Formatting Date

        # show the plot

        plt.plot(valid_close_data[["Predictions"]])  # prediction-blue

        STOCK = BytesIO()

        plt.savefig(STOCK, format="png")

        STOCK.seek(0)

        line_graph_url = base64.b64encode(STOCK.getvalue()).decode('utf8')

        return render_template("plot.html", line_graph_url=line_graph_url,sr_level_url=sr_level_url, msg=msg)


@app.route('/chart', methods=['POST'])
def show_chart():
    if request.method == "POST":
        current_userinput = request.form.get("stock", None)
        data = obtain_data(current_userinput)
        print(current_userinput)
        # Calling DataFrame constructor
        df = pd.DataFrame({
            'Date': [i for i in data['Date']],
            'Open': [i for i in data['Open']],
            'High': [i for i in data['High']],
            'Low': [i for i in data['Low']],
            'Close': [i for i in data['Close']],

        })

        # convert into datetime object
        df['Date'] = pd.to_datetime(df['Date'])

        # apply map function
        df['Date'] = df['Date'].map(mpdates.date2num)

        # creating Subplots
        fig, ax = plt.subplots(figsize=(8, 6))

        # plotting the data
        candlestick_ohlc(ax, df.values, width=1,
                         colorup='green', colordown='red',
                         alpha=1)

        # allow grid
        ax.grid(True)

        # Setting labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # setting title
        plt.title(f'{current_userinput} Prices')

        # Formatting Date
        date_format = mpdates.DateFormatter('%d-%m-%Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        fig.tight_layout()

        # show the plot

        # show the plot

        STOCK = BytesIO()
        plt.savefig(STOCK, format="png")
        STOCK.seek(0)
        raw_candle_url = base64.b64encode(STOCK.getvalue()).decode('utf8')
        data = obtain_data(current_userinput)

        df = pd.DataFrame({
            'Date': [i for i in data['Date']],
            'Close': [i for i in data['Close']],
        })
        print(df)
        df.index = df.Date
        # convert into datetime object
        df['Date'] = pd.to_datetime(df['Date'])

        # apply map function
        df['Date'] = df['Date'].map(mpdates.date2num)
        # creating Subplots

        fig, ax = plt.subplots(figsize=(8, 6))

        # allow grid

        ax.grid(True)

        # Setting labels

        ax.set_ylabel('Price')

        # setting title
        plt.title(f'{current_userinput} Prices')
        # Setting labels

        ax.set_xlabel('Date')

        ax.set_ylabel('Price')

        # setting title
        plt.title(f'{current_userinput} Prices')
        # Formatting Date

        date_format = mpdates.DateFormatter('%d-%m-%Y')

        ax.xaxis.set_major_formatter(date_format)

        fig.autofmt_xdate()

        # Formatting Date

        # show the plot

        plt.plot(data["Close"])  # prediction-blue

        STOCK = BytesIO()

        plt.savefig(STOCK, format="png")

        STOCK.seek(0)

        line_graph_url = base64.b64encode(STOCK.getvalue()).decode('utf8')

        return render_template("chart.html", line_graph_url=line_graph_url,
                               raw_candle_url=raw_candle_url)
