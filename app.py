# import required packages
import base64
from io import BytesIO
from flask import Flask, render_template, request
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.pylab import rcParams


# from mpl_finance import candlestick_ohlc

from nsepy import get_history

rcParams['figure.figsize'] = 10, 10
plt.rc('font', size=14)

scaler = MinMaxScaler(feature_range=(0, 1))

plt.rc('font', size=14)
app = Flask(__name__)

"""The default page will route to the form.html page where user can input
necessary variables for machine learning"""


def obtain_data(ticker, start, end):
    # Enter the start and end dates using the method date(yyyy,m,dd)
    stock = get_history(symbol=ticker, start=start, end=end, index=True)
    print(stock)
    df = stock.copy()
    df = df.reset_index()

    df.index = df.Date
    return df


@app.route('/')
def form():
    return render_template('form.html')


@app.route('/future', methods=['POST'])
def future():
    if request.method == "POST":
        command = request.form.get('command')

        if command == 'S & R Levels':
            current_userinput = request.form.get("stock", None)
            print(current_userinput)

            df = obtain_data(current_userinput, date(2021,6,8), date(2021,10,8))
            print(df)
            df['Date'] = pd.to_datetime(df.index)
            df['Date'] = df['Date'].apply(mpl_dates.date2num)
            def isSupport(df, i):
                support = df['Low'][i] < df['Low'][i - 1] < df['Low'][i - 2] and df['Low'][i] < df['Low'][i + 1] < \
                          df['Low'][i + 2]
                return support
            def isResistance(df, i):
                resistance = df['High'][i] > df['High'][i - 1] > df['High'][i - 2] and df['High'][i] > \
                             df['High'][i + 1]> df['High'][i + 2]
                return resistance
            # df = df.loc[:,['Date','Open', 'High', 'Low', 'Close']]
            levels = []
            for i in range(2, df.shape[0] - 2):
                if isSupport(df, i):
                    levels.append((i, df['Low'][i]))
                elif isResistance(df, i):
                    levels.append((i, df['High'][i]))
            def plot_all():
                figr, ax = plt.subplots()
                candlestick_ohlc(ax, df.values, width=1, colorup='green', colordown='red', alpha=1)
                date_format = mpl_dates.DateFormatter('%d %b %Y')
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
            STOCK = BytesIO()
            plt.savefig(STOCK, format="png")
            STOCK.seek(0)
            original_url = base64.b64encode(STOCK.getvalue()).decode('utf8')
            return render_template("plot.html", original_url=original_url)

        else:
            current_userinput = request.form.get("stock", None)

            start_date = request.form.get("start", None)
            end_date = request.form.get("end", None)
            start_date1 = start_date.split('-')
            end_date1 = end_date.split('-')
            print(start_date,end_date)
            data = obtain_data(current_userinput, date(2021,6,8), date(2021,10,8))

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
            fig, ax = plt.subplots(figsize=(10, 10))

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
            plt.title('Prices')

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
            original_url = base64.b64encode(STOCK.getvalue()).decode('utf8')
            return render_template("plot.html", original_url=original_url)



