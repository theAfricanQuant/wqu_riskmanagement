# Imports leveraging library functions
import warnings
import copy
import pandas as pd
import numpy as np
import scipy.stats
from pandas_datareader import data as pdr  # The pandas Data Module used for fetching data from a Data Source
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')  # Set the plotting style

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fix_yahoo_finance import pdr_override  # For overriding Pandas DataFrame Reader not connecting to YF


def yahoo_finance_bridge():
    """
    This function fixes problems w.r.t. fetching data from Yahoo Finance
    :return: None
    """
    pdr_override()


def main():
    """
    Main function for executing program logic
    :return: None
    """
    # Fix Yahoo Finance
    yahoo_finance_bridge()

    start = datetime.datetime(2013, 1, 1)  # Start time
    end = datetime.datetime(2018, 1, 1)  # End Time

    data = pdr.get_data_yahoo(['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM',
                               'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG',
                               'TRV', 'UTX', 'UNH', 'VZ', 'V', 'WMT'], start=start, end=end, auto_adjust=True)

    df = data['Close']  # Set the dataframes data as closed

    # resample daily close data to monthly data
    data = df.resample('M').mean()

    # calculate Monthly Returns#calcula
    returns = np.log(data / data.shift(1))
    returns_neg = copy.deepcopy(returns)
    returns_pos = copy.deepcopy(returns)

    # replace all Neg values with NaN, for Perf Calculations, this is for negative month
    returns_neg[returns_neg < 0] = np.nan
    # replace all positive values with NaN, for Perf Calculations, this is for positive
    returns_pos[returns_pos > 0] = np.nan

    # Mean for negative
    print('===== Average Monthly Return on Negative Month =====')
    print(returns_neg.describe().loc['mean'])

    # Mean for positive months
    print('===== Average Monthly Return on Positive Month =====')
    print(returns_pos.describe().loc['mean'])

    print('===== Porbability of a Positive Month =====')
    pos_month_dict = {'MMM': 0, 'AXP': 0, 'AAPL': 0, 'BA': 0, 'CAT': 0, 'CVX': 0, 'CSCO': 0, 'KO': 0, 'DIS': 0, 'XOM': 0,
                               'GE': 0, 'GS': 0, 'HD': 0, 'IBM': 0, 'INTC': 0, 'JNJ': 0, 'JPM': 0, 'MCD': 0, 'MRK': 0, 'MSFT': 0, 'NKE': 0, 'PFE': 0, 'PG': 0,
                               'TRV': 0, 'UTX': 0, 'UNH': 0, 'VZ': 0, 'V': 0, 'WMT': 0}
    ret_pos = returns[returns > 0]
    for index, col in enumerate(list(returns.columns)):
        pos_month_dict[col] = float(len(ret_pos[col].dropna())) / float(len(ret_pos[col]))
    print(pos_month_dict)

    returns = (data - data.shift(1)) / data.shift(1)
    returns = returns.dropna()
    # Calculate individual mean returns and covariance between the stocks
    meanDailyReturns = returns.mean()
    covMatrix = returns.cov()
    plot1 = (data / data.iloc[0] * 100).plot(figsize=(15, 6))
    plot1.plot()
    plt.show()
    plt.close()

    # we are only going to choose five random companies from the list.#we are o
    companies = list(returns.sample(5, axis=1, random_state=4).columns)
    print(companies)
    mean = returns.mean()
    sigma = returns.std()
    tdf, tmean, tsigma = scipy.stats.t.fit(returns.as_matrix())
    ret_hist = returns.hist(bins=40, normed=True, histtype='stepfilled', alpha=0.5)
    plt.show()

    print(returns.quantile(0.05))  # Print the quantiles

    days = 1800  # time horizon
    dt = 1 / float(days)
    sigma = 0.04  # volatility
    mu = 0.05  # drift (average growth rate)

    def random_walk(startprice):
        price = np.zeros(days)
        shock = np.zeros(days)
        price[0] = startprice
        for i in range(1, days):
            shock[i] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
            price[i] = max(0, price[i - 1] + shock[i] * price[i - 1])
        return price

    for run in range(30):
        plt.plot(random_walk(10.0))
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

    runs = 10000
    simulations = np.zeros(runs)
    for run in range(runs):
        simulations[run] = random_walk(10.0)[days - 1]
    q = np.percentile(simulations, 1)
    plt.hist(simulations, normed=True, bins=30, histtype='stepfilled', alpha=0.5)
    plt.figtext(0.6, 0.8, u"Start price: 10 dollars")
    plt.figtext(0.6, 0.7, u"Mean final price: {:.3} dollars".format(simulations.mean()))
    plt.figtext(0.6, 0.6, u"VaR(0.99): {:.3} dollars".format(10 - q))
    plt.figtext(0.15, 0.6, u"q(0.99): {:.3} dollars".format(q))
    plt.axvline(x=q, linewidth=4, color='r')
    plt.title(u"Final price distribution after {} days".format(days), weight='bold')
    plt.show()

    return 0


if __name__ == '__main__':
    main()
