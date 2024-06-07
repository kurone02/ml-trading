from typing import Callable
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import yfinance as yf 
import numpy as np
from multiprocessing import Process

def load_financial_data(ticker, start_date, end_date) -> pd.DataFrame:
    """
    Function to load the stock OHLC price in the list of tickers from `start_date` to `end_date
    """
    # Define the output file name
    output_file = './data/data_'+str(ticker)+'_'+str(start_date)+'_'+str(end_date)+'.pkl'
    try:
        # Try to read the data from the output file
        df = pd.read_pickle(output_file)
        # If the data is already downloaded, print a success message
        print('File data found...reading '+ticker+' data') 
    except FileNotFoundError:
        # If the file is not found, download the data from Yahoo Finance
        print('File not found...downloading the '+ticker+' data')  
        # Download the data
        df = yf.download(ticker, start=start_date, end=end_date)
        # Save the data to the output file
        df.to_pickle(output_file) 
    # Return the requested data
    return df 


def plot_ts(data, title, label_ts, x_label, y_label):
    plt.figure(figsize=(14, 7))

    plt.plot(data, label=label_ts)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    
def plot_multi_ts(data, title, label_ts, x_label, y_label):
    plt.figure(figsize=(14, 7))

    for idx, dat in enumerate(data):
        plt.plot(dat, label=label_ts[idx])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def sharpe_ratio(rate):
    r_f = 0.005 / 252  # 0.5% (US Treasury)
    r_m = rate.mean()
    sigma = rate.std()
    return (r_m - r_f) / sigma

def rate_of_return(portfolio: pd.Series) -> pd.Series:
    return portfolio / portfolio.iloc[0]

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def save_txt(file_path: str, data: str) -> None:
    with open(file_path, "w") as f:
        f.write(data)


def run_parallel(fns: list[Callable], fns_arg: list[tuple]):
    proc = []
    for fn, args in zip(fns, fns_arg):
        p = Process(target=fn, args=args)
        p.start()
        proc.append(p)

    for p in proc:
        p.join()

