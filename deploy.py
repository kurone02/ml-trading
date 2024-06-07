#!pip install ib_insync
#!pip install bs4
from ib_insync import *
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import quantstats as qs
from ml import *
from utils import *
from markov import *
from models import *
from datetime import datetime

# Top IT companies
TICKERS = [
    "NVDA", "MSFT", "AAPL", "TSLA",
    "META", "PYPL", "GOOG", "COST",
    "INTC", "AMZN", "AMD", "BABA",
]

# # Crypto
# TICKERS = [
#     "ETH", "LINK", "BCH", "LTC",
# ]


models: list[MLStrategy] = [0 for _ in range(len(TICKERS))]
results: dict[str, pd.DataFrame] = {}

    
def get_stock_data(ib: IB, duration: str, freq: int, ticker: str) -> pd.DataFrame:
    """
    Function to get the stock data from the IB API
    """
    stock = Stock(ticker, 'SMART', 'USD', primaryExchange='NASDAQ')
    # stock = Crypto(ticker, 'Paxos', 'USD')
    bars = ib.reqHistoricalData(stock, 
                                endDateTime = '', 
                                durationStr = duration, 
                                barSizeSetting = f"{freq} min{'' if freq == 1 else 's'}", 
                                whatToShow = 'ADJUSTED_LAST', 
                                useRTH = True, 
                                formatDate = 2,
                            )
    dfBars = util.df(bars)
    return dfBars
    
def train_model(ib: IB, duration: str, freq: str, ticker: str) -> MLStrategy:
    """
    Training the model given the stock ticker
    """
    print(f"Training the model {ticker}...")
    # Load the stock data
    security_data = get_stock_data(ib, duration, freq, ticker)
    # Define the strategy
    markovian_strategy = MLStrategy(
        price=security_data["close"],
        valid_data=security_data["close"],
        test_data=security_data["close"],
        threshold_to_buy=8,
        threshold_to_sell=8,
        lookback=20,
        forward=3,
    )
    # Train the model
    markovian_strategy.train(
        epochs=50,
        lr=1e-2,
    )
    # Return the trained model
    models[TICKERS.index(ticker)] = markovian_strategy        
    print(f"Finish training the model {ticker}!")

def deploy_model(ib: IB, duration: str, freq: int, ticker: str) -> None:
    """
    Deploy the model given the stock ticker
    """
    print(f"Start deploying the model {ticker}...")
    # Get the model
    model = models[TICKERS.index(ticker)]
    # Deploy the model
    results[ticker] = model.deploy(
        ib=ib, 
        ticker=ticker,
        initial_cash=2e3,
        num_share_per_trade=1,
        limit_borrow=0,
        limit_num_shorts=0,
        freq=freq,
        is_crypto=False,
    )
    print(f"Finish deploying the model {ticker}!")


def train_all(ib: IB, duration: str, freq: int) -> None:
    """
    Train all the models
    """
    fns = [train_model for _ in range(len(TICKERS))]
    fns_arg = [(ib, duration, freq, ticker) for ticker in TICKERS]
    print("=" * 15, "Training models", "=" * 15)
    run_parallel(fns, fns_arg)
    print("=" * 15, " Finish models ", "=" * 15)

    
def deploy_all(ib: IB, duration: str, freq: int) -> None:
    """
    Deploy all the models
    """
    fns = [deploy_model for _ in range(len(TICKERS))]
    fns_arg = [(ib, duration, freq, ticker) for ticker in TICKERS]
    print("=" * 15, "  Deploying models  ", "=" * 15)
    run_parallel(fns, fns_arg)
    print("=" * 15, "Finish the deployment", "=" * 15)


def main(client_id: int, duration: str, freq: int, ticker: str) -> None:
    """
    Main function to train and deploy the models
    """
    print(f"Conecting to IB {client_id + 1}...")
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=client_id + 1)
    print(f"Connected to IB {client_id + 1}!")
    train_model(ib, duration, freq, ticker)
    deploy_model(ib, duration, freq, ticker)


if __name__ == "__main__":
    print("=" * 30, "  Deploying models  ", "=" * 30)

    duration = "1 M"
    freq = 3
    fns = [main for _ in range(len(TICKERS))]
    fns_arg = [(idx, duration, freq, ticker) for idx, ticker in enumerate(TICKERS)]
    run_parallel(fns, fns_arg)

    print("=" * 30, "Finish the deployment", "=" * 30)