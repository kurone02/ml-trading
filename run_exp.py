from ml import MLStrategy, ml_benchmark
import matplotlib.pyplot as plt
import quantstats as qs
import numpy as np
from utils import save_txt
import pandas as pd

import os

import argparse

def save_vis_backtest(ticker, vis_dir, perf):
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(111, ylabel = 'Price in USD')
    perf.wealth.plot(ax=ax1, lw=2., label="net worth")
    perf.positions.plot(ax=ax1, lw=2., label="positions")
    perf.cash.plot(ax=ax1, lw=2., label="cash")

    ax1.plot(perf.loc[perf.signals == 1.0].index, perf.wealth[perf.signals == 1.0], '^', markersize=5, color='r', label="Buy")
    ax1.plot(perf.loc[perf.signals == -1.0].index, perf.wealth[perf.signals == -1.0], 'v', markersize=5, color='g', label="Sell")

    plt.xlabel("Date")
    plt.ylabel("Dolars")
    plt.title(f"Strategy Performance with {ticker}")
    plt.legend()
    
    vis_path = os.path.join(vis_dir, f"{ticker}.png")
    plt.savefig(vis_path)


def save_vis_compare(ticker, vis_dir, perf):
    price = (1 + perf.price.pct_change()).cumprod()
    wealth = (1 + perf.wealth.pct_change()).cumprod()

    plt.figure(figsize=(14, 7)) 
    plt.plot(price.index, price, label=ticker) 
    plt.plot(perf.index, wealth, label='ML Strategy')

    plt.plot(perf.loc[perf.signals == 1.0].index, price[perf.signals == 1.0], '^', markersize=5, color='r', label="Buy")
    plt.plot(perf.loc[perf.signals == -1.0].index, price[perf.signals == -1.0], 'v', markersize=5, color='g', label="Sell")

    plt.title(f"Compare the strategy's performance against {ticker} stock returns")
    plt.xlabel('Date') 
    plt.ylabel('Cumulative Returns') 
    plt.legend() 
    
    vis_path = os.path.join(vis_dir, f"{ticker}.png")
    plt.savefig(vis_path)

def save_sharpe(ticker, path, perf):
    returns = perf.wealth.pct_change(periods=21)
    text = "# Strategy Performance\n"
    text += f"Monthly Expected Return: {returns.mean():.3f}\n"
    text += f"Monthly Volatility: {returns.std():.3f}\n"
    text += f"Sharpe ratio: {returns.mean() / returns.std():.3f}\n"
    returns = perf.price.pct_change(periods=21)
    text += f"# Stock {ticker} Performance\n"
    text += f"Monthly Expected Return: {returns.mean():.3f}\n"
    text += f"Monthly Volatility: {returns.std():.3f}\n"
    text += f"Sharpe ratio: {returns.mean() / returns.std():.3f}\n"
    save_txt(path, text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="The stock's symbol", type=str, required=True)
    parser.add_argument("--train-start-date", help="The staring date to train the model", type=str, required=True)
    parser.add_argument("--train-end-date", help="The ending date to train the model", type=str, required=True)
    parser.add_argument("--test-start-date", help="The staring date to test the model", type=str, required=True)
    parser.add_argument("--test-end-date", help="The ending date to test the model", type=str, required=True)

    parser.add_argument("--lookback", help="The number of lookback periods", type=int, default=14)
    parser.add_argument("--forward", help="The number of forward periods", type=int, default=7)
    parser.add_argument("--initial-cash", help="The amount of initial cash for backtesting", type=int, default=1000)
    parser.add_argument("--num-per-trade", help="The number of shares per trade", type=int, default=1)

    parser.add_argument("--perf-dir", help="The directory to save the performance", type=str, default="./results/performance")
    parser.add_argument("--sharpe-dir", help="The directory to save the sharpe ratio", type=str, default="./results/sharpe")
    parser.add_argument("--vis-dir", help="The directory to save the visualizations", type=str, default="./results/vis")

    args = parser.parse_args()
    
    ticker = args.ticker

    train_start_date = args.train_start_date
    train_end_date = args.train_end_date
    test_start_date = args.test_start_date
    test_end_date = args.test_end_date

    lookback = args.lookback
    forward = args.forward
    initial_cash = args.initial_cash
    num_per_trade = args.num_per_trade

    vis_compare_dir = os.path.join(args.vis_dir, f"compare")
    if not os.path.exists(vis_compare_dir):
        os.makedirs(vis_compare_dir)

    vis_backtest_dir = os.path.join(args.vis_dir, f"backtest")
    if not os.path.exists(vis_backtest_dir):
        os.makedirs(vis_backtest_dir)

    print(f"Benchmarking {ticker}")

    perf = ml_benchmark(
        ticker=ticker,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        valid_start_date=test_start_date,
        valid_end_date=test_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        lookback=lookback,
        forward=forward,
        initial_cash=initial_cash,
        num_share_per_trade=num_per_trade,
    )

    csv_path = os.path.join(args.perf_dir, f"{ticker}.csv")
    perf.to_csv(csv_path)

    save_vis_backtest(ticker, vis_backtest_dir, perf)
    save_vis_compare(ticker, vis_compare_dir, perf)

    save_sharpe(ticker, os.path.join(args.sharpe_dir, f"{ticker}.md"), perf)