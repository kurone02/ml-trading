import FinanceDataReader as fdr 

import argparse

def find_tickers_sp500(sector):
    df_sp500 = fdr.StockListing('S&P500') # list of stocks in S&P500
    tickers = list(df_sp500[df_sp500['Sector']==sector]['Symbol']) # Filter those in the sector
    return tickers # Return the tickers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", help="The stocks' sector", type=str, required=True)

    args = parser.parse_args()

    sector = args.sector

    SECTOR_LIST = ['Industrials', 'Health Care', 'Information Technology', 'Utilities', 'Financials', 'Materials', 'Consumer Discretionary', 'Real Estate', 'Communication Services', 'Consumer Staples', 'Energy']

    if sector not in SECTOR_LIST:
        print("ERROR: --sector should be one of", SECTOR_LIST)

    # print("(", end='')
    for tic in find_tickers_sp500(sector):
        print(f"{tic} ", end='')
    # print(")")