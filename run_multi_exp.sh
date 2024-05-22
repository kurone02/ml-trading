#!/bin/bash


sectors=("Industrials" "Health Care" "Information Technology" "Utilities" "Financials" "Materials" "Consumer Discretionary" "Real Estate" "Communication Services" "Consumer Staples" "Energy")

for sector in "${sectors[@]}"; do
    # echo python get_tickers.py --sector \"$sector\"
    tickers=($(python get_tickers.py --sector "$sector"))
    echo "Running sector "$sector" in S&P500..."
    mkdir -p results/"$sector"/performance
    mkdir -p results/"$sector"/sharpe
    mkdir -p results/"$sector"/vis
    for ticker in "${tickers[@]}"; do
            python run_exp.py --ticker $ticker \
                              --train-start-date 2013-01-01 \
                              --train-end-date 2022-12-31 \
                              --test-start-date 2023-01-01 \
                              --test-end-date 2023-12-31 \
                              --perf-dir results/"$sector"/performance \
                              --sharpe-dir results/"$sector"/sharpe \
                              --vis-dir results/"$sector"/vis
    done
done