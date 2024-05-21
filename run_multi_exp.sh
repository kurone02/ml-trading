#!/bin/bash

tickers=("ACN" "ADBE" "AMD" "AKAM" "APH" "ADI" "ANSS" "AAPL" "AMAT" "ANET" "ADSK" "AVGO" "CDNS" "CDW" "CSCO" "CTSH" "GLW" "ENPH" "EPAM" "FFIV" "FICO" "FSLR" "FTNT" "IT" "GEN" "HPE" "HPQ" "IBM" "INTC" "INTU" "JBL" "JNPR" "KEYS" "KLAC" "LRCX" "MCHP" "MU" "MSFT" "MPWR" "MSI" "NTAP" "NVDA" "NXPI" "ON" "ORCL" "PANW" "PTC" "QRVO" "QCOM" "ROP" "CRM" "STX" "NOW" "SWKS" "SMCI" "SNPS" "TEL" "TDY" "TER" "TXN" "TRMB" "TYL" "VRSN" "WDC" "ZBRA" )

for ticker in "${tickers[@]}"; do
    python run_exp.py --ticker $ticker \
                      --train-start-date 2013-01-01 \
                      --train-end-date 2022-12-31 \
                      --test-start-date 2023-01-01 \
                      --test-end-date 2023-12-31
done
