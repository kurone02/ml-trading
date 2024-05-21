import typing
import pandas as pd
import numpy as np
import numpy.typing as npt
from utils import load_financial_data

STATE_COLORS = {
    'High Increase': 'green', 
    'Moderate Increase': 'lightgreen', 
    'Slight Increase': 'yellowgreen', 
    'Neutral': 'grey', 
    'Slight Decrease': 'salmon', 
    'Moderate Decrease': 'red', 
    'High Decrease': 'darkred'
}    

STATE_TO_NUM = {
    'High Increase': 3,
    'Moderate Increase': 2,
    'Slight Increase': 1,
    
    'Neutral': 0,
    
    'Slight Decrease': -1,
    'Moderate Decrease': -2,
    'High Decrease': -3,
}


class MarkovStrategy():

    def __init__(self, 
                 price: pd.Series, 
                 threshold_to_buy: int=2, 
                 threshold_to_sell: int=2,
                 lookback: int=5,
                ) -> None:
        self.historical_price: pd.Series = price
        self.threshold_to_buy = threshold_to_buy
        self.threshold_to_sell = threshold_to_sell
        self.lookback: int = lookback

        self.daily_return: pd.Series = self.historical_price.pct_change()
        self.quantiles_inc: pd.Series = self.daily_return[self.daily_return > 0].quantile([0.25, 0.5, 0.75])
        self.quantiles_dec: pd.Series = self.daily_return[self.daily_return < 0].quantile([0.25, 0.5, 0.75])

        self.historical_state: npt.ArrayLike = self.define_states(self.daily_return)
        self.transition_matrix: pd.DataFrame = self.calculate_transition_matrix(self.historical_state, lookback=self.lookback)
        self.predict = self.transition_matrix.idxmax(axis=1)


    def define_states(self, data: pd.Series) -> npt.NDArray:
        """
        Define states based on quartiles of daily returns.
        """
        
        conditions = {
            'High Increase': data > self.quantiles_inc[0.75], 
            'Moderate Increase': (self.quantiles_inc[0.5] < data) & (data <= self.quantiles_inc[0.75]),
            'Slight Increase': (self.quantiles_inc[0.25] < data) & (data <= self.quantiles_inc[0.5]),
            
            'Neutral': (self.quantiles_dec[0.75] < data) & (data <= self.quantiles_inc[0.25]),
            
            'Slight Decrease': (self.quantiles_dec[0.5] < data) & (data <= self.quantiles_dec[0.75]),
            'Moderate Decrease': (self.quantiles_dec[0.25] < data) & (data <= self.quantiles_dec[0.5]),
            'High Decrease':data < self.quantiles_dec[0.25], 
        }

        return np.select(list(conditions.values()), list(conditions.keys()), default='Neutral')

    def update_transition_matrix(self, prev_state, next_state) -> None:
        self.transition_matrix.loc[prev_state, next_state] += 1

    def calculate_transition_matrix(self, states, lookback=5) -> pd.DataFrame:
        """
        Calculate the state transition matrix.
        """
        unique_states = np.unique(states)
        state_space_id = list(np.ndindex(tuple([len(unique_states) for _ in range(lookback)])))
        state_space = []
        for state in state_space_id:
            state_space.append("-".join(list(map(lambda x: unique_states[x], state))))
        matrix = pd.DataFrame(0, index=state_space, columns=state_space, dtype=float)

        for i in range(lookback - 1, len(states) - lookback):
            prev = states[i-lookback+1:i+1]
            curr = states[i+1:i+lookback+1]
            matrix.loc["-".join(prev), "-".join(curr)] += 1

        return matrix
    
    def backtest(self, 
                 price: pd.Series, 
                 initial_cash: int | float, 
                 num_share_per_trade: None | float | int=None,
                 limit_borrow: int | float=0,
                 limit_num_shorts: int | float=0,
                ) -> pd.DataFrame:
        daily_return = price.pct_change()
        daily_return.iloc[0] = 1

        states = self.define_states(daily_return)
        
        position = 0
        cur_cash = initial_cash
        cash = [cur_cash for _ in range(self.lookback-1)]
        positions = [0 for _ in range(self.lookback-1)]
        signals = [0 for _ in range(self.lookback-1)]

        if num_share_per_trade is None:
            num_share_per_trade = int(initial_cash / price.iloc[0])

        for i in range(self.lookback - 1, len(daily_return)):
            cur_state = "-".join(states[i-self.lookback+1:i+1])
            signal = 0
            if cur_state in self.predict and not pd.isna(self.predict[cur_state]):
                if cur_cash - num_share_per_trade * price.iloc[i] >= -limit_borrow and\
                ((cur_state.count("Decrease") >= self.threshold_to_buy and self.predict[cur_state].count("Increase") >= self.threshold_to_buy) or\
                (cur_state.count("Increase") >= self.threshold_to_buy and self.predict[cur_state].count("Increase") >= self.threshold_to_buy)):
                    position += num_share_per_trade
                    cur_cash -= num_share_per_trade * price.iloc[i]
                    signal = 1
            
                elif position - num_share_per_trade >= -limit_num_shorts and\
                ((cur_state.count("Increase") >= self.threshold_to_sell and self.predict[cur_state].count("Decrease") >= self.threshold_to_sell) or\
                (cur_state.count("Decrease") >= self.threshold_to_buy and self.predict[cur_state].count("Decrease") >= self.threshold_to_buy)):
                    position -= num_share_per_trade
                    cur_cash += num_share_per_trade * price.iloc[i]
                    signal = -1
            
            signals.append(signal)
            cash.append(cur_cash)
            positions.append(position)
            if i >= 2 * self.lookback - 1:
                prev_state = "-".join(states[i - 2 * self.lookback + 1:i-self.lookback+1])
                self.update_transition_matrix(prev_state, cur_state)

        positions = np.array(positions) * price
        cash = np.array(cash)
        wealth = cash + positions

        performance = pd.DataFrame(index=price.index)
        performance["price"] = price
        performance["daily_return"] = daily_return
        performance["signals"] = signals
        performance["cash"] = cash
        performance["positions"] = positions
        performance["wealth"] = wealth

        return performance




def markov_train(
    start_date: str,
    end_date: str,
    ticker: str,
    threshold_to_buy: int=2, 
    threshold_to_sell: int=2,
    lookback: int=5,
) -> MarkovStrategy:
    
    security_data = load_financial_data(ticker, start_date, end_date)
    markovian_strategy = MarkovStrategy(
        price=security_data["Adj Close"],
        threshold_to_buy=threshold_to_buy,
        threshold_to_sell=threshold_to_sell,
        lookback=lookback
    )

    return markovian_strategy


def markov_test(
    start_date: str,
    end_date: str,
    ticker: str,
    markovian_strategy: MarkovStrategy,
    initial_cash: int | float,
    num_share_per_trade: float | int | None=None,
    limit_borrow: int | float=0,
    limit_num_shorts: int | float=0,
) -> pd.DataFrame:
    
    security_data = load_financial_data(ticker, start_date, end_date)
    performance = markovian_strategy.backtest(
        price=security_data["Adj Close"],
        initial_cash=initial_cash,
        num_share_per_trade=num_share_per_trade,
        limit_borrow=limit_borrow,
        limit_num_shorts=limit_num_shorts,
    )
    return performance


def markov_benchmark(
    ticker: str,
    train_start_date: str,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
    threshold_to_buy: int, 
    threshold_to_sell: int,
    lookback: int,
    initial_cash: int | float,
    num_share_per_trade: float | int | None=None,
    limit_borrow: int | float=0,
    limit_num_shorts: int | float=0,
) -> pd.DataFrame:
    
    markovian_strategy = markov_train(
        start_date=train_start_date,
        end_date=train_end_date,
        ticker=ticker,
        threshold_to_buy=threshold_to_buy,
        threshold_to_sell=threshold_to_sell,
        lookback=lookback,
    )

    markov_performance = markov_test(
        start_date=test_start_date,
        end_date=test_end_date,
        ticker=ticker,
        markovian_strategy=markovian_strategy,
        initial_cash=initial_cash,
        num_share_per_trade=num_share_per_trade,
        limit_borrow=limit_borrow,
        limit_num_shorts=limit_num_shorts
    )

    return markov_performance