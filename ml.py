import typing
import pandas as pd
import numpy as np
import numpy.typing as npt
from utils import load_financial_data, one_hot
from markov import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import AdamW
from models import *
from tqdm import tqdm
from ib_insync import *
import datetime
import time

class StockData(Dataset):
    def __init__(self, x_data, y_data, threshold: int=3) -> None:
        super().__init__()
        
        self.x = np.array(x_data).reshape((len(x_data), -1))
        y_data = np.array(y_data).reshape((len(y_data), -1))
        self.threshold = threshold

        self.x = np.array([one_hot(arr - STATE_TO_NUM["High Decrease"], len(STATE_TO_NUM)) for arr in self.x])
        self.x = torch.tensor(self.x, dtype=torch.float32)

        self.y = np.zeros((len(y_data), 3))
        for i, y in enumerate(y_data):
            x = self.x[i]
            past_ups = sum(x[x > STATE_TO_NUM["Neutral"]])
            past_downs = -sum(x[x < STATE_TO_NUM["Neutral"]])

            future_ups = sum(y[y > STATE_TO_NUM["Neutral"]])
            future_downs = -sum(y[y < STATE_TO_NUM["Neutral"]])

            if future_ups >= 2 * future_downs and \
                (past_ups >= 2 * future_downs or past_downs >= 2 * past_ups):
                self.y[i][2] = 1
            elif future_downs >= 2 * future_ups and \
                (past_downs >= 2 * future_downs or past_ups >= 2 * past_downs):
                self.y[i][0] = 1
            else:
                self.y[i][1] = 1

        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])



class MLStrategy(MarkovStrategy):

    def __init__(self, 
                 price: pd.Series, 
                 valid_data: pd.Series, 
                 test_data: pd.Series,
                 threshold_to_buy: int=2, 
                 threshold_to_sell: int=2,
                 lookback: int=5,
                 forward: int=5,
                 batch_size: int=128,
                ) -> None:
        self.historical_price: pd.Series = price
        self.threshold_to_buy = threshold_to_buy
        self.threshold_to_sell = threshold_to_sell
        self.lookback: int = lookback
        self.forward: int = forward

        # self.model = SimpleMLP(
        #     lookback=self.lookback,
        #     layer_dims=[
        #         16,
        #         16,
        #         8,
        #     ]
        # )

        self.model = SimpleLSTM(
            lookback=lookback,
            # hidden_size=16,
        )

        self.daily_return: pd.Series = self.historical_price.pct_change()
        self.quantiles_inc: pd.Series = self.daily_return[self.daily_return > 0].quantile([0.10, 0.5, 0.9])
        self.quantiles_dec: pd.Series = self.daily_return[self.daily_return < 0].quantile([0.10, 0.5, 0.9])

        self.historical_state: npt.ArrayLike = self.define_states(self.daily_return)
        self.batch_size = batch_size
        self.train_data = self.prepare_data(self.historical_state, self.lookback, self.forward, shuffle=True)

        historical_valid_state: npt.ArrayLike = self.define_states(valid_data.pct_change())
        self.valid_data = self.prepare_data(historical_valid_state, self.lookback, self.forward, shuffle=True)

        # historical_test_state: npt.ArrayLike = self.define_states(test_data.pct_change())
        self.test_price = test_data
        # self.test_data = self.prepare_data(historical_test_state, self.lookback, self.forward, shuffle=False)

    def define_states(self, data: pd.Series) -> npt.NDArray:
        """
        Define states based on quartiles of daily returns.
        """
        
        conditions = {
            'High Increase': data > self.quantiles_inc[0.9], 
            'Moderate Increase': (self.quantiles_inc[0.5] < data) & (data <= self.quantiles_inc[0.9]),
            'Slight Increase': (self.quantiles_inc[0.1] < data) & (data <= self.quantiles_inc[0.5]),
            
            'Neutral': (self.quantiles_dec[0.9] < data) & (data <= self.quantiles_inc[0.1]),
            
            'Slight Decrease': (self.quantiles_dec[0.5] < data) & (data <= self.quantiles_dec[0.9]),
            'Moderate Decrease': (self.quantiles_dec[0.1] < data) & (data <= self.quantiles_dec[0.5]),
            'High Decrease':data < self.quantiles_dec[0.1], 
        }

        return np.select(list(conditions.values()), list(conditions.keys()), default='Neutral')

    def update_transition_matrix(self, prev_state, next_state) -> None:
        self.transition_matrix.loc[prev_state, next_state] += 1

    def prepare_data(self, states, lookback: int=5, forward: int=5, shuffle: bool=True) -> DataLoader:
        """
        Prepare the data
        """

        x_data = []
        y_data = []

        for i in range(lookback - 1, len(states) - forward):
            prev = np.array(list(map(lambda x: STATE_TO_NUM[x], states[i-lookback+1:i+1])))
            curr = np.array(list(map(lambda x: STATE_TO_NUM[x], states[i+1:i+forward+1])))
            x_data.append(prev)
            y_data.append(curr)

        stock_data = StockData(x_data, y_data, self.threshold_to_buy)
        return DataLoader(stock_data, batch_size=self.batch_size, shuffle=shuffle)
    

    def train(self, 
              lr: float=1e-5,
              epochs: int=100,
            ) -> None:
        
        model = self.model
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(), 
            lr=lr,
        )

        for epoch in tqdm(range(epochs)):
            train_loss_epoch = 0
            train_num_correct = 0
            # print(f"Epoch {epoch}:")

            model.train()
            for (x, y) in self.train_data:
                batch_size = x.shape[0]
                x = x#.reshape(batch_size, -1) # flatten
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                train_loss_epoch += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_num_correct += torch.sum(y_pred.argmax(dim=1) == y.argmax(dim=1))

            valid_loss_epoch = 0
            valid_num_correct = 0
            model.eval()
            with torch.no_grad():
                for (x, y) in self.valid_data:
                    batch_size = x.shape[0]
                    x = x#.reshape(batch_size, -1) # flatten
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    valid_loss_epoch += loss.item()
                    # print("prediction:", y_pred.argmax(dim=1))
                    # print("ground truth:", y.argmax(dim=1))
                    valid_num_correct += torch.sum(y_pred.argmax(dim=1) == y.argmax(dim=1))

            
            # print(f"Trainning Loss: {train_loss_epoch / len(self.train_data):.6f}")
            # print(f"Trainning Accuracy: {train_num_correct / len(self.train_data.dataset):.6f}")
            # print(f"Validation Loss: {valid_loss_epoch / len(self.valid_data):.6f}")
            # print(f"Validation Accuracy: {valid_num_correct / len(self.valid_data.dataset):.6f}")
            # print("=" * 50)

    
    def backtest(self, 
                 initial_cash: int | float, 
                 num_share_per_trade: None | float | int=None,
                 limit_borrow: int | float=0,
                 limit_num_shorts: int | float=0,
                 allin: bool=False,
                 halfin: bool=False,
                ) -> pd.DataFrame:
        price = self.test_price
        daily_return = price.pct_change()
        daily_return.iloc[0] = 1

        states = self.define_states(daily_return)
        
        position = 0
        cur_cash = initial_cash
        cash = [cur_cash for _ in range(self.lookback-1)]
        positions = [0 for _ in range(self.lookback-1)]
        signals = [0 for _ in range(self.lookback-1)]

        if num_share_per_trade is None:
            num_share_per_trade = initial_cash / price.iloc[0]

        last_buy = 0
        last_sell = 0
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(self.lookback - 1, len(daily_return))):
                # cur_state = "-".join(states[i-self.lookback+1:i+1])
                cur_state = np.array(list(map(lambda x: STATE_TO_NUM[x], states[i-self.lookback+1:i+1])))
                cur_state = torch.tensor(one_hot(cur_state - STATE_TO_NUM["High Decrease"], len(STATE_TO_NUM)), 
                                         dtype=torch.float32).unsqueeze(0)#.reshape(1, -1)
                signal_pred = self.model(cur_state).argmax(dim=1).item() - 1
                signal = 0
                
                if allin:
                    num_share_per_trade = cur_cash / price.iloc[i]
                if halfin:
                    num_share_per_trade = cur_cash / price.iloc[i] / 2

                if cur_cash - num_share_per_trade * price.iloc[i] >= -limit_borrow \
                    and signal_pred == 1 and \
                    np.abs(price.iloc[i] - last_buy) >= min(price.iloc[i] * 0.01, 1):
                    position += num_share_per_trade
                    cur_cash -= num_share_per_trade * price.iloc[i]
                    signal = 1
                    last_buy = price.iloc[i]
            
                if position - num_share_per_trade >= -limit_num_shorts and \
                    signal_pred == -1 and \
                    np.abs(price.iloc[i] - last_sell) >= min(price.iloc[i] * 0.01, 1):
                    position -= num_share_per_trade
                    cur_cash += num_share_per_trade * price.iloc[i]
                    signal = -1
                    last_sell = price.iloc[i]
                
                signals.append(signal)
                cash.append(cur_cash)
                positions.append(position)

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
    

    def deploy(self, 
               ib: IB,
               ticker: str,
               initial_cash: int | float, 
               num_share_per_trade: None | float | int=None,
               limit_borrow: int | float=0,
               limit_num_shorts: int | float=0,
               freq: int = 30,
               is_crypto: bool=False,
               allin: bool=False,
               halfin: bool=False,
    ) -> pd.DataFrame:
        
        # util.startLoop()
        if is_crypto:
            stock = Crypto(ticker, 'Paxos', 'USD')
        else:
            stock = Stock(ticker, 'SMART', 'USD')

        position = 0
        cur_cash = initial_cash
        cash = []
        prices = []
        positions = []
        signals = []

        last_buy = 0
        last_sell = 0
        last_trade = 0

        self.model.eval()

        # log the signals
        with open(f"./log/{ticker}_signals.txt", "a") as f:
            while True:
                now = datetime.datetime.now()
                bars = ib.reqHistoricalData(
                    stock, 
                    endDateTime = '', 
                    durationStr = '1 D', 
                    barSizeSetting = f"{freq} min{'' if freq == 1 else 's'}", 
                    whatToShow = 'MIDPOINT', 
                    useRTH = True, 
                    formatDate = 2,
                    keepUpToDate=True,
                )
                financial_data = util.df(bars)
                
                price = financial_data["close"]
                daily_return = price.pct_change()
                daily_return.iloc[0] = 1

                states = self.define_states(daily_return)

                cur_state = np.array(list(map(lambda x: STATE_TO_NUM[x], states[-self.lookback:])))
                cur_state = torch.tensor(one_hot(cur_state - STATE_TO_NUM["High Decrease"], len(STATE_TO_NUM)), 
                                        dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    signal_pred = self.model(cur_state).argmax(dim=1).item() - 1

                prices.append(price.iloc[-1])

                commission = min(price.iloc[-1] * 0.01, 1)

                # Note that we are swapping the signal_pred here
                if cur_cash - num_share_per_trade * price.iloc[-1] >= -limit_borrow and \
                signal_pred == 1 and \
                np.abs(price.iloc[-1] - last_trade) > commission:
                    position += num_share_per_trade
                    cur_cash -= num_share_per_trade * price.iloc[-1]
                    cur_cash -= commission
                    signal = 1
                    last_trade = price.iloc[-1]
                    print(f"{now} Buy", file=f)
                    order = MarketOrder("Buy", num_share_per_trade)
                    ib.placeOrder(stock, order)
            
                elif position - num_share_per_trade >= -limit_num_shorts and \
                signal_pred == -1 and \
                np.abs(price.iloc[-1] - last_trade) > commission:
                    position -= num_share_per_trade
                    cur_cash += num_share_per_trade * price.iloc[-1]
                    cur_cash -= commission
                    signal = -1
                    last_trade = price.iloc[-1]
                    print(f"{now} Sell", file=f)
                    order = MarketOrder("Sell", num_share_per_trade)
                    ib.placeOrder(stock, order)
                else:
                    signal = 0
                    print(f"{now} No position", file=f)
                
                signals.append(signal)
                cash.append(cur_cash)
                positions.append(position)

                print(f"Garthering data for {freq} minute{'s' if freq > 1 else ''}", file=f)
                print("=" * 30, file=f)

                # save the results
                with open(f"./results/{ticker}_results.txt", "w") as f_res:
                    _prices = np.array(prices)
                    _positions = np.array(positions) * prices
                    _cash = np.array(cash)
                    _wealth = _cash + _positions

                    performance = pd.DataFrame()
                    performance["price"] = _prices
                    performance["signals"] = signals
                    performance["cash"] = _cash
                    performance["positions"] = _positions
                    performance["wealth"] = _wealth
                    performance.to_csv(f_res, index=False)

                ib.sleep(freq * 60)

                # if current time is 10 minutes before market close
                if now.time().hour == 4 and now.time().minute >= 50:
                    break

        return performance



def ml_train(
    train_start_date: str,
    train_end_date: str,
    valid_start_date: str,
    valid_end_date: str,
    test_start_date: str,
    test_end_date: str,
    ticker: str,
    threshold_to_buy: int=2, 
    threshold_to_sell: int=2,
    lookback: int=14,
    forward: int=7
) -> MLStrategy:
    
    security_data = load_financial_data(ticker, train_start_date, train_end_date)
    validation_data = load_financial_data(ticker, valid_start_date, valid_end_date)
    test_data = load_financial_data(ticker, test_start_date, test_end_date)

    markovian_strategy = MLStrategy(
        price=security_data["Adj Close"],
        valid_data=validation_data["Adj Close"],
        test_data=test_data["Adj Close"],
        threshold_to_buy=8,
        threshold_to_sell=8,
        lookback=lookback,
        forward=forward,
    )

    return markovian_strategy


def ml_test(
    markovian_strategy: MLStrategy,
    initial_cash: int | float,
    num_share_per_trade: float | int | None=None,
    limit_borrow: int | float=0,
    limit_num_shorts: int | float=0,
    allin: bool=False,
    halfin: bool=False,
) -> pd.DataFrame:
    
    performance = markovian_strategy.backtest(
        initial_cash=initial_cash,
        num_share_per_trade=num_share_per_trade,
        limit_borrow=limit_borrow,
        limit_num_shorts=limit_num_shorts,
        allin=allin,
        halfin=halfin
    )
    return performance


def ml_benchmark(
    ticker: str,
    train_start_date: str,
    train_end_date: str,
    valid_start_date: str,
    valid_end_date: str,
    test_start_date: str,
    test_end_date: str,
    lookback: int,
    forward: int,
    initial_cash: int | float,
    num_share_per_trade: float | int | None=None,
    limit_borrow: int | float=0,
    limit_num_shorts: int | float=0,
) -> pd.DataFrame:
    
    markovian_strategy = ml_train(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        valid_start_date=valid_start_date,
        valid_end_date=valid_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        ticker=ticker,
        lookback=lookback,
        forward=forward,
    )

    markovian_strategy.train(
        epochs=100,
        lr=1e-2,
    )

    markov_performance = ml_test(
        markovian_strategy=markovian_strategy,
        initial_cash=initial_cash,
        num_share_per_trade=num_share_per_trade,
        limit_borrow=limit_borrow,
        limit_num_shorts=limit_num_shorts
    )

    return markov_performance



