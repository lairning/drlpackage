# import itertools

import numpy as np
import pandas as pd

from lairningdecisions.trainer import Box

# Mandatory
BASE_CONFIG = {
    "SIM_DURATION": 80,  # Simulation time in Days
    "INITIAL_CASH": 10000
}

DEBUG = False


def dprint(*args):
    if DEBUG:
        print(*args)


ACTION_DICT = {0: 0, 1: -0.2, 2: 0.2, 3: -0.5, 4: 0.5}

cols_to_scale = ['Close', 'Volume', 'Open', 'High', 'Low', 'CloseMSFT', 'VolumeMSFT', 'OpenMSFT', 'HighMSFT',
                 'LowMSFT', 'CloseSPX', 'OpenSPX', 'HighSPX', 'LowSPX']
scaled_col_name = lambda col: 'Scaled' + col
scaled_cols = [scaled_col_name(col) for col in cols_to_scale]


def get_data(csv_name: str = 'trading_data.csv'):
    new_names = {'Close/Last': 'Close', 'Close/Last.1': 'CloseMSFT', 'Volume.1': 'VolumeMSFT', 'Open.1': 'OpenMSFT',
                 'High.1'    : 'HighMSFT', 'Low.1': 'LowMSFT', 'Close/Last.2': 'CloseSPX', 'Open.2': 'OpenSPX',
                 'High.2'    : 'HighSPX', 'Low.2': 'LowSPX'}
    df = pd.read_csv(csv_name, sep=';', parse_dates=['Date']).rename(columns=new_names).sort_values(
        by='Date', ignore_index=True)
    df['WeekDay'] = df['Date'].apply(lambda x: x.weekday())
    for col in cols_to_scale:
        min = df[col].min()
        max = df[col].max()
        df[scaled_col_name(col)] = df[col].apply(lambda x: (x - min) / (max - min))
    return df


N_ACTIONS = 5  # 0 - DoNothing; '-20' - Sell 20%; '20' - Buy 20%; '-50' - Sell 50%; '50' - Buy 50%
OBSERVATION_SPACE = Box(low=np.array([0, 0] + [0] * len(cols_to_scale) + [0]),
                        # Balance, quant shares, scaled values * 14, weekday
                        high=np.array([np.inf, np.inf] + [1] * len(cols_to_scale) + [6]),
                        dtype=np.float64)


class SimModel():
    def __init__(self, config: dict = None, data: pd.DataFrame = None):
        assert data is not None, 'A Data Frame is required'
        self.data = data
        self.sim_config = BASE_CONFIG.copy()
        if config is not None:
            self.sim_config.update(config)
        self.start = np.random.choice(len(data) - self.sim_config['SIM_DURATION'] - 1)
        self.cash = self.sim_config['INITIAL_CASH']
        self.idx = self.start
        self.shares = 0
        self.last_value = self.cash

    def get_observation(self):
        return [self.cash, self.shares * self.data['Close'].iloc[self.idx]] + \
               list(self.data[scaled_cols].iloc[self.idx].values) + [self.data['WeekDay'].iloc[self.idx]]

    def get_reward(self):
        value = self.data['Close'].iloc[self.idx] * self.shares + self.cash
        reward = value - self.last_value
        self.last_value = value
        return reward, (self.idx - self.start) >= self.sim_config["SIM_DURATION"], {}  # Reward, Done, Info

    # Executes an action
    def exec_action(self, action):
        def price(data):
            return np.random.triangular(data['Low'], data['Low'] + (data['High'] - data['Low']) / 2, data['High'])

        market_data = self.data.iloc[self.idx + 1]
        stock_price = price(market_data)
        # print("# exec_action:", market_data['High'], stock_price, market_data['Low'])
        percent = ACTION_DICT[action]
        action_cost = 0
        if percent > 0.0:
            action_cost = self.cash * percent
            self.shares += action_cost / stock_price
        if percent < 0.0:
            action_cost = percent * self.shares * stock_price
            self.shares = self.shares * (1 + percent)
        self.cash -= action_cost
        self.idx += 1


# Mandatory
class SimBaseline:
    def __init__(self, sim_config: dict = None, data: pd.DataFrame = None):
        self.sim = None
        self.sim_config = BASE_CONFIG.copy()
        if sim_config is not None:
            self.sim_config.update(sim_config)
        self.data = data

    def get_action(self, obs):
        return np.random.randint(5)

    def run(self):
        self.sim = SimModel(self.sim_config, self.data)
        done = False
        total_reward = 0
        while not done:
            obs = self.sim.get_observation()
            action = self.get_action(obs)
            self.sim.exec_action(action)
            reward, done, _ = self.sim.get_reward()
            total_reward += reward

        return total_reward


if __name__ == "__main__":
    n = 1
    total = 0
    config = {
        "SIM_DURATION": 80,  # Simulation time in Days
        "INITIAL_CASH": 10000
    }
    for _ in range(n):
        baseline = SimBaseline(sim_config=config, data=get_data())
        reward = baseline.run()
        total += reward
    print("### Average Rewards", total / n)
