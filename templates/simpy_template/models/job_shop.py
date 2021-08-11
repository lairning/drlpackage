import itertools
import numpy as np
import simpy
from lairningdecisions.trainer import SimpyModel, Box

MACHINES = ['M1', 'M2', 'M3']
PRODUCTS = ['P1', 'P2', 'P3']
JOB_DURATION = {'M1': {'P1': 5, 'P2': 3, 'P3': 2},
                'M2': {'P1': 3, 'P2': 5, 'P3': 2},
                'M3': {'P1': 3, 'P2': 4, 'P3': 6}}

# Mandatory
BASE_CONFIG = {
    "SIM_DURATION": 10 * 60,  # Simulation time in minutes
    "ACTION_INTERVAL": 1,  # Time in minutes between each action
    "JOB_DURATION": JOB_DURATION,
    "PRODUCT_REVENUE": {'P1': 10, 'P2': 8, 'P3': 5},
    "DEMAND_PROBABILITY": {'P1': 1 / 10, 'P2': 1 / 10, 'P3': 1 / 10},
    "STOCK_COST": 0.02
}

N_ACTIONS = 1 + len(PRODUCTS)  # 0 - Nothing; 1 - Product 1; 2 - Product 2; 3 - Product 3
# Observation:
#   Stock of each product
low = [0] * len(PRODUCTS)
high = [np.inf] * len(PRODUCTS)
#   Time each machine will take to be free
low += [0] * len(MACHINES)
high += [max(JOB_DURATION[m].values()) for m in MACHINES]
# Is M1, M2, M3 is working on one product?
low += [0] * len(PRODUCTS) * len(MACHINES)
high += [1] * len(PRODUCTS) * len(MACHINES)
#   Time it will take to end
low += [0]
high += [BASE_CONFIG["SIM_DURATION"]]

OBSERVATION_SPACE = Box(low=np.array(low),
                        high=np.array(high),
                        dtype=np.float64)


class SimModel(SimpyModel):
    def __init__(self, config: dict = None):
        super().__init__(BASE_CONFIG, config)
        self.stock = {p: simpy.Container(self, init=0) for p in PRODUCTS}  # Infinite capacity
        self.machine = {m: simpy.Resource(self, capacity=1) for m in MACHINES}
        for p in PRODUCTS:
            self.process(self.demand_generator(p))
        self.sales_revenue = 0
        self.machine_start_time = {m: -1 for m in MACHINES}  # -1 means it's stopped
        self.machine_product = {m: "" for m in MACHINES}  # "" means no product
        self.info = []

    # Agent gets the state of the environment
    def get_observation(self):
        def time_to_finish(machine):
            if self.machine_product[machine] == "":
                return 0
            product = self.machine_product[machine]
            ttf = self.sim_config['JOB_DURATION'][machine][product] - (self.now - self.machine_start_time[machine])
            if ttf < 0:
                raise Exception("Invalid Time to Finish: machine={}, product={}, now={}, start={}, duration={}".format(
                    machine, product, self.now, self.machine_start_time[machine],
                    self.sim_config['JOB_DURATION'][machine][product]))
            return ttf

        def product_encode(machine):
            p_encode = [0] * len(PRODUCTS)
            if self.machine_product[machine] != "":
                p_encode[PRODUCTS.index(self.machine_product[machine])] = 1
            return p_encode

        self.run_until_action()
        env_status = [self.stock[p].level for p in PRODUCTS]
        env_status += [time_to_finish(m) for m in MACHINES]
        env_status += [mp for m in MACHINES for mp in product_encode(m)]
        env_status += [self.sim_config['SIM_DURATION'] - self.now]
        return env_status

    # Agent Executes an action
    def exec_action(self, action):
        if action and self.machine['M1'].count:
            # The machine is not free
            return
        if action:
            self.process(self.job_schedule(PRODUCTS[action - 1]))

    # Agent Get the reward
    def get_reward(self):
        stock_cost = sum([self.stock[p].level * self.sim_config["PRODUCT_REVENUE"][p] * self.sim_config["STOCK_COST"]
                          for p in PRODUCTS])
        # self.info += [{'event_type':'stock_cost', 'product':p, 'cost':self.stock[p].level * self.sim_config["PRODUCT_REVENUE"][p] * self.sim_config["STOCK_COST"]} for p in PRODUCTS if self.stock[p].level)
        revenue = self.sales_revenue - stock_cost
        info = {'info': self.info }
        self.sales_revenue = 0
        self.info = []
        return revenue, self.done(), info  # Reward, Done, Info

    # Product Schedule definition
    def job_schedule(self, product):
        """Schedule the production for each product sequentaly starting in M1."""
        for machine in MACHINES:
            with self.machine[machine].request() as req:
                yield req
                self.machine_start_time[machine] = self.now
                self.machine_product[machine] = product
                if machine == 'M1':
                    start = self.now
                yield self.timeout(self.sim_config['JOB_DURATION'][machine][product])
                self.info.append({'event_type':'schedule', 'start_time':start, 'product':product, 'machine':machine, 'start':self.machine_start_time[machine], 'stop':self.now})
                self.machine_start_time[machine] = -1
                self.machine_product[machine] = ""
        # self.info.append({'event_type':'duration', 'start':start, 'product':product, 'stop':self.now})
        yield self.stock[product].put(1)

    # Demand generation model
    def demand_generator(self, product):
        """Generate new product requests that arrive at the factory."""
        for _ in itertools.count():
            yield self.timeout(1)
            amount = np.random.binomial(1, self.sim_config["DEMAND_PROBABILITY"][product])  # Bernouli distribution
            if amount:
                if self.stock[product].level:
                    yield self.stock[product].get(1)
                    self.sales_revenue += self.sim_config['PRODUCT_REVENUE'][product]
                    # self.info.append({'event_type':'demand_satisfied', 'product':product, 'revenue':self.sim_config['PRODUCT_REVENUE'][product]})
                else:
                    pass
                    # self.info.append({'event_type':'demand_non_satisfied', 'product':product, 'revenue':self.sim_config['PRODUCT_REVENUE'][product]})

# To benchmark the AI Agent Performance
class SimBaseline:
    def __init__(self, baseline_config: dict = None,
                 sim_config: dict = None):
        self.sim = None
        self.baseline_config = dict()
        if baseline_config is not None:
            self.baseline_config.update(baseline_config)
        self.sim_config = BASE_CONFIG.copy()
        if sim_config is not None:
            self.sim_config.update(sim_config)

    def get_action(self, obs):
        if self.sim.machine_product['M1'] != "":
            return 0
        product_stock = obs[:len(PRODUCTS)]
        min_stock = min(product_stock)
        if min_stock >= 1:
            return 0
        for i, s in enumerate(product_stock):
            if s == min_stock:
                return i + 1
        return 0

    def run(self):
        self.sim = SimModel(self.sim_config)
        done = False
        total_reward = 0
        while not done:
            obs = self.sim.get_observation()
            action = self.get_action(obs)
            self.sim.exec_action(action)
            reward, done, _ = self.sim.get_reward()
            total_reward += reward

        return total_reward


# Used for testing the simulator and the baseline
if __name__ == "__main__":
    n = 20
    total_reward = 0
    for i in range(n):
        baseline = SimBaseline()
        reward = baseline.run()
        total_reward += reward
        # print("# Reward={}".format(reward))
    print("# TOTAL # Avg Reward={}".format(total_reward / n))
