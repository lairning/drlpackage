from typing import Tuple

from simpy.events import AnyOf

import numpy as np
import simpy
from lairningdecisions.trainer import Box

# Two factories located in "Condeixa a Nova" (FCN) and "Vendas Novas (FVN)"
# Assumption: Factory have ilimited stock and production capacity and no stock costs. Only one product is considered.
FACTORIES = ['FCN', 'FVN']
FACTORY_LOCATION = 1

# Distribution Centers in Oporto (DP), Coimbra (DC), Lisbon (DL) and Faro (DF)
DISTRIBUTION_CENTERS = ['DP', 'DC', 'DL', 'DF']
DISTRIBUTION_CENTER_LOCATION = 2

# TRAVEL TIME IN MINUTES
TRAVEL_TIME_FCN = {('FCN','DP'):80,  ('FCN','DC'):25, ('FCN','DL'):130, ('FCN','DF'):240, ('FCN','FVN'):140}
TRAVEL_TIME_FVN = {('FVN','DP'):200,  ('FVN','DC'):140, ('FVN','DL'):70, ('FVN','DF'):140}
TRAVEL_TIME = { **TRAVEL_TIME_FCN, **{(d,o):t for (o,d),t in TRAVEL_TIME_FCN.items()}, 
                **TRAVEL_TIME_FVN, **{(d,o):t for (o,d),t in TRAVEL_TIME_FVN.items()}}

N_TRUCKS = 3

# Mandatory
BASE_CONFIG = {
    "SIM_DURATION"    : 20 * 24 * 60,  # Simulation time in minutes
    "DEMAND_INTERVAL" : 10,  # Time in minutes between each demand (order)
    "DEMAND_MEAN"     : {'DP':50/60, 'DC':20/60, 'DL':90/60, 'DF':15/60}, # Products per minute
    "REVENUE_UNIT"    : 10,  # Revenue per product
    "STOCK_COST_UNIT_MINUTE" : 1 / 60,  # Cost of holding a unit of product each minute
    "TRUCK_CAPACITY"  : 100,  # Amount of prosuct a truc can accomodate
    "KM_COST"         : 1.5,  # Truck cost per km
    "TRUCK_COST"      : 150,  # Fixed Transportation Cost
    "TRUCKS_INITIAL_FACTORY" : ["FVN", "FVN", "FCN"],  # Factory location un the beggining of thre simulation
  }

DEBUG = False


def dprint(*args):
    if DEBUG:
        print(*args)


def hot_encode(n, N):
    encode = [0] * N
    encode[n] = 1
    return encode


N_ACTIONS = 1 + len(FACTORIES) + len(DISTRIBUTION_CENTERS)

# Stock in the Distribution Centers
low = [0]*len(DISTRIBUTION_CENTERS)
high = [np.inf]*len(DISTRIBUTION_CENTERS)
# Truck Type of Location Status
low += [0]*N_TRUCKS
high += [2]*N_TRUCKS
# Truck Location Id Status
low += [0]*N_TRUCKS
high += [max(len(FACTORIES),len(DISTRIBUTION_CENTERS))]*N_TRUCKS
# Truck Time to Arrive
low += [0]*N_TRUCKS
high += [max(TRAVEL_TIME.values())]*N_TRUCKS

OBSERVATION_SPACE = Box(low=np.array(low),
                        high=np.array(high),
                        dtype=np.float64)

def map_action(action:int) -> Tuple: # Return: Type of Location: 0 (nothing), 1 (Factory), 2 (DC). Location id: index for the list
    assert action < N_ACTIONS

    if action >= 1 and action <= len(FACTORIES):
        return 1, action-1 # Type of Location 1 = Factory
    
    if action > len(FACTORIES):
        return 2, action-len(FACTORIES)-1 # Type of Location = Distribution Center

    return 0, 0 

def location_name(type_of_location, location_id):
    if type_of_location == FACTORY_LOCATION:
        return FACTORIES[location_id]
    if type_of_location == DISTRIBUTION_CENTER_LOCATION:
        return DISTRIBUTION_CENTERS[location_id]
    return ""

class Truck(simpy.Resource):
    def __init__(self, env, factory_location_id) -> None:
        super().__init__(env, 1)
        self.arrival_time = 0 # Travel arrival time
        self.location_type = FACTORY_LOCATION # Truck current/next DC location. 0 means no DC assigned, 1 for 1st DC, ...
        self.location_id = factory_location_id
        self.event = None
    
    def is_free(self):
        return self.count == 0

class SimModel(simpy.Environment):
    def __init__(self, config: dict = None):
        super().__init__()
        self.sim_config = BASE_CONFIG.copy()
        if config is not None:
            self.sim_config.update(config)
        self.sim_duration = self.sim_config["SIM_DURATION"]
        self.trucks = [Truck(self, FACTORIES.index(factory)) for factory in self.sim_config["TRUCKS_INITIAL_FACTORY"]]
        self.dc = [simpy.Container(self,init=0) for _ in DISTRIBUTION_CENTERS]
        self.revenue = 0
        self.stock_cost = 0
        self.travel_cost = 0
        self.process(self.demand_generator(time_interval = 10))
        for truck in self.trucks:
            truck.event = self.process(self.truck_wait(truck, 0))
        self.run(until=AnyOf(self,[truck.event for truck in self.trucks]))

    def get_observation(self):
        # Stock in Distribution Centers
        env_status = [dc.level for dc in self.dc]
        # Truck Type of Location Status
        env_status += [truck.location_type for truck in self.trucks]
        # Truck Location Id Status
        env_status += [truck.location_id for truck in self.trucks]        
        # Truck Time to Arrive
        env_status += [max(truck.arrival_time-self.now, 0) for truck in self.trucks]       
        return env_status

    def exec_action(self, action):
        type_of_location, location_id = map_action(action=action)
        free_trucks = [truck for truck in self.trucks if truck.is_free()]
        #print(self.now, location_name(type_of_location, location_id), [t.event for t in self.trucks])
        assert len(free_trucks) > 0, "At leats one free truck should be available"
        if not type_of_location \
            or (type_of_location == DISTRIBUTION_CENTER_LOCATION and free_trucks[0].location_type == DISTRIBUTION_CENTER_LOCATION) \
            or (type_of_location == FACTORY_LOCATION and free_trucks[0].location_type == FACTORY_LOCATION and location_id== free_trucks[0].location_id): 
            # do nothing and wait
            # Put the first free truck on wait mode. To allow to schedule other trucks the waiting time may be higher than 1
            free_trucks[0].event = self.process(self.truck_wait(free_trucks[0], len(free_trucks)))
            try:
                self.run(until=AnyOf(self,[truck.event for truck in self.trucks]))
            except Exception as e:
                print(self.now, location_name(type_of_location, location_id), [t.event for t in self.trucks])
                raise e            
        else: # dispatch a truck
            free_trucks[0].event = self.process(self.dispatch_truck(free_trucks[0], type_of_location, location_id))
            if len(free_trucks) > 1: # dispatch a new truck on the next time unit
                self.run(until=self.now+1)
            else: # dispatch a new truck when one becomes free
                try:
                    self.run(until=AnyOf(self,[truck.event for truck in self.trucks]))
                except Exception as e:
                    print(self.now, location_name(type_of_location, location_id), [t.event for t in self.trucks])
                    raise e

    def get_reward(self):
        reward = self.revenue - self.stock_cost - self.travel_cost
        info = {'revenue':self.revenue, 'stock_cost':self.stock_cost, 'travel_cost':self.travel_cost}
        self.revenue, self.stock_cost, self.travel_cost = 0, 0, 0
        return reward, self.now >= self.sim_duration, info # Reward, done, info

    def truck_wait(self, truck: Truck, waiting_time: int):
        with truck.request() as req:
            yield self.timeout(waiting_time)

    def dispatch_truck(self, truck: Truck, type_of_location, location_id):
        origin = location_name(truck.location_type,truck.location_id)
        destination = location_name(type_of_location, location_id)
        truck.location_type = type_of_location
        truck.location_id = location_id
        truck.arrival_time = self.now + TRAVEL_TIME[(origin,destination)]
        with truck.request() as req:
            yield self.timeout(TRAVEL_TIME[(origin,destination)])
            if location_id == DISTRIBUTION_CENTER_LOCATION:
                yield self.dc[location_id].put(self.sim_config["TRUCK_CAPACITY"])
            self.travel_cost += TRAVEL_TIME[(origin,destination)]*self.sim_config["KM_COST"]

    def demand_generator(self, time_interval):
        def demand_sample(dc_id, time_interval):
            demand_avg = self.sim_config["DEMAND_MEAN"][DISTRIBUTION_CENTERS[dc_id]]*time_interval
            demand_std = demand_avg
            return max(np.random.normal(demand_avg, demand_std),0)
        while True:
            yield self.timeout(time_interval)
            for i, dc in enumerate(self.dc):
                if dc.level > 0:
                    self.stock_cost += dc.level*self.sim_config["STOCK_COST_UNIT_MINUTE"]*time_interval
                    amount = demand_sample(i, time_interval)
                    if amount > 0:
                        if dc.level < amount:
                            amount = dc.level
                        yield dc.get(amount)
                        self.revenue += amount*self.sim_config["REVENUE_UNIT"]

# Mandatory
class SimBaseline:
    def __init__(self, baseline_config: dict = None, sim_config: dict = None):
        self.sim = None
        self.baseline_config = dict()
        if baseline_config is not None:
            self.baseline_config.update(baseline_config)
        self.sim_config = BASE_CONFIG.copy()
        if sim_config is not None:
            self.sim_config.update(sim_config)
        self.info = {'revenue':0, 'stock_cost':0, 'travel_cost':0}

    def get_action(self, obs):
        return np.random.randint(N_ACTIONS)

    def run(self):
        #self.sim = SimModel(self.sim_config)
        self.sim = SimModel()
        done = False
        total_reward = 0
        while not done:
            obs = self.sim.get_observation()
            action = self.get_action(obs)
            self.sim.exec_action(action)
            reward, done, info = self.sim.get_reward()
            for k,v in info.items():
                self.info[k] += v
            total_reward += reward

        return total_reward

if __name__ == "__main__":
    n = 5
    total = 0
    for _ in range(n):
        baseline = SimBaseline()
        reward = baseline.run()
        total += reward
        print("# Reward", reward, baseline.info)
    print("### Average Rewards", total / n)
