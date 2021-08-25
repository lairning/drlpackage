import itertools
import random
from typing import Tuple

from simpy.events import AnyOf

import numpy as np
import simpy
from lairningdecisions.trainer import SimpyModel, Box
from scipy.stats import truncnorm

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
OBSERVATION_SPACE = Box(low=np.array([0]),
                        high=np.array([1]),
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
    def __init__(self, env, factory_location) -> None:
        super().__init__(env, 1)
        self.arrival_time = 0 # Travel arrival time
        self.location_type = FACTORY_LOCATION # Truck current/next DC location. 0 means no DC assigned, 1 for 1st DC, ...
        self.location_id = factory_location
        self.event = None
    
    def is_free(self):
        return self.count == 0

class SimModel(SimpyModel):
    def __init__(self, config: dict = None):
        super().__init__(BASE_CONFIG, config)
        self.fuel_pump = simpy.Container(self, GAS_STATION_SIZE, init=GAS_STATION_SIZE / 2)
        self.gas_station = simpy.Resource(self, PUMP_NUMBER)
        self.process(self.car_generator(self.gas_station, self.fuel_pump))
        self.actual_revenue = 0
        self.last_revenue = 0
        self.trucks = [1,2]
        self.trucks_events = dict()
        self.trucks_waiting = {truck:True for truck in self.trucks}

    def get_observation(self):
        # self.run_until_action()
        hour_mask = hot_encode((self.now // 60) % 24, 24)
        env_status = [self.fuel_pump.level, self.gas_station.count] + hour_mask
        return env_status

    def get_reward(self):
        revenue = self.actual_revenue - self.last_revenue
        self.last_revenue = self.actual_revenue
        return revenue, self.done(), {}  # Reward, Done, Info

    # Executes an action
    # 0 nothing, 1 truck1, 2 truck2, 3 both trucks
    def exec_action(self, action):
        if self.truck_wait[1]:
            if action in [1,3]:
                self.trucks_events[1] = self.process(self.tank_truck(self.fuel_pump, 1))
                self.trucks_waiting[1] = False
            else:
                self.trucks_events[1] = self.process(self.truck_wait(1, self.step_time))
        if self.truck_wait[2]:
            if action in [2,3]:
                self.trucks_events[2] = self.process(self.tank_truck(self.fuel_pump, 2))
                self.trucks_waiting[2] = False
            else:
                self.trucks_events[2] = self.process(self.truck_wait(2, self.step_time))
        result = self.run(until=AnyOf(self,self.trucks_events.values()))
        for truck in result.values():
            self.trucks_waiting[truck] = True

            
    def truck_wait(self, truck, n=1):
        yield self.timeout(n, value=truck)

    # Process: Gas Tank Refuel by a Tank Truck
    def tank_truck(self, fuel_pump, truck):
        """Arrives at the gas station after a certain delay and refuels it."""
        #self.free_truck = 0
        yield self.timeout(BASE_CONFIG["TANK_TRUCK_TIME"])
        dprint('Tank truck arriving at time %d' % self.now)
        ammount = fuel_pump.capacity - fuel_pump.level
        dprint('Tank truck refuelling %.1f liters.' % ammount)
        self.actual_revenue -= BASE_CONFIG["TRUCK_COST"]
        if ammount > 0:
            yield fuel_pump.put(ammount)
        # Time to go back to the base
        result = yield self.timeout(BASE_CONFIG["TANK_TRUCK_TIME"], value=truck)
        return result
        

    # Process: Car Refuel
    def car(self, name, gas_station, fuel_pump):
        """A car arrives at the gas station for refueling.
        It requests one of the gas station's fuel pumps and tries to get the
        desired amount of gas from it. If the stations reservoir is
        depleted, the car has to wait for the tank truck to arrive.
        """
        avg, minim, maxim, scale = BASE_CONFIG["CAR_TANK_LEVEL"], 0, BASE_CONFIG["CAR_TANK_SIZE"], 10
        fuel_tank_level = truncnorm.rvs((minim - avg) / scale, (maxim - avg) / scale, avg, scale)
        # fuel_tank_level = random.triangular(0,BASE_CONFIG["CAR_TANK_LEVEL"],BASE_CONFIG["CAR_TANK_SIZE"])
        dprint('%s arriving at gas station at %.1f' % (name, self.now))
        with gas_station.request() as req:
            start = self.now
            # Request one of the gas pumps or leave if it take to long
            result = yield req | self.timeout(random.expovariate(1 / BASE_CONFIG["DRIVER_PATIENCE"]))

            if req in result:

                # Get the required amount of fuel
                liters_required = BASE_CONFIG["CAR_TANK_SIZE"] - fuel_tank_level
                yield fuel_pump.get(liters_required)

                # Pay the fuel
                self.actual_revenue += liters_required * BASE_CONFIG["MARGIN_PER_LITRE"]

                # The "actual" refueling process takes some time
                yield self.timeout(liters_required / BASE_CONFIG["REFUELING_SPEED"])

                dprint('%s finished refueling in %.1f minutes.' % (name, self.now - start))
            else:
                dprint("{} waited {} minutes and left without refueling".format(name, self.now - start))

    # Generator: Generate car arrivals at the gas station
    def car_generator(self, gas_station: simpy.Resource, fuel_pump: simpy.Container):
        """Generate new cars that arrive at the gas station."""
        for i in itertools.count():
            hour = (self.now // 60) % 24
            yield self.timeout(int(random.expovariate(1 / BASE_CONFIG["CAR_INTERVAL"][hour])))
            self.process(self.car('Car %d' % i, gas_station, fuel_pump))


# Mandatory
class SimBaseline:
    def __init__(self, baseline_config: dict = None, sim_config: dict = None):
        self.sim = None
        self.baseline_config = {"level": 25}
        if baseline_config is not None:
            self.baseline_config.update(baseline_config)
        self.sim_config = BASE_CONFIG.copy()
        if sim_config is not None:
            self.sim_config.update(sim_config)

    def get_action(self, obs):
        return obs[0] < self.baseline_config["level"]

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


def print_stats(sim: SimModel):
    pass


if __name__ == "__main__":
    n = 5
    total = 0
    for _ in range(n):
        baseline = SimBaseline(baseline_config={"level": 25})
        reward = baseline.run()
        total += reward
        print_stats(baseline.sim)
    print("### Average Rewards", total / n)
