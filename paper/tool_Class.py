import random
from Hyperparameters import *
from tool_Function import *
import numpy as np
import pandas as pd
from models.Application_model import Applications

random.seed(1)


class Car:
    def __init__(self, carID, speed, road_ID, time=0):
        self.carID = carID
        self.speed = speed
        self.roadID = road_ID
        self.time = time
        self.direction = True if road_ID <= Simulation_road_number / 2 else False
        self.loc = 0 if self.direction else Simulation_length
        self.is_Service_Provider = random.randint(1, 100) < 30
        self.task = Applications()
        self.concurrency_capability = random.choice([1, 2, 3, 4])
        self.computing_capability = random.choice([10, 20, 30])  # MB/s
        self.status_history = [
            (self.time, self.carID, self.speed, self.loc, self.roadID, self.is_Service_Provider, self.task,
             self.concurrency_capability, self.computing_capability)]

    def update_time_loc(self):
        if self.direction:
            self.loc += self.speed / 3.6 * Simulation_step_time
            # self.loc = Simulation_length if self.loc > Simulation_length else self.loc
        else:
            self.loc -= self.speed / 3.6 * Simulation_step_time
            # self.loc = 0 if self.loc < 0 else self.loc
        self.time += Simulation_step_time
        if 0 <= self.loc <= Simulation_length:
            self.status_history.append(
                (self.time, self.carID, self.speed, self.loc, self.roadID, self.is_Service_Provider, self.task,
                 self.concurrency_capability, self.computing_capability))


class Road:
    def __init__(self, road_id, road_speed):
        self.road_id = road_id
        self.road_speed = road_speed

    def __str__(self):
        return f"road_id:{self.road_id}, road_speed:{self.road_speed}"


class Roads:
    def __init__(self, road_num, min_speed, max_speed):
        self.road_num = road_num
        self.lis = speed2speeds(min_speed, max_speed, road_num // 2)
        self.road_dict = {}
        for i in range(road_num):
            id = Road(i + 1, self.lis[i]).road_id
            road_speed = Road(i + 1, self.lis[i]).road_speed
            self.road_dict[id] = road_speed

    def get_roads(self):
        return self.road_dict


class Simulation:

    def __init__(self):
        self.roads = Roads(Simulation_road_number, Simulation_lower_speed, Simulation_upper_speed)

    def generate_update_cars(self, current_time, cars):
        if generate_car_probability():
            carID = len(cars) + 1
            road_ID = random.randint(1, Simulation_road_number)
            speed = self.roads.road_dict[road_ID]
            car = Car(carID=carID, time=current_time, speed=speed, road_ID=road_ID)
            cars.append(car)

        for c in cars:
            c.update_time_loc()

    def run(self):
        print("----------------Simulation start----------------")
        current_time = 0
        cars = []
        all_cars_status = []
        while current_time <= Simulation_time:
            self.generate_update_cars(current_time, cars)
            current_time += Simulation_step_time
        all_cars_status.extend([s for c in cars for s in c.status_history])

        print("----------------Simulation end-----------------")
        # return np.array(all_cars_status, dtype=([float, int, float, float, int]))
        nparr = np.array(all_cars_status,
                         dtype=[('time', np.int64), ('carID', np.int64), ('speed', np.float64), ('loc', np.float64),
                                ('roadID', np.int64), ('is_Service_Provider', np.dtype), ('task', np.dtype), ('cc', np.int64), ('cp', np.int64)])
        return pd.DataFrame(nparr)
