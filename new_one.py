import random
import numpy as np
import pandas as pd

random.seed(1)

Simulation_time = 600
Simulation_step_time = 1
Simulation_length = 1000
Simulation_road_number = 8
Simulation_lower_speed = 60
Simulation_upper_speed = 180

Maximum_communication_distance = 600
Maximum_VFC_search_distance = 100
Maximum_number_VFC_vehicles = 5

total_Agent_num = 0
total_VFC_num = 0

Applications_dict = {
    'A': {
        'tasks': ['a1', 'a1', 'a1', 'a1'],
        'size': [10, 10, 10, 10],
        'latency': 100,
        'price': 100
    },
    'B': {
        'tasks': ['b1', 'b1', 'b2', 'b3'],
        'size': [15, 15, 20, 25],
        'latency': 150,
        'price': 200
    },
    'C': {
        'tasks': ['c1', 'c2', 'c3', 'c4'],
        'size': [10, 20, 30, 40],
        'latency': 200,
        'price': 300
    }
}


def select_application():
    return random.choice(list(Applications_dict.keys()))


def speed2speeds(min_speed, max_speed, times):
    res = []
    size = (max_speed - min_speed) / (times - 1)
    for i in range(times):
        res.append(min_speed + size * i)
    return res * 2


def generate_car_probability(n=0.36):
    return True if random.uniform(0, 1) < n else False


class Car:
    def __init__(self, carID, speed, road_ID, time=0):
        self.carID = carID
        self.speed = speed
        self.roadID = road_ID
        self.time = time
        self.direction = True if road_ID <= Simulation_road_number / 2 else False
        self.loc = 0 if self.direction else Simulation_length
        self.is_Service_Provider = random.randint(1, 100) < 30
        self.task = select_application()
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
        nparr = np.array(all_cars_status,
                         dtype=[('time', np.int64), ('carID', np.int64), ('speed', np.float64), ('loc', np.float64),
                                ('roadID', np.int64), ('is_Service_Provider', np.dtype), ('task', np.dtype),
                                ('cc', np.int64), ('cp', np.int64)])
        return pd.DataFrame(nparr)


if __name__ == '__main__':
    # 得到模拟结果
    df = Simulation().run()
    # 结果按时间顺序排序
    df = df.sort_values(by='time').reset_index(drop=True)

    print(df.head(50))

    df.to_csv('data.csv', index=True)
