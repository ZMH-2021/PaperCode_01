import numpy as np
from typing import List


class Road:
    def __init__(self, num_lanes: int, lane_width: float, road_length: float):
        self.num_lanes = num_lanes
        self.lanes = [Lane(lane_width, road_length) for _ in range(num_lanes)]

    def add_vehicle_to_lane(self, lane_index: int, position: float, vehicle: 'Vehicle'):
        """
        将车辆添加到指定车道的位置
        """
        if 0 <= lane_index < self.num_lanes:
            self.lanes[lane_index].add_vehicle(position, vehicle)

    def get_vehicle_positions(self, lane_index: int) -> List[float]:
        """
        获取指定车道所有车辆的位置列表
        """
        return [pos for pos, _ in self.lanes[lane_index].positions]



class Lane:
    def __init__(self, lane_max_speed, lane_number, vehicle_length=6, lane_width=3, lane_length=200):
        self.lane_width = lane_width  # 车道宽度
        self.lane_length = lane_length  # 车道长度
        self.lane_max_speed = lane_max_speed  # 车道最高速度
        self.lane_number = lane_number  # 车道编号（唯一）
        self.vehicle_length = vehicle_length
        self.lane_space = []  # 车道上的车辆分布

    def add_vehicle(self, ):


    def remove_vehicle(self, ):






class Vehicle:
    def __init__(self, id, width, length, speed, ru_status):
        self.id = id              #汽车ID唯一
        self.width = width
        self.length = length
        self.speed = speed
        self.ru_status = ru_status  # 计算资源状态


if __name__ == '__main__':
    # 示例创建道路对象
    # road = Road(num_lanes=2, lane_width=3, road_length=200)
    # vehicle1 = Vehicle(id=1, width=3, length=6, speed=50, ru_status=0.8)  # 假设速度单位为km/h，RU状态为0-1之间
    # road.add_vehicle_to_lane(0, 50, vehicle1)  # 在第一条车道上添加一辆车在50米位置

    # 根据实际需求，可以继续添加其他车辆和实现更多功能，如计算车距、判断能否建立VFC平台等
