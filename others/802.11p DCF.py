import random


class Vehicle:
    def __init__(self, id):
        self.id = id
        self.Wmin = 32  # 初始竞争窗口的最小值
        self.W = self.Wmin  # 当前竞争窗口的大小
        self.backoff_counter = 0  # 退避计数器
        self.retry_limit = 3  # 重传次数限制
        self.retry_count = 0  # 当前重传次数

    def send_subtask(self, subtask, channel_busy):
        if channel_busy:
            return False  # 信道忙碌，无法发送子任务

        if self.backoff_counter > 0:
            self.backoff_counter -= 1
            return False  # 退避计数器大于零，继续等待

        # 发送子任务
        print(f"Vehicle {self.id} sends subtask {subtask}")
        return True

    def increase_backoff_counter(self):
        self.backoff_counter = random.randint(0, self.W - 1)

    def increase_window_size(self):
        self.W *= 2

    def reset_window_size(self):
        self.W = self.Wmin

    def increase_retry_count(self):
        self.retry_count += 1

    def reset_retry_count(self):
        self.retry_count = 0


class Channel:
    def __init__(self):
        self.busy = False

    def set_busy(self):
        self.busy = True

    def set_idle(self):
        self.busy = False


def dcf_mechanism(vehicles):
    channel = Channel()
    successful_transmissions = 0

    for vehicle in vehicles:
        subtasks = ["subtask1", "subtask2", "subtask3"]

        for subtask in subtasks:
            while True:
                if vehicle.send_subtask(subtask, channel.busy):
                    successful_transmissions += 1
                    break

                if channel.busy:
                    continue

                if vehicle.backoff_counter == 0:
                    vehicle.increase_backoff_counter()

                channel.set_busy()

        channel.set_idle()

    return successful_transmissions


# 创建三辆车辆并执行DCF机制
vehicles = [Vehicle("Va"), Vehicle("Vb"), Vehicle("Vc")]
success_count = dcf_mechanism(vehicles)
print(f"Successful transmissions: {success_count}")
