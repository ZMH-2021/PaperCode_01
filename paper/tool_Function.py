import random


def generate_car_probability(n=0.36):
    return True if random.uniform(0, 1) < n else False


def car_request_probability(n=0):
    return True if random.uniform(0, 1) < n else False


def car_request_task_num(is_request):
    return random.choice([1, 2, 3]) if is_request else 0


def is_Agent_probability(n=0):
    return True if random.uniform(0, 1) < n else False


def save_cars_status_to_txt(all_cars_status, filename):
    with open(filename, 'w') as file:
        for car_status in all_cars_status:
            # 将元组转换为字符串，并用逗号分隔，然后写入一行
            line = ','.join(map(str, car_status)) + '\n'
            file.write(line)


import csv


def save_cars_status_to_csv(all_cars_status, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for car_status in all_cars_status:
            # 写入每个元组作为CSV文件的一行
            writer.writerow(car_status)


import openpyxl


def save_cars_status_to_excel(all_cars_status, filename):
    # 创建一个新的Excel工作簿
    wb = openpyxl.Workbook()
    # 选择默认的工作表
    ws = wb.active

    # 将每个元组写入工作表的一行
    for car_status in all_cars_status:
        ws.append(car_status)

    # 保存工作簿到指定的文件名
    wb.save(filename)


def speed2speeds(min_speed, max_speed, times):
    res = []
    size = (max_speed - min_speed) / (times - 1)
    for i in range(times):
        res.append(min_speed + size * i)
    return res * 2
