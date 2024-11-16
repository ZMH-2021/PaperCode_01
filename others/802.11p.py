import random


def calculateV2VTransmissionDelay(payloadSizeBytes, cwMin, cwMax, slotTimeUs, sifsTimeUs, eifsTimeUs, bitrateMbps,
                                  currentCw, numberOfRetries):
    """
    计算车辆间单次数据包传输延迟的函数。

    参数：
    payloadSizeBytes - 数据包有效负载大小（以字节为单位）
    cwMin            - 竞争窗口的最小值
    cwMax            - 竞争窗口的最大值
    slotTimeUs       - 时隙时间（微秒）
    sifsTimeUs       - SIFS 时间（微秒）
    eifsTimeUs       - EIFS 或 DIFS 时间（微秒）
    bitrateMbps      - 数据传输速率（Mbps）
    currentCw        - 当前竞争窗口大小（初始值由外部设定，例如从上一次传输后遗留下来）
    numberOfRetries  - 允许的最大重试次数（超过这个次数认为传输失败）

    返回值：
    transmissionDelayUs - 总传输延迟（微秒）
    """

    # MAC层开销，简化处理
    headerOverheadBits = 300  # 头部信息比特数（举例值）
    ackOverheadBits = 192  # ACK帧头部信息比特数（举例值）

    # 转换速率到每微秒传输的比特数
    bitratePerMicros = bitrateMbps * 1e6 / 8

    # 计算数据包总比特数（包括有效负载和开销）
    totalBits = payloadSizeBytes * 8 + headerOverheadBits

    # 计算有效数据包传输时间（微秒）
    dataPacketDurationUs = totalBits / bitratePerMicros

    # 计算ACK帧传输时间（微秒）
    ackDurationUs = ackOverheadBits / bitratePerMicros

    # 初始化传输延迟
    transmissionDelayUs = 0

    # 开始重试传输过程
    for retry in range(numberOfRetries):
        # 回退阶段
        backoffTimeUs = 0
        cw = max(currentCw, cwMin)
        while True:
            backoffSlotCount = random.randint(0, cw - 1)  # 随机选择一个回退时隙
            backoffTimeUs += backoffSlotCount * slotTimeUs

            if random.random() < (1 / cw):  # 模拟是否有冲突发生
                cw = min(cw * 2, cwMax)  # 竞争窗口翻倍
                continue  # 继续回退循环
            else:
                break  # 无冲突则退出回退循环

        # 计算本次尝试的总传输时间
        trialTransmissionTimeUs = backoffTimeUs + sifsTimeUs + dataPacketDurationUs + sifsTimeUs + ackDurationUs

        # 如果此次传输成功，则结束循环
        if retry == 0 or trialTransmissionTimeUs > 0:  # 这里假设成功的条件是至少有一次非零传输时间（实际情况需考虑ACK接收确认）
            transmissionDelayUs = trialTransmissionTimeUs
            break

        # 若本次尝试失败，增加当前竞争窗口大小并继续下次重试
        currentCw = cw

    # 返回总传输延迟（微秒）
    return transmissionDelayUs


if __name__ == '__main__':
    payloadSizeBytes = 100
    cwMin = 32
    cwMax = 64
    slotTimeUs = 9
    sifsTimeUs = 16
    eifsTimeUs = 34
    bitrateMbps = 27
    currentCw = 32
    numberOfRetries = 5

    transmissionDelayUs = calculateV2VTransmissionDelay(payloadSizeBytes, cwMin, cwMax, slotTimeUs, sifsTimeUs,
                                                        eifsTimeUs, bitrateMbps, currentCw, numberOfRetries)
    print("总传输延迟（微秒）：", transmissionDelayUs)
