import numpy as np


def initial_state(K):
    s = []
    event = ['A', 'D1', 'D2', 'D3', 'F+1', 'F-1']
    row = 0

    for M in range(1, K + 1):
        for i in range(K + 1):
            for j in range(int(np.ceil(K / 2)) + 1):
                for k in range(int(np.ceil(K / 3)) + 1):
                    for e in range(len(event)):
                        if 1 * i + 2 * j + 3 * k <= M:
                            state = [M, [i, j, k], ' ', 0]
                            if state[1][0] == 0 and event[e] == 'D1':
                                continue
                            if state[1][1] == 0 and event[e] == 'D2':
                                continue
                            if state[1][2] == 0 and event[e] == 'D3':
                                continue
                            if state[0] <= 2 and event[e] == 'F-1':
                                continue
                            if state[0] == K and event[e] == 'F+1':
                                continue
                            state[3] = event[e]
                            state[4] = state[0] - (1 * i + 2 * j + 3 * k)
                            s.append(state)
                            row += 1
    return s


s = initial_state(12)
print(s)
