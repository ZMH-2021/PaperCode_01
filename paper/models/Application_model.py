import random

A = {
    'tasks': ['a1', 'a1', 'a1', 'a1'],
    'size': [10, 10, 10, 10],
    'latency': 100,
    'price': 100
}
B = {
    'tasks': ['b1', 'b1', 'b2', 'b3'],
    'size': [15, 15, 20, 25],
    'latency': 150,
    'price': 200
}
C = {
    'tasks': ['c1', 'c2', 'c3', 'c4'],
    'size': [10, 20, 30, 40],
    'latency': 200,
    'price': 300
}


def Applications():
    return random.choice(['A', 'B', 'C'])


def App(which):
    if which == 'A':
        return A
    if which == 'B':
        return B
    if which == 'C':
        return C
    return None


if __name__ == '__main__':
    a = Applications()
    print(a)
    print(App(a))

    a = Applications()
    print(a)
    print(App(a))

    a = Applications()
    print(a)
    print(App(a))

    a = Applications()
    print(a)
    print(App(a))
