import math

def f1(tau, tf = 10, a = 1):
    return a * tau - (tf / (2 * math.pi * a)) * (a - 1) * math.sin((2 * math.pi * a / tf) * tau)


def df1(tau, tf = 10, a = 1):
    return a - (a - 1) * math.cos((2 * math.pi * a / tf) * tau)


def f2(tau, tf=10, a=1):
    return (2 * (a**2 - a**3) / tf**2) * tau**3 + (3 * (a**2 - a) / tf) * tau**2 + tau


def df2(tau, tf=10, a=1):
    return (6 * (a**2 - a**3) / tf**2) * tau**2 + (6 * (a**2 - a) / tf) * tau + 1
