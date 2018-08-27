import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def gen_line_data(sample_num = 100):
    x1 = np.linspace(0, 9, sample_num)
    x2 = np.linspace(4, 13, sample_num)
    x = np.concatenate(([x1],[x2]),axis=0).T
    y = np.dot(x, np.array([3, 4]).T)
    return x,y

x,y = gen_line_data()

m,n = x.shape
x1 = np.ones((m,n + 1))
x1 = x[:,:-1]
print(x)
print(x1)