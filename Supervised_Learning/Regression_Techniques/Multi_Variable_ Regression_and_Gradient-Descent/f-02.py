import numpy as np
data = np.loadtxt("data.csv", delimiter=",")
data.shape


def gd(points, learning_rate, num_iterations):
    m = 0
    c = 0
    for i in range(num_iterations):
        m, c = step_gradient(points, learning_rate, m , c)
        print(i, " Cost: ", cost(points, m, c))
    return m, c


def step_gradient(points, learning_rate, m , c):
    m_slope = 0
    c_slope = 0
    M = len(points)
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        m_slope += (-2/M)* (y - m * x - c)*x
        c_slope += (-2/M)* (y - m * x - c)
    new_m = m - learning_rate*m_slope
    new_c = c - learning_rate*c_slope
    return new_m, new_c


def cost(points, m, c):
    total_cost = 0
    M = len(points)
    print((m*points[:,0]).shape)
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (1/M)*((y - m*x - c)**2)
        # print('1:', y - m*x - c)
        # print('2:',((y - m*x - c)**2))
    return total_cost


def run():
    data = np.loadtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    num_iterations = 1
    m, c = gd(data, learning_rate, num_iterations)
    print(m, c)


run()
