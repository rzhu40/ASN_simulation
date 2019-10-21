import numpy as np
import matplotlib.pyplot as plt

def mackey_glass_eq(x_t, x_t_minus_tau, a, b):
    x_dot = -b*x_t + a*x_t_minus_tau/(1 + x_t_minus_tau**10)
    return x_dot

def mackey_glass_rk4(x_t, x_t_minus_tau, dt, a, b):
    k1 = dt * mackey_glass_eq(x_t, x_t_minus_tau, a, b)
    k2 = dt * mackey_glass_eq(x_t+0.5*k1, x_t_minus_tau, a, b)
    k3 = dt * mackey_glass_eq(x_t+0.5*k2, x_t_minus_tau, a, b)
    k4 = dt * mackey_glass_eq(x_t+k3, x_t_minus_tau, a, b)
    x_t_plus_dt = x_t + k1/6 + k2/3 + k3/3 + k4/6
    return x_t_plus_dt

def mkg_generator(length = 10000, dt = 0.1, tau = 17,
                a = 0.2, b = 0.1, x0 = 1.2):
    time = 0
    index = 0
    history_length = np.floor(tau/dt).astype(int)
    x_history = np.zeros(history_length)
    x_t = x0

    X = np.zeros(length)
    T = np.zeros(length)

    for i in range(len(X)):
        X[i] = x_t
        
        if tau == 0:
            x_t_minus_tau = 0.0
        else:
            x_t_minus_tau = x_history[index-1]
            
        x_t_plus_dt = mackey_glass_rk4(x_t, x_t_minus_tau, dt, a, b)
        
        if tau != 0:
            x_history[index-1] = x_t_plus_dt
            index = index % history_length + 1
            
        time = time + dt
        T[i] = time
        x_t = x_t_plus_dt
    return X