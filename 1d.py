import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2

def grad(x):
    return 2*x

def hessian(x):
    return 2

x_left = -2
x_right = 2
n_sample = 100

X = np.arange(x_left, x_right + 0.1, 0.1)
y = np.array([f(x) for x in X])

x_start = 2
epslion = 0.05

plt.plot(X, y)

def get_search_direction(x, f, grad, type='gradient'):
    second_order = (1.0/hessian(x)) if type == 'newton' else 1
    delta_x = -1 * second_order * grad(x)
    return delta_x

def get_step(f, x, grad_x, delta_x):
    t = 1
    alpha = 0.25
    beta = 0.45
    while f(x + t*delta_x) > f(x) + alpha * t * grad_x * delta_x:
        t = beta * t
    return t

def annote_descent(x, t, delta_x, f):
    x_old = x
    x_new = x_old + t * delta_x
    plt.scatter(x_new, f(x_new), marker='x')
    plt.annotate('', xy=(x_new,f(x_new)), xytext=(x_old,f(x_old)),
                 arrowprops=dict(arrowstyle="->",color = 'black',
                                 connectionstyle='arc3'))


#gradient descent
x = x_start
x_star = 0
while True:
    grad_x = grad(x)
    delta_x = get_search_direction(x, f, grad, 'newton')
    if np.linalg.norm(delta_x) <= epslion:
        x_star = x
        break
    t = get_step(f, x, grad_x, delta_x)
    annote_descent(x, t, delta_x, f)
    x = x + t * delta_x


plt.show()