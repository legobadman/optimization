import matplotlib.pyplot as plt
import numpy as np
from Arrow3D import Arrow3D


def f_(x,y):
    return 1./3 * x**2 + 1./2 * y**2

def f(x):
    return 1./3 * x[0]**2 + 1./2 * x[1]**2

def grad(x):
    return np.array([2./3 * x[0], x[1]])

def get_search_direction(x, f, grad, type='gradient'):
    second_order = 1
    if type == 'newton':
        pass
    delta_x = -1 * second_order * grad(x)
    return delta_x

def get_step(f, x, grad_x, delta_x, type='backtracing'):
    if type == 'backtracing':
        t = 1
        alpha = 0.25
        beta = 0.45
        while f(x + t*delta_x) > f(x) + alpha * t * grad_x.dot(delta_x):
            t = beta * t
    else:
        t = 0.1
    return t



show_3d = False

if show_3d:
    x, y = np.mgrid[-2:2:20j,-2:2:20j]
    z = f_(x, y)
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca(projection='3d')
    #ax.plot_surface(x,y,z,cmap=plt.get_cmap('rainbow'))
    ax.plot_wireframe(x,y,z)
else:
    x, y = np.mgrid[-2:2:20j,-2:2:20j]
    z = f_(x, y)
    plt.contourf(x,y,z, levels=50, cmap='jet')

def annote_descent(x, t, delta_x, f):
    x_old = x
    x_new = x_old + t * delta_x
    if show_3d:
        a = Arrow3D([x_old[0], x_new[0]], [x_old[1], x_new[1]], 
                [f(x_old), f(x_new)], mutation_scale=5, 
                lw=3, arrowstyle="-|>", color="black")
        ax.add_artist(a)
    else:
        plt.scatter(x_new[0], x_new[1], marker='x')
        plt.annotate('', xy=(x_new[0], x_new[1]), xytext=(x_old[0],x_old[1]),
                     arrowprops=dict(arrowstyle="->",color = 'black',
                                     connectionstyle='arc3'))

epslion = 0.05
x_start = np.array([-2,-2]) #这个x是二元坐标(x,y)
x = x_start

while True:
    grad_x = grad(x)
    delta_x = get_search_direction(x, f, grad)
    if np.linalg.norm(delta_x) <= epslion:
        x_star = x
        break
    t = get_step(f, x, grad_x, delta_x, type='backtracing')
    annote_descent(x, t, delta_x, f)
    x = x + t * delta_x


plt.show()
