import matplotlib.pyplot as plt
import numpy as np
from Arrow3D import Arrow3D

#User specific:
def f_(x,y):
    #return 1./3 * x**2 + 1./2 * y**2
    #return x**2 / (x**2 + y**2)
    return x**2 + x*y + y**2

def f(x):
    #return 1./3 * x[0]**2 + 1./2 * x[1]**2
    #return x[0]**2 / (x[0]**2 + x[1]**2)
    return x[0]**2 + x[0]*y[0] + y[0]**2

def grad(X):
    x, y = X[0], X[1]
    return np.array([2./3 * x, y])
    Jx = np.zeros(2)
    Jx[0] = 2*x*y**2 / (x**2+y**2)**2
    Jx[1] = -2*y*x**2 / (x**2+y**2)**2
    return Jx

def hessian(X):
    return np.array([[2./3, 0], [0, 1]])
    x, y = X[0], X[1]
    H = np.zeros((2,2))
    H[0,0] = float(2*y**4 - 6*(x**2)*(y**2)) / (x**2 + y**2)**3
    H[0,1] = float(4*x*y*(x**2 - y**2)) / (x**2 + y**2)**3
    H[1,0] = H[0,1]
    H[1,1] = float(6*(x**2)*(y**2) - 2*x**4) / (x**2 + y**2)**3
    return H
#End


def get_search_direction(x, f, grad, type='gradient'):
    second_order = 1
    lamba_square = 0
    if type == 'newton':
        grad_x = grad(x)
        H_x = hessian(x)
        inv_Hx = np.linalg.inv(H_x)
        second_order = -1 * inv_Hx
        delta_x = second_order.dot(grad_x)
        lamba_square = np.sqrt(grad_x.dot(inv_Hx).dot(grad_x)) 
    else:
        delta_x = -1 * grad(x)
    return delta_x, lamba_square

def stop_criteria(delta_x, lambda_square, method, epsilon):
    if method == 'newton':
        return lambda_square / 2.0 <= epsilon
    else:
        return np.linalg.norm(delta_x) <= epsilon

def get_step(f, x, grad_x, delta_x, type='backtracing'):
    if type == 'backtracing':
        t = 1
        alpha = 0.25
        beta = 0.45
        while f(x + t*delta_x) > f(x) + alpha * t * grad_x.dot(delta_x):
            t = beta * t
    else:
        t = 0.5
    return t



show_3d = True

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
                [f(x_old), f(x_new)], mutation_scale=3, 
                lw=2, arrowstyle="-|>", color="black")
        ax.add_artist(a)
    else:
        plt.scatter(x_new[0], x_new[1], marker='x')
        plt.annotate('', xy=(x_new[0], x_new[1]), xytext=(x_old[0],x_old[1]),
                     arrowprops=dict(arrowstyle="->",color = 'black',
                                     connectionstyle='arc3'))

epsilon = 0.05
x_start = np.array([-2, 0]) #这个x是二元坐标(x,y)
x = x_start
method = 'newton'

while 0:
    grad_x = grad(x)
    delta_x, lambda_square = get_search_direction(x, f, grad, method)
    if stop_criteria(delta_x, lambda_square, method, epsilon):
        x_star = x
        break
    t = get_step(f, x, grad_x, delta_x, type='backtracing')
    annote_descent(x, t, delta_x, f)
    x = x + t * delta_x


plt.show()
