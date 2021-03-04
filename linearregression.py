import numpy as np
import matplotlib.pyplot as plt


f = open('data.csv', 'r')

Ls = f.readlines()
n = len(Ls[0].split(','))-1
m = len(Ls)
X = np.zeros((m, n+1))
y = np.zeros(m)

for i,line in enumerate(Ls):
    L = line.strip().split(',')
    L = list(map(lambda s:float(s), L))
    X[i,0] = 1
    X[i,1:] = L[0:n]
    y[i] = L[-1]

n = n+1
XT = X.transpose()
XTX = XT.dot(X)
XTX_1 = np.linalg.inv(XTX)

def h_theta(x, theta):
    return x.dot(theta)

def grad(theta, lamb=0.01):
    return XT.dot(X.dot(theta) - y) + lamb*theta
    m, n = X.shape
    s = np.zeros(n)
    for i in range(m):
        s += (h_theta(X[i], theta) - y[i]) * X[i]
    s = s / m
    return s

def hessian(theta):
    return XTX

def J(theta, lamb = 0.01):
    return 1.0/(2*m) * np.linalg.norm(h_theta(X, theta) - y) ** 2 + \
        lamb/2 * theta.transpose().dot(theta)

def get_search_direction(theta, type='gradient'):
    second_order = 1
    lamba_square = 0
    if type == 'newton':
        grad_theta = grad(theta)
        inv_Hx = XTX_1
        delta_theta = -1 * inv_Hx.dot(grad_theta)
        lamba_square = np.sqrt(grad_theta.dot(inv_Hx).dot(grad_theta)) 
    else:
        delta_theta = -1 * grad(theta)
    return delta_theta, lamba_square

def get_step(f, theta, grad_theta, delta_theta, type='backtracing'):
    if type == 'backtracing':
        t = 1
        alpha = 0.8
        beta = 0.99
        while J(theta + t*delta_theta) > \
                J(theta) + alpha * t * grad_theta.dot(delta_theta):
            t = beta * t
    else:
        t = 0.000002
    return t

def stop_criteria(delta_theta, lambda_square, method, epsilon):
    if method == 'newton':
        return lambda_square / 2.0 <= epsilon
    else:
        return np.linalg.norm(delta_theta) <= epsilon


wtf = np.linalg.inv(XT.dot(X))
theta_analy = wtf.dot(XT).dot(y)
cost = J(theta_analy)
print(cost)

method = 'newton'
epsilon = 0.05
theta = np.random.randn(n)

while 1:
    grad_theta = grad(theta, lamb=0)
    delta_theta, lambda_square = get_search_direction(theta, method)
    if stop_criteria(delta_theta, lambda_square, method, epsilon):
        theta_star = theta
        break
    t = get_step(f, theta, grad_theta, delta_theta, type='backtracin')
    theta = theta + t * delta_theta
    cost = J(theta,lamb=0)
    print(cost)

plt.show()