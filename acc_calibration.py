import numpy as np
import serial

com = serial.Serial('COM5', 115200)

# Θ =    [Δψa,   Δθa,     ΔΦa,  Sax, Say, Saz, bax, bay, baz]
#Theta = [psi_a, theta_a, phi_a,Sax, Say, Saz, bax, bay, baz] 
#def Ha(Theta, a):
#    psi_a, theta_a, phi_a,Sax, Say, Saz, bax, bay, baz = Theta
#    TK = np.array([[Sax, psi_a, -theta_a],
#                   [-psi_a, Say, phi_a],
#                   [theta_a, -phi_a, Saz]])
#    ba = np.array([bax, bay, baz])
#    return TK.dot(a + ba)

m = 1000    #样本容量
n = 3       #ax, ay, az测量值
g = 1
X = []
while True:
    L = com.readline().decode().strip().split(',')
    if len(L) != 6:
        continue
    L = list(map(lambda s:float(s), L))
    X.append(L[0:3])
    if len(X) == m:
        break

X = np.array(X)

Theta = np.random.randn(9)

def Ha(Theta, A):
    #A: m*n (其中m是容量, n=3)
    psi_a, theta_a, phi_a,Sax, Say, Saz, bax, bay, baz = Theta
    ba = np.array([bax, bay, baz])
    A = A + ba
    TK = np.array([[Sax, psi_a, -theta_a],
                   [-psi_a, Say, phi_a],
                   [theta_a, -phi_a, Saz]])
    result = TK.dot(A.T)
    result = result.T
    return result

def J(Theta):
    HA = Ha(Theta, X)
    HA_norm = np.linalg.norm(HA, axis=1)
    HA_norm = HA_norm - g
    J_theta = 1./2 * HA_norm.T.dot(HA_norm)
    return J_theta

def cached_operator(Theta):
    HA = Ha(Theta, X)
    HA_norm = np.linalg.norm(HA, axis=1) - g
    return HA, HA_norm

def grad_H(Theta, a):
    ax,ay,az = a
    d_psi, d_theta, d_phi, Sx, Sy, Sz, bx, by, bz = Theta
    grad_H_to_theta = np.array([
            [ay+by, -(ax+bx), 0],     #δha/δpsi_a
            [-(az+bz), 0, ax+bx],     #δha/δtheta_a
            [0, az+bz, -(ay+by)],     #δha/δphi_a
            [ax+bx, 0, 0],     #δha/δSa
            [0, ay+by, 0],     #δha/δSb
            [0, 0, az+bz],     #δha/δSc
            [Sx, -d_psi, d_theta],     #δha/δbax
            [d_psi, Sy, -d_phi],     #δha/δbay
            [-d_theta, d_phi, Sz]     #δha/δbaz        
        ])
    return grad_H_to_theta

def grad(Theta):#, HA, HA_norm):
    #factor = (HA_norm-g) / (HA_norm)
    #顺序: #Theta = [psi_a, theta_a, phi_a,Sax, Say, Saz, bax, bay, baz]
    # 直接矩阵相乘有点麻烦，先写成逐项求导
    sum = 0
    for i in range(m):
        ha_i = Ha(Theta, X[i])
        ha_norm = np.linalg.norm(ha_i)
        ax,ay,az = X[i]
        d_psi, d_theta, d_phi, Sx, Sy, Sz, bx, by, bz = Theta
        grad_H_to_theta = np.array([
            [ay+by, -(ax+bx), 0],     #δha/δpsi_a
            [-(az+bz), 0, ax+bx],     #δha/δtheta_a
            [0, az+bz, -(ay+by)],     #δha/δphi_a
            [ax+bx, 0, 0],     #δha/δSa
            [0, ay+by, 0],     #δha/δSb
            [0, 0, az+bz],     #δha/δSc
            [Sx, -d_psi, d_theta],     #δha/δbax
            [d_psi, Sy, -d_phi],     #δha/δbay
            [-d_theta, d_phi, Sz]     #δha/δbaz
            ]).T
        der = ((ha_norm - g) / ha_norm) * ha_i.dot(grad_H_to_theta)
        sum += der
    return sum

def Hessian(Theta):
    sum = 0
    for i in range(m):
        ha_i = Ha(Theta, X[i])
        ha_norm = np.linalg.norm(ha_i)
        ax, ay, az = X[i]
        grad_H_to_theta = grad_H(Theta, X[i])

        gH = grad_H_to_theta.dot(ha_i).reshape(9, 1)
        hG = ha_i.T.dot(grad_H_to_theta.T).reshape(1,9)
        part1 = (g/(ha_norm**3)) * gH.dot(hG)
        part2 = grad_H_to_theta.dot(grad_H_to_theta.T)

        #part3是一个ha对theta的二阶导:
        #顺序: #Theta = [psi_a, theta_a, phi_a,Sax, Say, Saz, bax, bay, baz]
        H_ = np.array([
            [0, 0, 0, 0, 0, 0, -ha_i[1], ha_i[0], 0], #[ay+by, -(ax+bx), 0]
            [0, 0, 0, 0, 0, 0, ha_i[2], 0, -ha_i[0]], #[-(az+bz), 0, ax+bx]
            [0, 0, 0, 0, 0, 0, 0, -ha_i[2], ha_i[1]], #[0, az+bz, -(ay+by)]
            [0, 0, 0, 0, 0, 0, ha_i[0], 0, 0], #[ax+bx, 0, 0]
            [0, 0, 0, 0, 0, 0, 0, ha_i[1], 0], #[0, ay+by, 0]
            [0, 0, 0, 0, 0, 0, 0, 0, ha_i[2]], #[0, 0, az+bz]
            [-ha_i[1], ha_i[2], 0, ha_i[0], 0, 0, 0, 0, 0], #[Sx, -d_psi, d_theta]
            [ha_i[0], 0, -ha_i[2], 0, ha_i[1], 0, 0, 0, 0], #[d_psi, Sy, -d_phi]
            [0, -ha_i[0], ha_i[1], 0, 0, ha_i[2], 0, 0, 0], #[-d_theta, d_phi, Sz] 
            ])
        der = part1 + part2 + H_
        sum += der
    return sum

def get_search_direction(Theta, type='gradient'):
    second_order = 1
    lamba_square = 0
    if type == 'newton':
        grad_theta = grad(Theta)
        H_theta = Hessian(Theta)
        inv_H_theta = np.linalg.inv(H_theta)
        delta_theta = -1 * inv_H_theta.dot(grad_theta)
        lamba_square = np.sqrt(grad_theta.dot(inv_H_theta).dot(grad_theta)) 
    else:
        delta_theta = -1 * grad(Theta)
    return delta_theta, lamba_square

def get_step(method):
    if method == 'newton':
        t = 0.1
    else:
        t = 0.0001
    return t

def stop_criteria(delta_theta, lambda_square, method, epsilon):
    if method == 'newton':
        return lambda_square / 2.0 <= epsilon
    else:
        return np.linalg.norm(delta_theta) <= epsilon

method = 'newton'
epsilon = 0.005
#Theta = np.random.randn(9)
Theta = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

while 0:
    cost = J(Theta)
    print(cost)

while 1:
    delta_Theta, lambda_square = get_search_direction(Theta, method)
    if stop_criteria(delta_Theta, lambda_square, method, epsilon):
        Theta_star = Theta
        break
    t = get_step(method)
    Theta = Theta + t * delta_Theta
    cost = J(Theta)
    print(cost)

print("优化完成，以下是参数的值:")
d_psi, d_theta, d_phi, Sx, Sy, Sz, bx, by, bz = Theta_star
print("Δψa = %f, Δθa = %f, ΔΦa = %f, \
        Sax = %f, Say = %f, Saz = %f, \
        bax = %f, bay = %f, baz = %f" %\
        (d_psi, d_theta, d_phi, Sx, Sy, Sz, bx, by, bz))

#读数据，观察实际效果
while True:
    L = com.readline().decode().strip().split(',')
    if len(L) != 6:
        continue
    L = list(map(lambda s:float(s), L))
    a = np.array(L[0:3])
    ha = Ha(Theta_star, a)
    ha_norm = np.linalg.norm(ha)
    #print(ha_norm)
    print("%f\t%f\t%f\n", ha[0], ha[1], ha[2])