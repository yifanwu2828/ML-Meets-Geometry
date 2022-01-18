import math 

import numpy as np
import scipy.linalg as LA
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# Note Matplotlib is only suitable for simple 3D visualization.
# For later problems, you should not use Matplotlib to do the plotting
from icecream import ic

np.set_printoptions(suppress=True)

sqrt2 = math.sqrt(2)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def skew2vec(x_hat: np.ndarray) -> np.ndarray:
    """
    hat map so3 to vector
    :param x_hat:
    :return (3,) vector
    """
    assert x_hat.shape == (3, 3), "x_hat must be a 3x3 matrix"
    x1, x2, x3 = x_hat[2, 1], x_hat[0, 2], x_hat[1, 0]
    return np.array([x1, x2, x3])

def vec2skew(x: np.ndarray) -> np.ndarray:
    """
    vector to hat map so3
     [[0, -x3, x2],
      [x3, 0, -x1],
      [-x2, x1, 0]]
    :param x: vector
    :type x: numpy array vector
    :return: skew symmetric matrix 
    """
    assert x.size == 3, "x must be a vector with 3 elements"
    x_hat = np.zeros((3, 3), dtype=np.float64)
    x_hat[0, 1] = -x[2]
    x_hat[0, 2] =  x[1]
    x_hat[1, 0] =  x[2]
    x_hat[1, 2] = -x[0]
    x_hat[2, 0] = -x[1]
    x_hat[2, 1] =  x[0]
    
    return x_hat

def qs_qv_fromQuat(q: np.ndarray):
    """Extract qs and qv from quaternion"""
    assert q.size == 4, "q must be a quaternion with 4 elements"
    qs = q[0]
    qv = q[1:]
    return qs, qv

def rotMat_fromQuat(q: np.ndarray):
    assert q.size == 4, "q must be a quaternion with 4 elements"
    assert abs(LA.norm(q) - 1) < 1e-6, "q must be a unit quaternion"
    I = np.eye(3)
    qs, qv = qs_qv_fromQuat(q)
    qv = qv.reshape(3, 1)
    
    qv_skew = vec2skew(qv)
    a = qs * I + qv_skew
    b = qs * I - qv_skew
    
    Eq = np.hstack([-qv, a])
    Gq = np.hstack([-qv, b])
    return Eq @ Gq.T

def expCoord_fromQuat(q: np.ndarray):
    """
    theta: rotational angle
    omega_hat: unit axis
    """
    qs, qv = qs_qv_fromQuat(q)
    
    # A more numerically stable expression of the rotation angle
    # theta = 2 * np.arccos(qs)
    theta = 2 * np.arctan2(LA.norm(qv), qs)
    
    if theta == 0:
        return theta, np.zeros_like(qv)
    omega =  qv / np.sin(theta/2)
    return theta, omega
    

def exp_map(x_hat: np.ndarray):
    """
    exp map of so3
    :return: exp map of so3
    """
    theta = LA.norm(skew2vec(x_hat))
    ans = np.eye(3) + np.sin(theta)/ theta * x_hat  +  (1 - np.cos(theta))/theta**2 * (x_hat @ x_hat)
    assert np.allclose(ans, LA.expm(x_hat)), "exp map of so3 is not correct"
    return ans 

def Q1():
    # (1) ==============================================
    p = np.array([1/sqrt2, 1/sqrt2, 0, 0])
    q = np.array([1/sqrt2, 0, 1/sqrt2, 0]) 
    
    r = (q+p)/2 
    print(f"norm of r: {LA.norm(r)}") 
    
    # unit quaternion
    unit_r = r / np.linalg.norm(r)
    norm_r = LA.norm(unit_r)
    assert abs(norm_r - 1) < 1e-6, "r is not a unit quaternion"
    
    M_r = rotMat_fromQuat(unit_r)
    ic(M_r)
    
    theta, omega_hat = expCoord_fromQuat(unit_r)
    ic(theta, np.rad2deg(theta))
    ic(omega_hat)

def Q2():
    p = np.array([1/sqrt2, 1/sqrt2, 0, 0])
    q = np.array([1/sqrt2, 0, 1/sqrt2, 0]) 
    theta_p, omega_hat_p = expCoord_fromQuat(p)
    theta_q, omega_hat_q = expCoord_fromQuat(q)
    ic(theta_p, np.rad2deg(theta_p), omega_hat_p)
    ic(theta_q, np.rad2deg(theta_q), omega_hat_q)
    
def Q3_a():
    p = np.array([1/sqrt2, 1/sqrt2, 0, 0])
    q = np.array([1/sqrt2, 0, 1/sqrt2, 0]) 
    
    p_qs, p_qv = qs_qv_fromQuat(p)
    q_qs, q_qv = qs_qv_fromQuat(q)
    
    p_qv_hat = vec2skew(p_qv) 
    q_qv_hat = vec2skew(q_qv)
    ic(p_qv_hat, q_qv_hat)
    
    R_p = rotMat_fromQuat(p)
    R_q = rotMat_fromQuat(q)
    ic(R_p, R_q)

def Q3_b():
    p = np.array([1/sqrt2, 1/sqrt2, 0, 0])
    q = np.array([1/sqrt2, 0, 1/sqrt2, 0]) 
    
    p_qs, p_qv = qs_qv_fromQuat(p)
    q_qs, q_qv = qs_qv_fromQuat(q)
    
    p_skew = vec2skew(p_qv) 
    q_skew = vec2skew(q_qv) 
    
    a = exp_map(p_skew)
    b = exp_map(q_skew)
    
    c = a@b
    d = exp_map(p_skew + q_skew)
    ic(c, d)

def Q4_a():
    p = np.array([1/sqrt2, 1/sqrt2, 0, 0])
    q = np.array([1/sqrt2, 0, 1/sqrt2, 0]) 
    
    p_prime = -p
    q_prime = -q
    
    '''
    Two unit quaternions correspond to the same rotation
    R(q) = R(-q)
    '''    
        
    theta_p_prime, omega_hat_p_prime = expCoord_fromQuat(p_prime)
    theta_q_prime, omega_hat_q_prime = expCoord_fromQuat(q_prime)
    
    
    ic(theta_p_prime, np.rad2deg(theta_p_prime), omega_hat_p_prime)
    ic(theta_q_prime, np.rad2deg(theta_q_prime), omega_hat_q_prime)

    

def Q4_b():
    pass

def show_points(points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d([-2, 2])
    ax.set_ylim3d([-2, 2])
    ax.set_zlim3d([0, 4])
    ax.scatter(points[0], points[2], points[1])
    
def compare_points(points1, points2):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d([-2, 2])
    ax.set_ylim3d([-2, 2])
    ax.set_zlim3d([0, 4])
    ax.scatter(points1[0], points1[2], points1[1])
    ax.scatter(points2[0], points2[2], points2[1])  

def newtonsMethod(f, x0, tol=1.48e-08, max_iter=100):
    x = x0
    for itr in range(max_iter):
        df = misc.derivative(f, x, dx=1e-6)
        ic(df.shape)
        x1 = x - f(x)/df
        if abs(x1 - x) < tol:
            print(f"the root was found to be at {x1} after {itr} iterations")
            return x1
        x = x1
    print("Maximum number of iterations exceeded")
    return x


def hw0_solve(A, b, eps=1):
    """
    To find x
    x = h(\lambda)) 
    """
    I = np.eye(A.shape[1])
    h = lambda l: LA.inv(A.T @ A + 2 * l *I) @ A.T @ b
    f = lambda l: h(l).T @ h(l) - eps
    
    l0=0
    l = newtonsMethod(f, l0, eps, max_iter=100)
    
    return h(l)



if __name__ == '__main__':
    npz = np.load('data/HW1_P1.npz')
    X = npz['X']
    Y = npz['Y']
    ic(X.shape, Y.shape)
    compare_points(X, Y)  # noisy teapotsand

    # # implemntation of Q3
    # R1 = np.eye(3)
    # # solve this problem here, and store your final results in R1
    # for _ in range(1):
    #     hw0_solve(R1@X, Y, eps=1)
    # # Testing code, you should see the points of the 2 teapots roughly overlap
    # compare_points(R1@X, Y)
    # # plt.show()
    # print(R1.T@R1)
 
    
    
    
    