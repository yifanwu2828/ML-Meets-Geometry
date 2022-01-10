# Newton's Method
import numpy as np
from numpy import linalg as LA
from scipy import misc
from icecream import ic
'''
step
    (1) Pick an initial guess fo the root (x0)
    (2) 1st order taylor series: f(x) = f(x0) + 1/(1!) f'(x0)(x-x0) where (x-x0) is the step size
    (3) Find the value of x1: tangent of the derivative hits the axis
        0 = f(x0) + f'(x0)*x - f'(x0)*x0
        x = x0 - f(x0)/f'(x0)
    (4) Find f(x1) and repear step 3-4 unitl reach the acceptable level of error
        error = x_{approx} - x_{actual}
'''

def newtonsMethod(f, x0, tol=1.48e-08, max_iter=100):
    x = x0
    for itr in range(max_iter):
        df = misc.derivative(f, x, dx=1e-6)
        x1 = x - f(x)/df
        if abs(x1 - x) < tol:
            print(f"the root was found to be at {x1} after {itr} iterations")
            return x1
        x = x1
    print("Maximum number of iterations exceeded")
    return x


def solve(A, b, eps):
    """
    To find x
    x = h(\lambda)) 
    """
    I = np.eye(A.shape[1])
    h = lambda l: LA.inv(A.T @ A + 2 * l *I) @ A.T @ b
    f = lambda l: h(l).T @ h(l) - eps
    
    l0=0
    l = newtonsMethod(f, l0, eps, 50)
    
    return h(l)


if __name__ == '__main__':
    npz = np.load('./data/HW0_P1.npz')
    A = npz['A']
    b = npz['b']
    eps = npz['eps']
    ic(A.shape, b.shape, eps)
    
    x= solve(A, b, eps)
    print('x norm square', x.T@x)  # x@x should be close to or less then eps
    print('optimal value', ((A@x - b)**2).sum())
    
    
    
    
    
    