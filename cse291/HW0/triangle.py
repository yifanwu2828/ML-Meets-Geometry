import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

try:
    from icecream import install  # noqa
    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


# vertice are at A=(0,0), B=(0,1), C=(1, 0)
pts = np.array([[0,0], [0,1], [1,0]])

def draw_background(index):
    # DRAW THE TRIANGLE AS BACKGROUND
    p = Polygon(pts, closed=True, facecolor=(1,1,1,0), edgecolor=(0, 0, 0))

    plt.subplot(1, 2, index + 1)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.add_patch(p)
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)


def triangle_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def isInside(vertices, P):
    """
    Check if P is inside the triangle ABC
    vertices A=(x1,y1), B=(x2,y2), C=(x3,y3) 
    """
    x, y = P
    x1, y1, x2, y2, x3, y3 = vertices.flatten()   
    
    # Calculate area of triangle ABC
    A = triangle_area (x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = triangle_area (x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = triangle_area (x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = triangle_area (x1, y1, x2, y2, x, y)
     
    # Check if sum of A1, A2 and A3
    # is same as A
    return A == A1 + A2 + A3


def wrong_sample_in_triangle(n: int, rng, vertices: np.ndarray):
    """
    wrong algorithm for sample n points uniformly inside the triangle
    
    (1) sample alpha, beta, gamma ~ U[0,1], normalize to get
        alpha' = alpha / (alpha + beta + gamma)) 
        beta' = beta / (alpha + beta + gamma))
        gamma' = gamma / (alpha + beta + gamma))
        so that alpha' + beta' + gamma' = 1
    (2) sample P = alpha' * A + beta' * B + gamma' * C inside the triangle ABC
    However, this straight-forward idea is wrong: 
        it does not assure that P is uniformly sampled in 4ABC.
    """
    assert isinstance(n, int)
    assert n > 0
    points = np.empty((n, 2))
    
    for i in range(n):
        samples = rng.uniform(0, 1, size=3)
        normalized_samples =  samples / samples.sum()
        assert np.allclose(normalized_samples.sum(), 1)
        P = (normalized_samples.reshape(-1,1)*vertices).sum(axis=0)
        points[i,:] = P
    return points

def correct_sample_in_triangle(n: int, rng, vertices: np.ndarray):
    """
    correct algorithm for sample n points uniformly inside the triangle
    (1) sample alpha, beta ~ U[0,1]
    (2) P' = A + alpha * (B - A) + beta * (C - A)
    (3) if P' inside triangle ABC, P=P'
        else P = B + C - P'
    """
    assert isinstance(n, int)
    assert n > 0
    A, B, C = vertices
    points = np.empty((n, 2))
    for i in range(n):
        alpha, beta = rng.uniform(0, 1, size=2)
        P_prime = A + alpha * (B - A) + beta * (C - A)
        points[i,:] = P_prime if isInside(vertices, P_prime) else B + C - P_prime
    return points

    
if __name__ == '__main__':
    
    seed = 42
    rng =  np.random.default_rng(seed=42)
    n_samples = 1_000
    
    wrong_samples = wrong_sample_in_triangle(n_samples, rng, pts)

    correct_samples = correct_sample_in_triangle(n_samples, rng, pts)    
        
    draw_background(0)
    # REPLACE THE FOLLOWING LINE USING YOUR DATA (incorrect method)
    plt.scatter(wrong_samples[:, 0], wrong_samples[: ,1], s=3) 

    draw_background(1)
    # REPLACE THE FOLLOWING LINE USING YOUR DATA (correct method)
    plt.scatter(correct_samples[:,0], correct_samples[:,1], s=3) 

    plt.show()