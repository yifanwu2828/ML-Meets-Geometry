from math import pi, cos, sin

import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


# semi-axis
a, b, c = 1, 1, 0.5

vis = o3d.visualization.Visualizer()
vis.create_window(visible = False)
all_gemos = None

def draw_geometries(geoms):
    for g in geoms:
        vis.add_geometry(g)
    view_ctl = vis.get_view_control()
    view_ctl.set_up((0, 1e-4, 1))
    view_ctl.set_front((0, 0.5, 2))
    view_ctl.set_lookat((0, 0, 0))
    # do not change this view point
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(True)
    plt.figure(figsize=(8, 6))
    plt.imshow(np.asarray(img)[::-1, ::-1])
    for g in geoms:
        vis.remove_geometry(g)
    plt.show()    


def create_arrow_from_vector(origin, vector, color = [1, 0, 1]):
    """
    origin: origin of the arrow
    vector: direction of the arrow
    """
    v = np.array(vector)
    v /= np.linalg.norm(v)
    z = np.array([0, 0, 1])
    angle = np.arccos(z @ v)
    
    arrow = o3d.geometry.TriangleMesh.create_arrow(0.05, 0.1, 0.25, 0.2)
    arrow.paint_uniform_color(color)
    T = np.eye(4)
    T[:3, 3] = np.array(origin)
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(np.cross(z, v) * angle)
    arrow.transform(T)
    return arrow


def create_ellipsoid(a, b, c):
    sphere = o3d.geometry.TriangleMesh.create_sphere()
    sphere.transform(np.diag([a, b, c, 1]))
    sphere.compute_vertex_normals()
    return sphere


def create_lines(points):
    lines = []
    for p1, p2 in zip(points[:-1], points[1:]):
        height = np.linalg.norm(p2 - p1)
        center = (p1 + p2) / 2
        d = p2 - p1
        d /= np.linalg.norm(d)
        axis = np.cross(np.array([0, 0, 1]), d)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.array([0, 0, 1]) @ d)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(0.02, height)
        cylinder.transform(T)
        cylinder.paint_uniform_color([1, 0, 0])
        lines.append(cylinder)
    return lines


def collect_gemos():
    """
    exapmle code to draw ellipsoid, curve, and arrows
    """
    arrow = create_arrow_from_vector([0.0, 0.0, 1.0], [1.0, 1.0, 0.0])
    ellipsoid = create_ellipsoid(a, b, c)
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    cf.scale(1.5, (0, 0, 0))
    curve = create_lines(
        np.array(
            [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, 1]],
            dtype=np.float64,
        )
    )
    return ellipsoid, cf, arrow , curve

# -----------------------------------------------------------------------------

def f(u, v):
    "map f : R2 -> R3 | (u,v) to (x,y,z)"
    assert -pi < u < pi
    assert 0 < v < pi
    return np.array([
        [a * cos(u) * sin(v)],
        [b * sin(u) * sin(v)],
        [c * cos(v)],
    ])


def D_fp(u,v):
    return np.array(
        [
            [-a * sin(u) * sin(v), a * cos(u) * cos(v)],
            [b * cos(u) * sin(v), b * sin(u) * cos(v)],
            [0, -c * sin(v)],
        ]
    )
   

def surface_normal(u,v):
    dfp = D_fp(u,v)
    fu = dfp[:, 0]
    fv = dfp[:, 1]
    cross = np.cross(fu, fv)
    norm = np.linalg.norm(cross)
    return cross / norm


def gamma(t):
    """
    Parameterized Curves
    """
    # gamma_prime_t = (1, 0)
    # integrate gamma_prime_t -> (t+c1, c2)
    gamma_t0 = (pi / 4, pi / 6)
    gamma_t = [t + gamma_t0[0], gamma_t0[1]]
    return gamma_t


def Q2(geoms):
    """
    Let p = ( pi/4 , pi/6 ) and v = (1, 0). Draw the curve of ( f ◦ γ)(t) on the surface
    of the ellipsoid.
    """
    p = np.array([pi / 4, pi / 6])

    time_steps = np.arange(0, 1, 0.01)
    gamma_t = np.array([gamma(t) for t in time_steps])
                        
    curve2D = np.vstack([p, gamma_t])

    f_curve3D = np.asarray([f(u, v).ravel() for u, v in curve2D])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(f_curve3D)
    geoms_with_curve = geoms + [pcd]
    draw_geometries(geoms_with_curve)

def flatten2list(arr):
    return arr.flatten().tolist()

def Q3(geoms):
    p = [pi/4, pi/6]
    v = np.array([1, 0]).reshape(2, 1)
    Dfp = D_fp(*p)

    Dfp_v = Dfp @ v
    ic(Dfp_v)

    fp = f(*p).flatten()
    Dfpv_arrow =  create_arrow_from_vector(flatten2list(fp), flatten2list(Dfp_v), color=[1, 0.706, 0])
    geoms_with_arrow = geoms + [Dfpv_arrow]
    draw_geometries(geoms_with_arrow)
    


if __name__ == "__main__":

    ellipsoid, cf, arrow , curve = collect_gemos()
    orignal_gemos = [ellipsoid, cf, arrow] + curve
    # o3d.visualization.draw_geometries(all_gemos)
    
    
    # # (2)
    Q2(orignal_gemos)
   
    # (3)
    Q3(orignal_gemos)