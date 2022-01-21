import argparse

from math import pi, cos, sin

import open3d as o3d

import numpy as np

import matplotlib.pyplot as plt
from icecream import ic


# semi-axis
a, b, c = 1, 1, 0.5

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
all_gemos = None

colormap = {
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "yellow": [1, 0.706, 0],
    "black": [0, 0, 0],
    "white": [1, 1, 1],
}


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
    # plt.show()


def create_arrow_from_vector(origin, vector, color=[1, 0, 1]):
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
    return ellipsoid, cf, arrow, curve


# -----------------------------------------------------------------------------


def gamma(t:float):
    """
    Parameterized Curves
    """
    # gamma_prime_t = (1, 0)
    # integrate gamma_prime_t -> (t+c1, c2)
    gamma_t0 = (pi / 4, pi / 6)
    gamma_t = [t + gamma_t0[0], gamma_t0[1]]
    return gamma_t


def f(u, v):
    "map f : R2 -> R3 | (u,v) to (x,y,z)"
    assert -pi < u < pi
    assert 0 < v < pi
    return np.array(
        [
            [a * cos(u) * sin(v)],
            [b * sin(u) * sin(v)],
            [c * cos(v)],
        ]
    )


def get_D_fp(u, v):
    return np.array(
        [
            [-a * sin(u) * sin(v), a * cos(u) * cos(v)],
            [b * cos(u) * sin(v), b * sin(u) * cos(v)],
            [0, -c * sin(v)],
        ]
    )
    

def get_D_Np(u, v):
    
    return np.array(
        [
            [b * c * sin(u) * sin(v) ** 2, -2 * b * c * cos(u) * sin(v) * cos(v)],
            [-a * c * cos(u) * sin(v) ** 2, -2 * a * c * sin(u) * sin(v) * cos(v)],
            [a * b * cos(u) * cos(v), a * b * sin(u) * sin(v)],
        ]
    )


def Q2(geoms, use_viewer=False):
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
    if use_viewer:
        o3d.visualization.draw_geometries(geoms_with_curve)
    else:
        draw_geometries(geoms_with_curve)
    return pcd


def flatten2list(arr):

    return arr.flatten().tolist()


def Q3_c(geoms, color, use_viewer=False, ):
    """"""
    p = [pi / 4, pi / 6]
    v = np.array([1, 0]).reshape(2, 1)

    Dfp_v = get_D_fp(p[0], p[1]) @ v

    fp = f(p[0], p[1])
    Dfpv_arrow = create_arrow_from_vector(
        flatten2list(fp), flatten2list(Dfp_v), color
    )
    geoms_with_Dfp = geoms + [Dfpv_arrow]

    if use_viewer:
        o3d.visualization.draw_geometries(geoms_with_Dfp)
    else:
        draw_geometries(geoms_with_Dfp)
    return Dfpv_arrow


def Q3_d(geoms, color, use_viewer=False):
    """
    what is the normal vecotr of the tangent plant at p
    N(u,v) = f_u X f_v / ||f_u X f_v||
    """
    u, v = [pi / 4, pi / 6]
    D_fp = get_D_fp(u, v)
    f_u = D_fp[:, 0]
    f_v = D_fp[:, 1]

    N = np.cross(f_u, f_v)
    N /= np.linalg.norm(N)
      
    assert(N@f_u < 1e-6)
    assert(N@f_v < 1e-6)
         
    N_arrow = create_arrow_from_vector(
        flatten2list(f(u, v)), flatten2list(N), color
    )
    geoms_with_surface_normal = geoms + [N_arrow]
    if use_viewer:
        o3d.visualization.draw_geometries(geoms_with_surface_normal)
    else:
        draw_geometries(geoms_with_surface_normal)
    return N_arrow

def Q3_e(geoms, color, use_viewer=False):
    u, v = [pi / 4, pi / 6]
    X = np.array([1, 0]).reshape(2, 1)
    D_fp = get_D_fp(u, v)
    f_u = D_fp[:, 0]
    f_v = D_fp[:, 1]

    N = np.cross(f_u, f_v)
    N /= np.linalg.norm(N)
    
    # binormal vector  
    B = np.cross((D_fp@X).flatten(), N)
    B /= np.linalg.norm(B)
    
    B_arrow = create_arrow_from_vector(
        flatten2list(f(u, v)), flatten2list(B), color=[1,1,0]
    )
    geoms_with_binormal = original_gemos + [B_arrow]
    if use_viewer:
        o3d.visualization.draw_geometries(geoms_with_binormal)
    else:
        draw_geometries(geoms_with_binormal)
    return B_arrow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_viewer", "-v", action="store_true")
    args = parser.parse_args()

    ellipsoid, cf, arrow, curve = collect_gemos()
    original_gemos = [ellipsoid, cf, arrow] + curve
    all_gemos = original_gemos

    # (2)
    curve3D = Q2(original_gemos, args.use_viewer)
    all_gemos.append(curve3D)

    # (3)c
    tangent3D = Q3_c(original_gemos, colormap["black"], args.use_viewer)
    all_gemos.append(tangent3D)

    # (3)d
    # surfanceNorm3D = Q3_d(original_gemos, colormap["yellow"], args.use_viewer)
    # all_gemos.append(surfanceNorm3D)
    
    # (3)e
    B_3D = Q3_e(original_gemos, colormap["white"], use_viewer=args.use_viewer)
    all_gemos.append(B_3D)
    
    o3d.visualization.draw_geometries(all_gemos)
    
    

    


    