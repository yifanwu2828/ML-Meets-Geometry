# You may want to restart your notebook here, to reinitialize Open3D
import open3d
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import trimesh

from tqdm import tqdm
from icecream import ic

vis = open3d.visualization.Visualizer()
vis.create_window(visible=False)

# Make sure you call this function to draw the points for proper viewing direction
def draw_geometries(geoms):
    for g in geoms:
        vis.add_geometry(g)
    view_ctl = vis.get_view_control()
    view_ctl.set_up((0, 1, 0))
    view_ctl.set_front((0, 2, 1))
    view_ctl.set_lookat((0, 0, 0))
    view_ctl.set_zoom(1)
    # do not change this view point
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(True)
    plt.figure(figsize=(8, 6))
    plt.imshow(np.asarray(img))
    for g in geoms:
        vis.remove_geometry(g)


def solve():
    import CalculCurvature as CC

    # Calculate Rusinkiewicz estimation of mean and gauss curvatures
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = CC.GetCurvaturesAndDerivatives(
        mesh
    )

    ic(PrincipalCurvatures, PrincipalDir1, PrincipalDir2)

    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])


def normalize_row_matrix(arr):
    """
    Normalize each row of a matrix to make it a unit vector
    """
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def Rusinkiewicz_method(mesh, vertex_normals, face_normals, check=False):
    """

    S[Df].T Y = [Df].T \delta_n

    e0 = n2 - n1
    e1 = n0 - n2
    e2 = n1 - n0
    :param mesh:
    :param vertex_normals: numpy array of shape (n_vertices, 3)
    :param face_normals: numpy array of shape (n_faces, 3)
    """
    # Df = [xi_u xi_v]     xi _u (3,1) xi_v (3, 1)

    """
    Get all the edge vectors
    e0 is p2 - p1
    e1 is p0 - p2
    e2 is p1 - p0
    Note: all edge is not at normal
    """
    e0 = np.array(
        mesh.vertices[mesh.faces[:, 2], :] - mesh.vertices[mesh.faces[:, 1], :]
    )
    e1 = np.array(
        mesh.vertices[mesh.faces[:, 0], :] - mesh.vertices[mesh.faces[:, 2], :]
    )
    e2 = np.array(
        mesh.vertices[mesh.faces[:, 1], :] - mesh.vertices[mesh.faces[:, 0], :]
    )

    # Normalize edge vectors to use as xi_u (any edge among e0, e1, e2 is OK)
    e0_normalize = normalize_row_matrix(e0)
    # e1_normalize = normalize_row_matrix(e1)
    # e2_normalize = normalize_row_matrix(e2)


    principal_curvature = np.zeros((len(mesh.faces), 2))
    principal_dir1 = np.zeros((len(mesh.faces), 3), )
    principal_dir2 = np.zeros_like(principal_dir1)
    
    for i in tqdm(range(len(mesh.faces))):
        # Build orthonormal basis Dfp = [xi_u, xi_v] for each face
        # To find xi_v is easy, we just need to find the cross product of xi_u and face Normal
        xi_u = e0_normalize[i, :]
        face_N = face_normals[i, :]

        #  xi_v is also binormal vector
        xi_v = np.cross(xi_u, face_N)
        xi_v /= np.linalg.norm(xi_v)

        Dfp_i_transpose = np.vstack((xi_u, xi_v))

        n0 = vertex_normals[mesh.faces[i, 0], :].reshape(-1, 1)
        n1 = vertex_normals[mesh.faces[i, 1], :].reshape(-1, 1)
        n2 = vertex_normals[mesh.faces[i, 2], :].reshape(-1, 1)
        # assuming vertex normals are already normalized
        if check:
            np.testing.assert_almost_equal(np.linalg.norm(n0), 1.0)
            np.testing.assert_almost_equal(np.linalg.norm(n1), 1.0)
            np.testing.assert_almost_equal(np.linalg.norm(n2), 1.0)
        """
        S Df.T e0 = Df.T (n2 - n1)
        S Df.T e1 = Df.T (n0 - n2)
        S Df.T e2 = Df.T (n1 - n0)   
        """
        # * solve least squares problem of  form Ax = b
        # 6 equations and 4 unknowns
        # A (6,4)
        A = np.array(
            [
                [e0[i, :] @ xi_u, e0[i, :] @ xi_v, 0,                0],
                [0              , 0,               e0[i, :] @ xi_u,  e0[i, :] @ xi_v],
                [e1[i, :] @ xi_u, e1[i, :] @ xi_v, 0,                0],
                [0,               0,               e1[i, :] @ xi_u,  e1[i, :] @ xi_v],
                [e2[i, :] @ xi_u, e2[i, :] @ xi_v, 0,                0],
                [0,               0,               e2[i, :] @ xi_u,  e2[i, :] @ xi_v],
            ]
        )

        # (6,1)
        b = np.vstack(
            [
                Dfp_i_transpose @ (n2 - n1),
                Dfp_i_transpose @ (n0 - n2),
                Dfp_i_transpose @ (n1 - n0),
            ]
        )
        # (4,1) -> (2,2)
        S = np.linalg.lstsq(A, b, rcond=None)[0]
        S = S.reshape(2, 2)
        
        eigval, eigvec = LA.eigh(S)
        
        k1_idx = np.argmax(eigval)
        k2_idx = np.argmin(eigval)
        principal_curvature[i, 0] = eigval[k1_idx]
        principal_curvature[i, 1] = eigval[k2_idx]
        
        principal_dir1[i, :2] = eigvec[:, k1_idx]
        principal_dir2[i, :2] = eigvec[:, k2_idx]
        
    return principal_curvature, principal_dir1, principal_dir2



def test():
    import CalculCurvature as CC
    # Generate a sphere
    mesh = trimesh.load('./data/sievert.obj')

    # Show th sphere
    mesh.show()

    # Calculate Rusinkiewicz estimation of mean and gauss curvatures
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = CC.GetCurvaturesAndDerivatives(mesh)
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

    # Plot mean curvature
    vect_col_map = \
        trimesh.visual.color.interpolate(mean_curv, color_map='jet')

    if mean_curv.shape[0] == mesh.vertices.shape[0]:
        mesh.visual.vertex_colors = vect_col_map
    elif mean_curv.shape[0] == mesh.faces.shape[0]:
        mesh.visual.face_colors = vect_col_map
    mesh.show( background=[0, 0, 0, 255])

    # PLot Gauss curvature
    vect_col_map = \
        trimesh.visual.color.interpolate(gaussian_curv, color_map='jet')
    if gaussian_curv.shape[0] == mesh.vertices.shape[0]:
        mesh.visual.vertex_colors = vect_col_map
    elif gaussian_curv.shape[0] == mesh.faces.shape[0]:
        mesh.visual.face_colors = vect_col_map
    mesh.show(background=[0, 0, 0, 255])


def main():
    mesh = trimesh.load('./data/sievert.obj')
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(mesh.vertices)
    # draw_geometries([pcd])
    # plt.show()
    # mesh.show()

    # vertex norms
    ic(mesh.vertex_normals.shape)

    # face normals
    ic(mesh.face_normals.shape)

    principal_curvature, principal_dir1, principal_dir2 = Rusinkiewicz_method(
        mesh,
        vertex_normals=mesh.vertex_normals,
        face_normals=mesh.face_normals,
        check=True,
    )

    ic(principal_curvature[:22, :])
    
    gaussian_curv = principal_curvature[:, 1] * principal_curvature[:, 0]
    mean_curv = 0.5*(principal_curvature[:, 1] + principal_curvature[:, 0])

    # Plot mean curvature
    vect_col_map = trimesh.visual.color.interpolate(mean_curv, color_map='jet')
    
    if mean_curv.shape[0] == mesh.vertices.shape[0]:
        mesh.visual.vertex_colors = vect_col_map
    elif mean_curv.shape[0] == mesh.faces.shape[0]:
        mesh.visual.face_colors = vect_col_map
    mesh.show( background=[0, 0, 0, 255])
    
    # PLot Gauss curvature
    vect_col_map = trimesh.visual.color.interpolate(gaussian_curv, color_map='jet')
    if gaussian_curv.shape[0] == mesh.vertices.shape[0]:
        mesh.visual.vertex_colors = vect_col_map
    elif gaussian_curv.shape[0] == mesh.faces.shape[0]:
        mesh.visual.face_colors = vect_col_map
    mesh.show(background=[0, 0, 0, 255])

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
