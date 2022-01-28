# You may want to restart your notebook here, to reinitialize Open3D
import open3d
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import trimesh

from icecream import ic

vis = open3d.visualization.Visualizer()
vis.create_window(visible = False)

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
    plt.figure(figsize=(8,6))
    plt.imshow(np.asarray(img))
    for g in geoms:
        vis.remove_geometry(g)
        



if __name__ == "__main__":
    mesh = trimesh.load('./data/sievert.obj')
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(mesh.vertices)
    draw_geometries([pcd])
    plt.title("Sievert's surface")
    # plt.show()
    # mesh.show()
    
    import CalculCurvature as CC
    # Calculate Rusinkiewicz estimation of mean and gauss curvatures
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = CC.GetCurvaturesAndDerivatives(mesh)
    
    ic(PrincipalCurvatures, PrincipalDir1, PrincipalDir2)
    
    # gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    # mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

    # # Plot mean curvature
    # vect_col_map = \
    #     trimesh.visual.color.interpolate(mean_curv, color_map='jet')

    # if mean_curv.shape[0] == mesh.vertices.shape[0]:
    #     mesh.visual.vertex_colors = vect_col_map
    # elif mean_curv.shape[0] == mesh.faces.shape[0]:
    #     mesh.visual.face_colors = vect_col_map
    # mesh.show( background=[0, 0, 0, 255])

    # # PLot Gauss curvature
    # vect_col_map = \
    #     trimesh.visual.color.interpolate(gaussian_curv, color_map='jet')
    # if gaussian_curv.shape[0] == mesh.vertices.shape[0]:
    #     mesh.visual.vertex_colors = vect_col_map
    # elif gaussian_curv.shape[0] == mesh.faces.shape[0]:
    #     mesh.visual.face_colors = vect_col_map
    # mesh.show(background=[0, 0, 0, 255])