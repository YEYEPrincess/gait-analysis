import numpy as np
import open3d as o3d
import cv2

def render_mesh_offscreen(vertices, faces, w, h):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w), height=int(h), visible=True)

    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)  # (H,W,3) float->uint8

    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

    return img