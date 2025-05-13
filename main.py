import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cv2
from numpy.linalg import inv
import mathutils

# --- Camera Calibration Data ---

# Depth camera intrinsics
fx_d = 0.8832 / 0.0035
fy_d = 0.8832 / 0.0035
fx_d = 252.376083
fy_d = 252.347733
# cx_d = 258.8008728027344
# cy_d = 254.26467895507812
cx_d = 258.800873
cy_d = 254.264679
K_depth = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])
dist_depth = np.array(
    [12.565655, 6.072149, 0.000045, 0.000020, 0.208180, 12.887096, 10.294142, 1.362217],
    dtype=np.float64,
)

dist_depth = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


# RGB camera intrinsics
fx_rgb = 1.039 / 0.00125
fy_rgb = 1.039 / 0.00125
fx_rgb = 747.707031
fy_rgb = 747.948425

# cx_rgb = 626.0764770507812
# cy_rgb = 356.4728698730469
cx_rgb = 626.076477
cy_rgb = 356.472870
K_rgb = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
dist_rgb = np.array(
    [0.079764, -0.108468, -0.000167, -0.000509, 0.044803], dtype=np.float64
)

#############################aligned
fx_d = fx_rgb
fy_d = fy_rgb
cx_d = cx_rgb
cy_d = cy_rgb
K_depth = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])

# T_from_blender = [
#     [0.99404520, 0.00560837, -0.10874305, -0.03248300],
#     [-0.00527298, 0.99998039, 0.00336748, 0.00138313],
#     [0.10875984, -0.00277311, 0.99405527, 0.00255620],
#     [0.00000000, 0.00000000, 0.00000000, 1.00000000],
# ]


T_real = [
    [0.994052, 0.002774, 0.005608, -32.665543 / 1000.0],
    [-0.003367, 0.994064, 0.108743, -0.986931 / 1000.0],
    [-0.005723, -0.108760, 0.994054, 2.863724 / 1000.0],
    [0.0, 0.0, 0.0, 1.0],
]

F = np.diag([-1, 1, -1, 1]).astype(np.float32)

T_real = np.eye(4)


# T_real = F @ T_real @ F
############im bardziej skomplikowana ta macierz tym gorzej działa -> błędy numeryczne?, czemu ostatni wiersz musi mieć zmieniony znak???? OS X


# --- Step 1: Project Depth Image to 3D Point Cloud in Depth Camera Frame ---
def depth_image_to_point_cloud_with_K(depth_img, K, dist_coeffs):
    """
    Converts a depth image to a point cloud using the camera intrinsic matrix K,
    but filters out any pixels whose normalized depth is effectively 1.0 (the far clip).

    Parameters:
        depth_img: A 2D NumPy array of shape (H, W), dtype uint16, where each element
                   is the depth mapped to [0..65535].
        K: The 3x3 camera intrinsic matrix.

    Returns:
        point_cloud: An (M, 3) array where each row is a 3D point (x, y, z) in the camera frame,
                     for M <= H*W valid pixels.
    """
    H, W = depth_img.shape
    # near, far = 0.1, 15.0

    # # 1) Normalize and map back to meters
    # normalized = depth_img.astype(np.float32) / 65535.0
    # # depth_m = normalized * (far - near) + near
    depth_m = depth_img.astype(np.float32) / 1000.0
    # print("depth_m range:", depth_m.min(), depth_m.max())

    # # 2) Build mask of “real” pixels (drop those at or above the far‐clip)
    # valid_mask = (normalized.flatten() > 1e-3) & (normalized.flatten() < 0.999)

    # # 3) Flatten and mask depths
    # depth_flat = depth_m.flatten()[valid_mask.flatten()]
    # # 4) Build and mask pixel coordinates
    # u, v = np.meshgrid(np.arange(W), np.arange(H))
    # uv1 = np.stack((u, v, np.ones_like(u)), axis=0).reshape(3, -1)
    # uv1 = uv1[:, valid_mask.flatten()]

    # print("→ u range:", u.min(), u.max())
    # print("→ v range:", v.min(), v.max())
    # print("→ obraz ma W,H =", rgb_img.shape[1], rgb_img.shape[0])
    # print("→ ile punktów przed filtrem:", u.size)

    # # 5) Back‐project only valid pixels
    # K_inv = np.linalg.inv(K)
    # dirs = K_inv @ (uv1)  # shape (3, M)
    # lengths = np.linalg.norm(dirs, axis=0)
    # unit_dirs = dirs / lengths  # unit‐length rays

    # z_axial = depth_flat / lengths  # actual Z along camera axis
    # points = unit_dirs * depth_flat  # this gives [x',y',1]*depth_rad
    # # but to be precise, X = x'*z_axial, Y = y'*z_axial, Z = z_axial:
    # X = dirs[0] * z_axial / dirs[2]
    # Y = dirs[1] * z_axial / dirs[2]
    # Z = z_axial

    # point_cloud = np.stack((X, Y, Z), axis=-1)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # maska nie-zerowych pikseli (głębia w mm)
    valid = (depth_m > 0.001) & (depth_m < 5)
    # u_valid = u[valid]
    # v_valid = v[valid]
    # Z = depth_m[valid]
    # X = (u_valid - cx_d) * Z / fx_d
    # Y = (v_valid - cy_d) * Z / fy_d
    # point_cloud = np.stack((X, Y, Z), axis=-1)

    # return point_cloud
    uv = np.stack([u[valid], v[valid]], axis=1)  # shape=(N,2)
    Z = depth_m[valid]  # shape=(N,)

    # 3) Undistortuj te piksele do normalizowanych współrzędnych promienia
    pts = uv.reshape(-1, 1, 2).astype(np.float32)
    undist = cv2.undistortPoints(pts, K, dist_coeffs, P=None)  # shape=(N,1,2)
    x_norm = undist[:, 0, 0]
    y_norm = undist[:, 0, 1]

    # 4) Back-project: (X,Y,Z) = (x_norm*Z, y_norm*Z, Z)
    X = x_norm * Z
    Y = y_norm * Z
    points = np.stack([X, Y, Z], axis=1)  # shape=(N,3)

    return points


def transform_point_cloud_to_rgb(point_cloud, T):
    """
    Transforms a point cloud from the depth camera coordinate frame to the RGB camera coordinate frame.

    Parameters:
        point_cloud: An (N, 3) array of points in the depth camera coordinate system.
        T: A 4x4 homogeneous transformation matrix that maps points from the depth camera
           coordinate frame to the RGB camera coordinate frame.

    Returns:
        points_rgb: An (N, 3) array where each row is the transformed point in the RGB camera space.
    """
    N = point_cloud.shape[0]
    # Convert points to homogeneous coordinates: add a column of ones.
    ones = np.ones((N, 1), dtype=np.float32)
    points_hom = np.hstack((point_cloud, ones))  # Shape: (N, 4)

    # Apply the transformation: p_rgb_hom = T * p_depth_homogeneous.
    points_rgb_hom = (T @ points_hom.T).T  # Shape: (N, 4)

    # Convert back to 3D by dropping the homogeneous coordinate.
    points_rgb = points_rgb_hom[:, :3]
    xs, ys, zs = points_rgb.T
    print("X range:", xs.min(), xs.max())
    print("Y range:", ys.min(), ys.max())
    print("Z range:", zs.min(), zs.max())
    return points_rgb


def project_points_to_pixels_filtered(points_rgb, K_rgb, image_shape):
    """
    Projects 3D points in the RGB camera space onto the image plane using the intrinsic matrix K_rgb.
    If multiple points project to the same integer pixel coordinate (collision), only the point with
    the smallest (closest) z value is retained.

    Parameters:
        points_rgb: (N, 3) NumPy array of points [x, y, z] in the RGB camera coordinate system.
        K_rgb: 3x3 intrinsic matrix for the RGB camera.
        image_shape: Tuple (height, width) of the target RGB image.

    Returns:
        filtered_proj_pixels: (M, 2) NumPy array of filtered floating-point pixel coordinates for the points.
        filtered_depths: (M,) NumPy array containing the corresponding z values (depths) of those points.
        indices: List of indices of the points that are kept.
    """
    H, W = image_shape

    # Extract x, y, and z from the point cloud.
    x = points_rgb[:, 0]
    y = points_rgb[:, 1]
    z = points_rgb[:, 2]

    # Prevent division by zero:
    z_safe = np.maximum(z, 1e-6)

    # Compute pixel coordinates using the pinhole camera model:
    u = (K_rgb[0, 0] * x / z_safe) + K_rgb[0, 2]
    v = (K_rgb[1, 1] * y / z_safe) + K_rgb[1, 2]

    # Floating-point pixel coordinates (for visualization).
    proj_pixels = np.stack((u, v), axis=-1)

    # Round to integer pixel indices to determine which points collide.
    u_int = np.round(u).astype(int)
    v_int = np.round(v).astype(int)

    # Dictionary to keep the index of the point with the lowest z for each pixel.
    pixel_to_index = {}
    print("→ u range:", u.min(), u.max())
    print("→ v range:", v.min(), v.max())
    print("→ obraz ma W,H =", rgb_img.shape[1], rgb_img.shape[0])
    print("→ ile punktów przed filtrem:", u.size)

    u2i = np.round(u).astype(int)
    v2i = np.round(v).astype(int)
    inside = (
        (u2i >= 0) & (u2i < rgb_img.shape[1]) & (v2i >= 0) & (v2i < rgb_img.shape[0])
    )
    print("punkty w kadrze:", inside.sum(), "z", len(u2i))
    # Iterate over all points.
    for i in range(points_rgb.shape[0]):
        # Check that the projected indices are within image bounds.
        if 0 <= u_int[i] < W and 0 <= v_int[i] < H:
            key = (u_int[i], v_int[i])
            if key in pixel_to_index:
                # If a collision occurs, keep the point with the lower (closer) z.
                if z[i] < z[pixel_to_index[key]]:
                    pixel_to_index[key] = i
            else:
                pixel_to_index[key] = i

    # Retrieve indices of points to keep.
    indices = list(pixel_to_index.values())
    filtered_proj_pixels = proj_pixels[indices]
    filtered_depths = z[indices]
    colors = rgb_img[v_int[indices], u_int[indices]].astype(np.float32) / 255.0

    return filtered_proj_pixels, filtered_depths, indices, colors


# --- Example Data & Usage ---

# depth_img = cv2.imread("rendered/render_with_compositing_0.png", cv2.IMREAD_UNCHANGED)
depth_img = cv2.imread("rendered/depth_0000.png", cv2.IMREAD_UNCHANGED)
print("depth:", depth_img.dtype, depth_img.min(), depth_img.max())

# rgb_img = cv2.imread("rendered/render_no_compositing_0.png")
rgb_img = cv2.imread("rendered/rgb_0000.png")
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

print("DEBUG: depth_img.shape =", depth_img.shape)
print("DEBUG: rgb_img.shape   =", rgb_img.shape)


# depth_u = cv2.undistort(depth_img, K_depth, dist_depth, None, K_depth)
# rgb_u = cv2.undistort(rgb_img,   K_rgb,   dist_rgb,   None, K_rgb)

# ----- Step 1: Convert Depth Image to Point Cloud -----
point_cloud_depth = depth_image_to_point_cloud_with_K(
    depth_img, K_depth, dist_coeffs=dist_depth
)
print("Depth point cloud shape:", point_cloud_depth.shape)


pc_rgb_via_inv = transform_point_cloud_to_rgb(point_cloud_depth, T_real)


proj1, filt_depth1, idx, colors = project_points_to_pixels_filtered(
    pc_rgb_via_inv, K_rgb, rgb_img.shape[:2]
)


# proj2, _, _ = project_points_to_pixels_filtered(pc_rgb_via_direct,K_rgb, rgb_img.shape[:2])

import open3d as o3d

pcd = o3d.geometry.PointCloud()
# punkty w przestrzeni RGB:
pcd.points = o3d.utility.Vector3dVector(pc_rgb_via_inv[idx])
# kolory:
pcd.colors = o3d.utility.Vector3dVector(colors)

# 3) Zapis do pliku PLY:
o3d.io.write_point_cloud("colored_pointcloud.ply", pcd)

# 4) Interaktywna wizualizacja:
o3d.visualization.draw_geometries(
    [pcd], window_name="Kolorowy PointCloud", width=1280, height=720
)

# 3) Plot them on top of your real RGB image:
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(rgb_img, origin="upper")
ax.set_xlim(0, rgb_img.shape[1])
ax.set_ylim(rgb_img.shape[0], 0)

# inv (red) vs direct (lime)
ax.scatter(proj1[:, 0], proj1[:, 1], s=1, c="red", label="using T_depth2color")
# ax.scatter(proj2[:,0], proj2[:,1], s=1, c='lime',  label='using T_color2depth')

ax.legend(loc="lower right")
ax.set_title("Red=inv, Green=direct")
plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(rgb_img, origin="upper")
ax.scatter(proj1[:, 0], proj1[:, 1], s=1, c="red", label="reprojection")
ax.legend(loc="lower right")
ax.set_title("RGB + reprojection")
ax.axis("off")
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(rgb_img, origin="upper")
ax.scatter(proj1[:, 0], proj1[:, 1], s=1, c="red", label="reprojection")
ax.legend(loc="lower right")
ax.set_title("RGB + reprojection")
ax.axis("off")
plt.show()


H, W = rgb_img.shape[:2]
depth_reproj = np.full((H, W), np.nan, dtype=np.float32)

u2 = np.round(proj1[:, 0]).astype(int)
v2 = np.round(proj1[:, 1]).astype(int)
for ui, vi, zi in zip(
    u2, v2, filt_depth1
):  # tutaj "_" to filtered_depths zwrócone z project_points...
    if 0 <= ui < W and 0 <= vi < H:
        depth_reproj[vi, ui] = zi

plt.figure(figsize=(6, 6))
plt.imshow(depth_reproj, cmap="gray", origin="upper")
plt.title("Reprojected Depth Map (gray)")
plt.axis("off")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(depth_reproj, cmap="jet", origin="upper")
plt.colorbar(label="Depth [m]")
plt.title("Reprojected Depth Map (jet)")
plt.axis("off")
plt.show()


from scipy.interpolate import griddata

input_depth = depth_reproj.copy()

# mask
valid_mask = ~np.isnan(input_depth)
kernel = np.ones((5, 5), np.uint8)
eroded_mask = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

H, W = input_depth.shape
u, v = np.meshgrid(np.arange(W), np.arange(H))

if np.count_nonzero(eroded_mask) < 20:
    print("eroded_mask is too sparse; falling back to full valid_mask")
    eroded_mask = valid_mask

known_coords = np.stack((v[eroded_mask], u[eroded_mask]), axis=-1)
known_values = input_depth[eroded_mask]

# Step 5: Prepare target coordinates (interior NaNs only)
target_mask = np.isnan(input_depth) & (
    cv2.dilate(eroded_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
)
target_coords = np.stack((v[target_mask], u[target_mask]), axis=-1)

# Step 6: Use griddata to interpolate missing values
if known_coords.size > 0 and target_coords.size > 0:
    inpainted_values = griddata(
        known_coords, known_values, target_coords, method="linear"
    )
else:
    print("not enough valid data.")
    inpainted_values = np.array([])

inpainted_depth = input_depth.copy()
if inpainted_values.size > 0:
    inpainted_depth[target_mask] = inpainted_values

plt.figure(figsize=(6, 6))
plt.imshow(inpainted_depth, cmap="jet", origin="upper")
plt.colorbar(label="Depth [m]")
plt.title("Inpainted Depth Map (Preserving Background NaNs)")
plt.axis("off")
plt.show()
plt.show()
