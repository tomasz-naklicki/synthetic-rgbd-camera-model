import numpy as np
import os
import cv2
import open3d as o3d
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter


def filter_depth_with_local_min_scipy(
    depth_img: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    """
    Jak wyżej, ale korzysta z scipy.ndimage.minimum_filter.
    """
    mask_valid = np.isfinite(depth_img) & (depth_img > 0)
    # NaN → +inf, aby nie brać ich pod uwagę
    depth_inf = np.where(mask_valid, depth_img, np.inf)

    # aplikuj minimum_filter
    local_min = minimum_filter(
        depth_inf, size=kernel_size, mode="constant", cval=np.inf
    )

    # zachowaj oryginalne NaN-y
    filtered = np.where(mask_valid, np.minimum(depth_img, local_min), depth_img)
    return filtered.astype(np.float32)


class ProjectionManager:
    def __init__(
        self, rgb_params: dict, depth_params: dict, transformation_matrix: list
    ):
        self.rgb_params = rgb_params
        self.depth_params = depth_params
        self.K_rgb = np.array(
            [
                [rgb_params["fx"], 0, rgb_params["cx"]],
                [0, rgb_params["fy"], rgb_params["cy"]],
                [0, 0, 1],
            ]
        )
        self.K_depth = np.array(
            [
                [depth_params["fx"], 0, depth_params["cx"]],
                [0, depth_params["fy"], depth_params["cy"]],
                [0, 0, 1],
            ]
        )
        self.T = transformation_matrix

    # --- Step 1: Project Depth Image to 3D Point Cloud in Depth Camera Frame ---
    def depth_image_to_point_cloud_with_K(self, depth_img, K, dist_coeffs):
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
        depth_m = depth_img.astype(np.float32) / 1000.0
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        # maska nie-zerowych pikseli (głębia w mm)
        valid = (depth_m > 0.001) & (depth_m < 10)

        # return point_cloud
        uv = np.stack([u[valid], v[valid]], axis=1)  # shape=(N,2)
        Z = depth_m[valid]  # shape=(N,)

        # 3) Undistortuj te piksele do normalizowanych współrzędnych promienia
        pts = uv.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.undistortPoints(pts, K, dist_coeffs, P=None)  # shape=(N,1,2)
        x_norm = undist[:, 0, 0]
        y_norm = undist[:, 0, 1]

        denom = np.sqrt(1.0 + x_norm**2 + y_norm**2)
        Z_cam = Z / denom
        # # 4) Back-project: (X,Y,Z) = (x_norm*Z, y_norm*Z, Z)
        X = x_norm * Z
        Y = y_norm * Z
        X = x_norm * Z_cam
        Y = y_norm * Z_cam
        Z = Z_cam

        points = np.stack([X, Y, Z], axis=1)  # shape=(N,3)

        return points

    def transform_point_cloud_to_rgb(self, point_cloud, T):
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

    def project_points_to_pixels_filtered(
        self, rgb_img, points_rgb, K_rgb, image_shape
    ):
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
        print("→ ile punktów przed filtrem:", u.size)

        u2i = np.round(u).astype(int)
        v2i = np.round(v).astype(int)
        inside = (
            (u2i >= 0)
            & (u2i < rgb_img.shape[1])
            & (v2i >= 0)
            & (v2i < rgb_img.shape[0])
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

    def _get_projected_depth_image(
        self,
        points_rgb: np.ndarray,
        K_rgb: np.ndarray,
        image_shape: tuple,
        output_path: str = "output/depth_aligned_16bit.png",
    ):
        """
        Projects a point cloud into the RGB image frame and saves it as a 16-bit grayscale image.

        Parameters:
            points_rgb: (N, 3) 3D point cloud in RGB camera space.
            K_rgb: (3, 3) Intrinsic matrix of RGB camera.
            image_shape: (height, width) shape of the RGB image.
            output_path: Path to save the 16-bit PNG.
        """
        H, W = image_shape
        depth_img = np.full((H, W), 0, dtype=np.uint16)

        x, y, z = points_rgb[:, 0], points_rgb[:, 1], points_rgb[:, 2]
        z_safe = np.maximum(z, 1e-6)

        u = K_rgb[0, 0] * x / z_safe + K_rgb[0, 2]
        v = K_rgb[1, 1] * y / z_safe + K_rgb[1, 2]

        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        # Keep nearest point only
        z_mm = (z * 1000).astype(np.uint16)  # meters to millimeters
        depth_buffer = np.full((H, W), np.iinfo(np.uint16).max, dtype=np.uint16)

        for i in range(points_rgb.shape[0]):
            if 0 <= u_int[i] < W and 0 <= v_int[i] < H:
                z_val = z_mm[i]
                if z_val < depth_buffer[v_int[i], u_int[i]]:
                    depth_buffer[v_int[i], u_int[i]] = z_val

        # Replace max values with 0 (invalid)
        depth_img = np.where(depth_buffer < np.iinfo(np.uint16).max, depth_buffer, 0)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        return depth_img

    def get_aligned_depth_img(self, depth_img: np.ndarray, rgb_img: np.ndarray):
        # ----- Step 1: Convert Depth Image to Point Cloud -----
        point_cloud_depth = self.depth_image_to_point_cloud_with_K(
            depth_img, self.K_depth, dist_coeffs=self.depth_params["dist"]
        )
        print("Depth point cloud shape:", point_cloud_depth.shape)

        pc_rgb_via_inv = self.transform_point_cloud_to_rgb(point_cloud_depth, self.T)

        proj1, filt_depth1, idx, colors = self.project_points_to_pixels_filtered(
            rgb_img, pc_rgb_via_inv, self.K_rgb, rgb_img.shape[:2]
        )

        pcd = o3d.geometry.PointCloud()
        # punkty w przestrzeni RGB:
        pcd.points = o3d.utility.Vector3dVector(pc_rgb_via_inv[idx])
        # kolory:
        pcd.colors = o3d.utility.Vector3dVector(colors)

        H, W = rgb_img.shape[:2]
        depth_reproj = np.full((H, W), np.nan, dtype=np.float32)

        u2 = np.round(proj1[:, 0]).astype(int)
        v2 = np.round(proj1[:, 1]).astype(int)
        for ui, vi, zi in zip(
            u2, v2, filt_depth1
        ):  # tutaj "_" to filtered_depths zwrócone z project_points...
            if 0 <= ui < W and 0 <= vi < H:
                depth_reproj[vi, ui] = zi

        mask_valid = ~np.isnan(depth_reproj)
        mask_invalid = np.isnan(depth_reproj)

        # 2) Distance transform to find holes near real samples
        dist = cv2.distanceTransform(
            mask_invalid.astype(np.uint8), cv2.DIST_L2, maskSize=5
        )

        max_radius = 5
        mask_fill = mask_invalid & (dist <= max_radius)
        # 4) Block edge regions so we don't fill across depth discontinuities
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        d = depth_reproj.copy()
        d[np.isnan(d)] = 0.0
        # gradient w kierunku X i Y
        sobelx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)

        # 4.2) magnituda gradientu
        grad_mag = np.hypot(sobelx, sobely)

        # 4.3) maska miejsc o dużym gradiencie
        grad_thresh = 0.08  # [m/piksel] – dostosuj wg zakresu głębi
        mask_depth_edges = grad_mag > grad_thresh

        # 4.4) (Opcjonalnie) rozsuń maskę krawędzi głębi
        kernel = np.ones((15, 15), np.uint8)
        mask_depth_edges = cv2.dilate(
            mask_depth_edges.astype(np.uint8), kernel, iterations=1
        ).astype(bool)

        # 4.5) Usuń strefy dyskontynuacji z mask_fill
        fill_inside_px = (
            2  # tune this: holes closer than 5px to an edge will NOT be filled
        )
        # distance from the nearest edge pixel (i.e., inside edge zones)
        edge_mask_uint8 = mask_depth_edges.astype(np.uint8)
        edge_dist = cv2.distanceTransform(edge_mask_uint8, cv2.DIST_L2, 5)

        # only allow filling of NaNs that are at least N pixels away from any edge
        mask_fill = mask_invalid & (edge_dist >= fill_inside_px)

        # 5) Prepare known & to-be-filled lists
        rv, cvv = np.where(mask_valid)
        pts_valid = np.column_stack((cvv, rv))
        vals_valid = depth_reproj[rv, cvv]
        rf, cf = np.where(mask_fill)
        pts_fill = np.column_stack((cf, rf))

        # 6) Interpolate only at those hole locations
        filled_vals = griddata(
            pts_valid, vals_valid, (pts_fill[:, 0], pts_fill[:, 1]), method="nearest"
        )

        # 8) Zapisz wynik w depth_filled
        depth_filled = depth_reproj.copy()
        ok = ~np.isnan(filled_vals)
        depth_filled[rf[ok], cf[ok]] = filled_vals[ok]

        depth_filled_filtered = filter_depth_with_local_min_scipy(depth_filled)
        # 8) Now back-project only those pixels into 3D in the RGB frame:
        valid_pixels = ~np.isnan(depth_filled_filtered)
        vs, us = np.where(valid_pixels)  # row, col of only the good depths
        Z = depth_filled_filtered[vs, us]  # now Z contains no NaNs

        # back-project in RGB frame:
        X = (us - self.rgb_params["cx"]) * Z / self.rgb_params["fx"]
        Y = (vs - self.rgb_params["cy"]) * Z / self.rgb_params["fy"]

        points_rgb = np.stack([X, Y, Z], axis=1)
        colors = rgb_img[vs, us].astype(np.float32) / 255.0

        # 9) Build & save the dense Open3D point cloud exactly like before:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_rgb)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if not os.path.exists("output"):
            os.makedirs("output")
        o3d.io.write_point_cloud("output/colored_pointcloud_local_fill.ply", pcd)

        print(f"filled: {pcd}")
        depth_img_aligned = self._get_projected_depth_image(
            points_rgb, self.K_rgb, rgb_img.shape[:2]
        )
        return depth_img_aligned

    def get_aligned_depth_img_no_interp(
        self, depth_img: np.ndarray, rgb_img: np.ndarray
    ) -> np.ndarray:
        """
        Projects depth image into RGB frame using point cloud and camera calibration.
        Skips all interpolation/inpainting steps.

        Parameters:
            depth_img: 2D depth map in millimeters (uint16)
            rgb_img: RGB image (H, W, 3)

        Returns:
            Aligned depth image in RGB frame, dtype uint16
        """
        # Step 1: Get point cloud from depth image
        point_cloud_depth = self.depth_image_to_point_cloud_with_K(
            depth_img, self.K_depth, dist_coeffs=self.depth_params["dist"]
        )

        # Step 2: Transform point cloud to RGB space
        point_cloud_rgb = self.transform_point_cloud_to_rgb(point_cloud_depth, self.T)

        # Step 3: Project onto RGB image and keep closest point per pixel
        proj1, filt_depth1, idx, colors = self.project_points_to_pixels_filtered(
            rgb_img, point_cloud_rgb, self.K_rgb, rgb_img.shape[:2]
        )

        # Step 4: Keep only filtered subset of the RGB-space point cloud
        filtered_points_rgb = point_cloud_rgb[idx]

        # Step 5: Generate aligned depth image without interpolation
        depth_img_aligned = self._get_projected_depth_image(
            filtered_points_rgb, self.K_rgb, rgb_img.shape[:2]
        )

        return depth_img_aligned
