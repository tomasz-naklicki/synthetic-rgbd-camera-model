import numpy as np
import os
import cv2
import open3d as o3d
import numpy as np
from scipy.interpolate import griddata
from src.utils import filter_depth_with_local_min_scipy


class ProjectionManager:
    """Handles conversion between depth images and point clouds, and reprojection."""
    def __init__(
        self, rgb_params: dict, depth_params: dict, transformation_matrix: list
    ):
        """Initialize with camera intrinsics and extrinsic transform.

        Args:
            rgb_params (dict): Intrinsics for RGB camera (fx, fy, cx, cy).
            depth_params (dict): Intrinsics for depth camera, plus distortion.
            transformation_matrix (list or np.ndarray): 4×4 transform from
                depth-camera frame to RGB-camera frame.
        """
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

    def depth_image_to_point_cloud_with_K(self, depth_img, K, dist_coeffs):
        """Back-project a depth map to a 3D point cloud, undistorting as needed.

        Args:
            depth_img (np.ndarray): 2D depth image in raw units (e.g., mm).
            K (np.ndarray): 3×3 camera intrinsic matrix.
            dist_coeffs (array-like): Distortion coefficients for undistortion.

        Returns:
            np.ndarray: N×3 array of 3D points in the camera frame (meters).
        """
        H, W = depth_img.shape
        depth_m = depth_img.astype(np.float32) / 1000.0
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        valid = (depth_m > 0.001) & (depth_m < 10)

        uv = np.stack([u[valid], v[valid]], axis=1)
        Z = depth_m[valid]

        pts = uv.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.undistortPoints(pts, K, dist_coeffs, P=None)
        x_norm = undist[:, 0, 0]
        y_norm = undist[:, 0, 1]

        denom = np.sqrt(1.0 + x_norm**2 + y_norm**2)
        Z_cam = Z / denom
        X = x_norm * Z
        Y = y_norm * Z
        X = x_norm * Z_cam
        Y = y_norm * Z_cam
        Z = Z_cam

        points = np.stack([X, Y, Z], axis=1)

        return points

    def transform_point_cloud_to_rgb(self, point_cloud, T):
        """Apply a 4×4 transform to bring points into RGB camera frame.

        Args:
            point_cloud (np.ndarray): N×3 array of points in depth frame.
            T (np.ndarray): 4×4 homogeneous transformation matrix.

        Returns:
            np.ndarray: N×3 array of points in RGB camera frame.
        """
        N = point_cloud.shape[0]
        ones = np.ones((N, 1), dtype=np.float32)
        points_hom = np.hstack((point_cloud, ones))

        points_rgb_hom = (T @ points_hom.T).T

        points_rgb = points_rgb_hom[:, :3]
        xs, ys, zs = points_rgb.T
        print("X range:", xs.min(), xs.max())
        print("Y range:", ys.min(), ys.max())
        print("Z range:", zs.min(), zs.max())
        return points_rgb

    def project_points_to_pixels_filtered(
        self, rgb_img, points_rgb, K_rgb, image_shape
    ):
        """Project 3D points to pixel coordinates, keeping only closest per-pixel.

        Args:
            rgb_img (np.ndarray): RGB image array.
            points_rgb (np.ndarray): N×3 points in RGB camera frame.
            K_rgb (np.ndarray): 3×3 intrinsics of RGB camera.
            image_shape (tuple): (height, width) of the image.

        Returns:
            tuple:
                - proj_pixels (np.ndarray): M×2 float pixel coordinates.
                - filtered_depths (np.ndarray): M depth values (meters).
                - indices (list[int]): Indices of kept points.
                - colors (np.ndarray): M×3 RGB colors normalized to [0,1].
        """
        H, W = image_shape

        x = points_rgb[:, 0]
        y = points_rgb[:, 1]
        z = points_rgb[:, 2]

        z_safe = np.maximum(z, 1e-6)

        u = (K_rgb[0, 0] * x / z_safe) + K_rgb[0, 2]
        v = (K_rgb[1, 1] * y / z_safe) + K_rgb[1, 2]

        proj_pixels = np.stack((u, v), axis=-1)

        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)

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
        for i in range(points_rgb.shape[0]):
            if 0 <= u_int[i] < W and 0 <= v_int[i] < H:
                key = (u_int[i], v_int[i])
                if key in pixel_to_index:
                    if z[i] < z[pixel_to_index[key]]:
                        pixel_to_index[key] = i
                else:
                    pixel_to_index[key] = i

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
        """Render a depth image from 3D points in RGB frame without interpolation.

        Args:
            points_rgb (np.ndarray): M×3 points in RGB camera frame (meters).
            K_rgb (np.ndarray): 3×3 intrinsics of RGB camera.
            image_shape (tuple): (height, width) for output image.
            output_path (str, optional): Path to save uint16 depth PNG
                (not currently written here). Defaults to above.

        Returns:
            np.ndarray: H×W uint16 depth image (0 = no data).
        """
        H, W = image_shape
        depth_img = np.full((H, W), 0, dtype=np.uint16)

        x, y, z = points_rgb[:, 0], points_rgb[:, 1], points_rgb[:, 2]
        z_safe = np.maximum(z, 1e-6)

        u = K_rgb[0, 0] * x / z_safe + K_rgb[0, 2]
        v = K_rgb[1, 1] * y / z_safe + K_rgb[1, 2]

        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        z_mm = (z * 1000).astype(np.uint16)
        depth_buffer = np.full((H, W), np.iinfo(np.uint16).max, dtype=np.uint16)

        for i in range(points_rgb.shape[0]):
            if 0 <= u_int[i] < W and 0 <= v_int[i] < H:
                z_val = z_mm[i]
                if z_val < depth_buffer[v_int[i], u_int[i]]:
                    depth_buffer[v_int[i], u_int[i]] = z_val

        depth_img = np.where(depth_buffer < np.iinfo(np.uint16).max, depth_buffer, 0)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        return depth_img

    def get_aligned_depth_img(self, depth_img: np.ndarray, rgb_img: np.ndarray):
        """Full pipeline: align a processed depth image to the RGB frame.

        1. Back-project depth → 3D points.
        2. Transform to RGB frame.
        3. Project and interpolate missing pixels.
        4. Filter and re-project to depth image.

        Args:
            depth_img (np.ndarray): 2D depth map (uint16 or float32).
            rgb_img (np.ndarray): Corresponding RGB image (H×W×3).

        Returns:
            np.ndarray: Aligned uint16 depth image in RGB resolution.
        """
        point_cloud_depth = self.depth_image_to_point_cloud_with_K(
            depth_img, self.K_depth, dist_coeffs=self.depth_params["dist"]
        )
        print("Depth point cloud shape:", point_cloud_depth.shape)
        pc_rgb_via_inv = self.transform_point_cloud_to_rgb(point_cloud_depth, self.T)

        proj1, filt_depth1, idx, colors = self.project_points_to_pixels_filtered(
            rgb_img, pc_rgb_via_inv, self.K_rgb, rgb_img.shape[:2]
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_rgb_via_inv[idx])
        pcd.colors = o3d.utility.Vector3dVector(colors)

        H, W = rgb_img.shape[:2]
        depth_reproj = np.full((H, W), np.nan, dtype=np.float32)

        u2 = np.round(proj1[:, 0]).astype(int)
        v2 = np.round(proj1[:, 1]).astype(int)
        for ui, vi, zi in zip(u2, v2, filt_depth1):
            if 0 <= ui < W and 0 <= vi < H:
                depth_reproj[vi, ui] = zi

        mask_valid = ~np.isnan(depth_reproj)
        mask_invalid = np.isnan(depth_reproj)

        dist = cv2.distanceTransform(
            mask_invalid.astype(np.uint8), cv2.DIST_L2, maskSize=5
        )

        max_radius = 5
        mask_fill = mask_invalid & (dist <= max_radius)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        d = depth_reproj.copy()
        d[np.isnan(d)] = 0.0
        sobelx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)

        grad_mag = np.hypot(sobelx, sobely)

        grad_thresh = 0.08
        mask_depth_edges = grad_mag > grad_thresh

        kernel = np.ones((15, 15), np.uint8)
        mask_depth_edges = cv2.dilate(
            mask_depth_edges.astype(np.uint8), kernel, iterations=1
        ).astype(bool)

        fill_inside_px = 2
        edge_mask_uint8 = mask_depth_edges.astype(np.uint8)
        edge_dist = cv2.distanceTransform(edge_mask_uint8, cv2.DIST_L2, 5)

        mask_fill = mask_invalid & (edge_dist >= fill_inside_px)

        rv, cvv = np.where(mask_valid)
        pts_valid = np.column_stack((cvv, rv))
        vals_valid = depth_reproj[rv, cvv]
        rf, cf = np.where(mask_fill)
        pts_fill = np.column_stack((cf, rf))

        filled_vals = griddata(
            pts_valid, vals_valid, (pts_fill[:, 0], pts_fill[:, 1]), method="nearest"
        )

        depth_filled = depth_reproj.copy()
        ok = ~np.isnan(filled_vals)
        depth_filled[rf[ok], cf[ok]] = filled_vals[ok]

        depth_filled_filtered = filter_depth_with_local_min_scipy(depth_filled)
        valid_pixels = ~np.isnan(depth_filled_filtered)
        vs, us = np.where(valid_pixels)
        Z = depth_filled_filtered[vs, us]

        X = (us - self.rgb_params["cx"]) * Z / self.rgb_params["fx"]
        Y = (vs - self.rgb_params["cy"]) * Z / self.rgb_params["fy"]

        points_rgb = np.stack([X, Y, Z], axis=1)
        colors = rgb_img[vs, us].astype(np.float32) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_rgb)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if not os.path.exists("output"):
            os.makedirs("output")
        o3d.io.write_point_cloud("output/last_point_cloud.ply", pcd)

        print(f"filled: {pcd}")
        depth_img_aligned = self._get_projected_depth_image(
            points_rgb, self.K_rgb, rgb_img.shape[:2]
        )
        return depth_img_aligned

    def get_aligned_depth_img_no_interp(
        self, depth_img: np.ndarray, rgb_img: np.ndarray
    ) -> np.ndarray:
        """Simplified alignment: no interpolation, just nearest-depth reprojection.

        Args:
            depth_img (np.ndarray): Raw depth map.
            rgb_img (np.ndarray): RGB image.

        Returns:
            np.ndarray: Aligned uint16 depth image.
        """
        point_cloud_depth = self.depth_image_to_point_cloud_with_K(
            depth_img, self.K_depth, dist_coeffs=self.depth_params["dist"]
        )

        point_cloud_rgb = self.transform_point_cloud_to_rgb(point_cloud_depth, self.T)

        proj1, filt_depth1, idx, colors = self.project_points_to_pixels_filtered(
            rgb_img, point_cloud_rgb, self.K_rgb, rgb_img.shape[:2]
        )

        filtered_points_rgb = point_cloud_rgb[idx]

        depth_img_aligned = self._get_projected_depth_image(
            filtered_points_rgb, self.K_rgb, rgb_img.shape[:2]
        )

        return depth_img_aligned
