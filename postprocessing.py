import numpy as np
import cv2
from scipy.ndimage import (
    map_coordinates,
    label,
    find_objects,
    gaussian_filter,
    binary_dilation,
    binary_erosion,
)
from scipy.ndimage import convolve


def add_lateral_noise_depth(Z, fx, px, cx, cy):
    """
    Dodaje lateralny szum do mapy głębi i zwraca 2D obraz Z_noisy.

    Wejście:
      Z   – 2D numpy array (H×W) głębi w metrach
      fx  – ogniskowa kamery w tych samych jednostkach co px (np. metry)
      px  – fizyczny rozmiar piksela (pixel pitch) w metrach
      cx, cy – współrzędne środka obrazu w pikselach

    Wyjście:
      Z_noisy – 2D numpy array (H×W) zaszumionej głębi w metrach
    """
    H, W = Z.shape
    # przygotuj siatkę pikseli
    i, j = np.indices((H, W))

    # promień r i kąt theta
    r = np.sqrt((i - cy) ** 2 + (j - cx) ** 2) * px
    theta = np.arctan(r / fx)

    # odchylenie standardowe szumu w pikselach
    sigma_px = 0.8 + 0.035 * (theta / (np.pi / 2 - theta))

    # losowe przesunięcia w pikselach
    dx = np.random.randn(H, W) * sigma_px
    dy = np.random.randn(H, W) * sigma_px

    # zbuduj tablicę współrzędnych o kształcie (2, H, W)
    coords = np.stack([i + dy, j + dx], axis=0)

    # interpolacja – Z_noisy ma od razu kształt (H, W)
    Z_noisy = map_coordinates(Z, coords, order=1, mode="nearest")
    return Z_noisy


def add_lateral_noise_remap(Z, fx, px, cx, cy, max_delta=0.03):
    """
    Dodaje lateralny szum metodą remap + clamp.

    Z         – 2D numpy array (H×W) głębi w metrach
    fx, px    – ogniskowa i pixel pitch [m]
    cx, cy    – współrzędne środka obrazu [px]
    max_delta – maksymalna dopuszczalna zmiana głębi [m]
    """
    H, W = Z.shape
    # 1) siatka pikseli
    i, j = np.indices((H, W))
    # 2) oblicz kąt theta
    r = np.sqrt((i - cy) ** 2 + (j - cx) ** 2) * px
    theta = np.arctan(r / fx)
    # 3) sigma w pikselach
    sigma_px = 0.8 + 0.035 * (theta / (np.pi / 2 - theta))
    # 4) losowe przesunięcia w pikselach
    dx = np.random.randn(H, W) * sigma_px
    dy = np.random.randn(H, W) * sigma_px
    # 5) remap + interpolacja
    coords = np.stack([i + dy, j + dx], axis=0)
    Z_noisy = map_coordinates(Z, coords, order=1, mode="nearest")
    # 6) clamp: przywróć oryginał, jeśli delta > max_delta
    diff = np.abs(Z_noisy - Z)
    Z_noisy[diff > max_delta] = Z[diff > max_delta]
    return Z_noisy


def add_axial_noise(Z, fx, px, cx, cy):
    """
    Dodaje szum osiowy (axial noise) do mapy głębi.

    Z   – 2D numpy array (H×W) głębi w metrach
    fx  – ogniskowa kamery w tych samych jednostkach co px (np. metry)
    px  – fizyczny rozmiar piksela (pixel pitch) w metrach
    cx, cy – współrzędne środka obrazu w pikselach

    Zwraca:
      Z_noisy – 2D array (H×W) zaszumionej głębi w metrach
    """
    H, W = Z.shape
    # 1) siatka pikseli
    i, j = np.indices((H, W))

    # 2) promień r i kąt theta
    r = np.sqrt((i - cy) ** 2 + (j - cx) ** 2) * px
    theta = np.arctan(r / fx)

    # 3) oblicz σ_z zgodnie z modelem:
    #    podstawowy kwadratowy + hyperboliczny dla dużych kątów
    base = 0.0012 + 0.0019 * (Z - 0.4) ** 2
    hyperb = (
        (0.0001 / np.sqrt(np.clip(Z, 1e-6, None)))
        * (theta**2)
        / ((np.pi / 2 - theta) ** 2)
    )
    sigma_z = base + hyperb

    # 4) dodaj szum ~ N(0, σ_z^2)
    noise = np.random.randn(H, W) * sigma_z
    return Z + noise


def circular_kernel(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


def remove_pixels_above_angle(Z, fx, fy, px, cx, cy, max_angle_deg):
    def depth_to_point_cloud(depth, fx, fy, cx, cy):
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return np.stack((X, Y, Z), axis=-1)  # Shape: (H, W, 3)

    def compute_normals(point_cloud):
        kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8.0
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8.0

        H, W, _ = point_cloud.shape
        dX = np.zeros((H, W, 3))
        dY = np.zeros((H, W, 3))

        for i in range(3):
            dX[..., i] = convolve(point_cloud[..., i], kx, mode="nearest")
            dY[..., i] = convolve(point_cloud[..., i], ky, mode="nearest")

        v1 = dX
        v2 = dY
        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8
        normals = cross / norm
        return normals

    def compute_view_angles(point_cloud, normals):
        view_vectors = -point_cloud
        view_norms = np.linalg.norm(view_vectors, axis=-1, keepdims=True) + 1e-8
        view_dir = view_vectors / view_norms

        # Flip normals to face toward the camera
        dot = np.sum(normals * view_dir, axis=-1, keepdims=True)
        normals = np.where(dot < 0, -normals, normals)

        dot = np.sum(normals * view_dir, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        angles = np.arccos(dot)
        return angles  # Radians

    def depth_image_to_view_angles(depth, fx, fy, cx, cy):
        pc = depth_to_point_cloud(depth, fx, fy, cx, cy)
        normals = compute_normals(pc)
        angles = compute_view_angles(pc, normals)
        return angles

    # smoothing_sigma = (1.0, 1.0)
    # discontinuity_thresh = 0.1
    # # 0) Sprawdź, czy depth jest 2D
    # if Z.ndim != 2:
    #     raise ValueError(f"Depth musi być 2D, a ma ndim={Z.ndim}")

    # # 1) Przygotuj sigma tak, żeby miał długość = depth.ndim
    # sig = np.atleast_1d(smoothing_sigma).astype(float)
    # if sig.size == 1:
    #     sigma = [sig.item()] * Z.ndim
    # elif sig.size == Z.ndim:
    #     sigma = sig.tolist()
    # else:
    #     raise ValueError(
    #         f"smoothing_sigma musi być scalar lub sekwencją długości 1 lub {Z.ndim}, a ma długość {sig.size}"
    #     )

    # # 2) Wygładzanie mapy głębi
    # depth_s = gaussian_filter(Z, sigma=sigma)

    # # 3) Centralne różnice dZ/dx, dZ/dy
    # dzdx = (np.roll(depth_s, -1, axis=1) - np.roll(depth_s, 1, axis=1)) * 0.5
    # dzdy = (np.roll(depth_s, -1, axis=0) - np.roll(depth_s, 1, axis=0)) * 0.5

    # # 4) Oblicz normalne
    # Nx = -fx * dzdx
    # Ny = -fy * dzdy
    # Nz = np.ones_like(depth_s)
    # N = np.stack((Nx, Ny, Nz), axis=-1)
    # N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-8

    # # 5) Back-projection i wektor promienia
    # h, w = Z.shape
    # u = np.arange(w)
    # v = np.arange(h)[:, None]
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = Z
    # P = np.stack((X, Y, Z), axis=-1)
    # D = P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-8)

    # # 6) Kąt padania
    # cos_theta = np.sum(N * D, axis=2)
    # cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # theta = np.arccos(cos_theta)
    H, W = Z.shape
    # i, j = np.indices((H, W))

    # x = (j - cx) * Z * px
    # y = (i - cy) * Z * px
    # z = Z

    # norm = np.sqrt(x**2 + y**2 + z**2)
    # cos_theta = z / np.clip(norm, 1e-6, None)
    # theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    theta = depth_image_to_view_angles(Z, fx, fy, cx, cy)

    max_angle_rad = np.radians(max_angle_deg)
    over_threshold = theta > max_angle_rad

    smoothed = gaussian_filter(over_threshold.astype(np.float32), sigma=10)
    mask = smoothed > 0.02
    expanded = binary_erosion(mask, iterations=3)
    clustered = binary_dilation(expanded, iterations=10)

    theta_max = np.max(theta)
    angle_diff = theta - max_angle_rad
    angle_diff[angle_diff < 0] = 0
    prob_map = np.clip(angle_diff / (theta_max - max_angle_rad), 0.5, 1.0)

    random_vals = np.random.rand(H, W)
    drop_mask = (random_vals < prob_map) & clustered

    Z_filtered = Z.copy()
    Z_filtered[drop_mask] = np.nan

    # Post-pass cleanup using circular kernel
    valid_mask = ~np.isnan(Z_filtered)
    kernel = circular_kernel(radius=3)  # radius = 3 → 7x7 disk
    neighbor_count = convolve(
        valid_mask.astype(np.uint8), kernel, mode="constant", cval=0
    )
    Z_filtered[(valid_mask) & (neighbor_count <= 7)] = np.nan

    return Z_filtered


def circular_kernel(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


def remove_pixels_above_angle2(
    Z,
    fx,
    fy,
    px,
    cx,
    cy,
    max_angle_deg=60.0,
    smoothing_sigma=1.5,
    discontinuity_thresh=0.05,
    normal_edge_thresh=0.1,
    edge_dilate_iters=8,
    morph_sigma=2.0,
    erode_iters=1,
    dilate_iters=2,
    min_prob=0.8,
    gamma=0.5,
):
    """
    Agresywne usuwanie tła wg kąta padania,
    przy jednoczesnej ochronie wszystkich krawędzi obiektów
    wykrytych na podstawie głębi i normalnych.
    """
    H, W = Z.shape
    i, j = np.indices((H, W))

    # 1) Wygładzimy głębię, by normalne były stabilne
    Zs = gaussian_filter(Z, sigma=smoothing_sigma)

    # 2) Obliczamy chmurę punktów do normalnych
    pc_s = np.stack([(j - cx) * Zs / fx, (i - cy) * Zs / fy, Zs], axis=-1)

    # 3) Sobel → dX, dY → normalne N
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32) / 8.0
    ky = kx.T
    dX = np.stack(
        [convolve(pc_s[..., k], kx, mode="nearest") for k in range(3)], axis=-1
    )
    dY = np.stack(
        [convolve(pc_s[..., k], ky, mode="nearest") for k in range(3)], axis=-1
    )
    N = np.cross(dX, dY)
    N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-8

    # 4) Obliczamy kąty padania θ
    pc = np.stack([(j - cx) * Z / fx, (i - cy) * Z / fy, Z], axis=-1)
    V = -pc
    V /= np.linalg.norm(V, axis=2, keepdims=True) + 1e-8
    dot = np.sum(N * V, axis=2, keepdims=True)
    N = np.where(dot < 0, -N, N)
    cos_t = np.clip(np.sum(N * V, axis=2), -1.0, 1.0)
    theta = np.arccos(cos_t)

    # 5) Maska ostrych skoków głębi
    dzdx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) * 0.5
    dzdy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) * 0.5
    mask_disc = (np.abs(dzdx) > discontinuity_thresh) | (
        np.abs(dzdy) > discontinuity_thresh
    )

    # 6) Maska skoków w normalnych (kontury obiektów)
    ndx = N - np.roll(N, 1, axis=1)
    ndy = N - np.roll(N, 1, axis=0)
    grad_norm = np.linalg.norm(np.concatenate([ndx, ndy], axis=-1), axis=2)
    mask_norm = grad_norm > normal_edge_thresh

    # 7) Scalamy obie maski i rozszerzamy
    mask_edges = mask_disc | mask_norm
    mask_edges = binary_dilation(mask_edges, iterations=edge_dilate_iters)

    # 8) Budujemy probabilistyczną mapę usuwania tła
    max_rad = np.deg2rad(max_angle_deg)
    diff = theta - max_rad
    diff[diff < 0] = 0
    max_diff = theta.max() - max_rad + 1e-8
    base_prob = diff.squeeze() / max_diff
    prob_map = min_prob + (1 - min_prob) * (base_prob**gamma)
    prob_map = np.clip(prob_map, 0, 1)

    # 9) Morfologiczna klasteryzacja tła
    bg = diff.squeeze() > 0
    cluster = gaussian_filter(bg.astype(float), sigma=morph_sigma) > 0.5
    cluster = binary_erosion(cluster, iterations=erode_iters)
    cluster = binary_dilation(cluster, iterations=dilate_iters)

    # 10) Losowe usuwanie tylko w klastrach tła poza maską krawędzi
    rnd = np.random.rand(H, W)
    drop = (rnd < prob_map) & cluster & (~mask_edges)

    Zf = Z.copy()
    Zf[drop] = np.nan

    # 11) Sprzątanie izolowanych pikseli
    valid = ~np.isnan(Zf)
    kernel = circular_kernel(radius=3)
    neigh = convolve(valid.astype(np.uint8), kernel, mode="constant", cval=0)
    Zf[(valid) & (neigh <= 7)] = np.nan

    return Zf


def compute_incidence_with_random_distance(
    depth,
    fx,
    fy,
    cx,
    cy,
    smoothing_sigma=1.0,
    theta0_deg=55.0,
    slope=50.0,
    depth_min=0.5,
    seed=None,
):
    """
    Maskuje piksele z losowością zależną od kąta padania i odległości.

    Parameters
    ----------
    depth : 2D float array
        Mapa głębi w metrach.
    fx, fy, cx, cy : float
        Parametry intrinsics.
    smoothing_sigma : float
        Sigma Gaussa dla wygładzenia głębi.
    theta0_deg : float
        Kąt [°], przy którym P(return)=0.5.
    slope : float
        Stromość przejścia w funkcji logistycznej.
    depth_min : float
        Minimalna odległość [m]; dla depth < depth_min P(return)=1.
    seed : int or None
        Ziarno generatora losowego.

    Returns
    -------
    depth_out : 2D float array
        Mapa głębi z NaN tam, gdzie pomiar “nie wraca”.
    theta : 2D float array
        Mapa kątów padania [rad].
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = depth.shape

    # 1) Wygładzenie głębi + centralne różnice → normalne N
    depth_s = gaussian_filter(depth, sigma=smoothing_sigma)
    dzdx = (np.roll(depth_s, -1, axis=1) - np.roll(depth_s, 1, axis=1)) * 0.5
    dzdy = (np.roll(depth_s, -1, axis=0) - np.roll(depth_s, 1, axis=0)) * 0.5
    Nx = -dzdx * fx
    Ny = -dzdy * fy
    Nz = np.ones_like(depth_s)
    N = np.stack((Nx, Ny, Nz), axis=-1)
    N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-8

    # 2) Wektor promienia D
    u = np.arange(w)
    v = np.arange(h)[:, None]
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    P = np.stack((X, Y, depth), axis=-1)
    D = P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-8)

    # 3) Kąt padania θ
    cos_t = N[:, :, 2]  # dot product with [0, 0, 1]
    theta = np.arccos(cos_t)  # w radianach
    # 4) Funkcja logistyczna P(return | θ)
    theta0 = np.deg2rad(theta0_deg)
    p_ret = 1.0 / (1.0 + np.exp(slope * (theta - theta0)))
    cv2.imwrite("rendered/angles.png", np.rad2deg(theta))
    # 5) Dla punktów bliżej niż depth_min: zawsze wraca
    p_ret[depth < depth_min] = 1.0
    p_ret[abs(theta - 90) < 4.0] = 1.0

    # 6) Monte Carlo: jeśli u > p_ret → brak pomiaru
    u_rand = np.random.rand(h, w)
    mask_drop = u_rand > p_ret

    # 7) Zastosuj maskę
    depth_out = depth.copy()
    depth_out[mask_drop] = np.nan

    return depth_out, theta


def main():
    # 1) Wczytaj:
    Z_img = cv2.imread("chybadobre/render_with_compositing3.png", cv2.IMREAD_UNCHANGED)
    depth_calib = {
        "resolution": (1024, 1024),
        "fx": 504.752167,
        "fy": 504.695465,
        "cx": 517.601746,
        "cy": 508.529358,
    }
    W, H = depth_calib["resolution"]
    fx, fy = depth_calib["fx"], depth_calib["fy"]
    cx, cy = (
        depth_calib["cx"],
        depth_calib["cy"],
    )  # fx, px, cx, cy – znane parametry kamery Kinect
    px = 0.0035

    # 2) Zamień na metry:
    if Z_img.dtype == np.uint16:
        Z = Z_img.astype(np.float32) / 1000.0  # z mm → metry
    else:
        Z = Z_img.astype(np.float32)  # zakładamy już w metrach
    Z[Z == 65.535] = 0.0
    # 3) Dodaj szum
    Z_angles, theta = compute_incidence_with_random_distance(
        Z, fx, fy, cx, cy, theta0_deg=75, depth_min=1.0
    )
    Z_noisy = add_lateral_noise_remap(Z_angles, fx, px, cx, cy)
    Z_final = add_axial_noise(Z_noisy, fx, px, cx, cy)
    # Z_final = remove_pixels_above_angle(Z, fx, fy, px, cx, cy, max_angle_deg=60.0)
    # Z_final = remove_pixels_above_angle2(
    #     Z,
    #     fx,
    #     fy,
    #     px,
    #     cx,
    #     cy,
    #     max_angle_deg=60.0,
    #     smoothing_sigma=1.5,
    #     discontinuity_thresh=0.05,
    #     normal_edge_thresh=0.1,  # czułość na kontury obiektów
    #     edge_dilate_iters=8,
    #     morph_sigma=2.0,
    #     erode_iters=1,
    #     dilate_iters=2,
    #     min_prob=0.9,
    #     gamma=0.5,
    # )
    # Z_final, theta = compute_incidence_jagged(
    #     Z_noisy_2, fx, fy, cx, cy, theta0_deg=70, depth_min=3, angle_jitter_deg=10
    # )
    # 4) Zapisz jako 16-bit mm:
    Zmm = np.where(np.isnan(Z_final), 0.0, Z_final)
    Zmm2 = np.clip(Zmm * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite("rendered/render_with_compositing_0_noise.png", Zmm2)
    print(f"Zapisano zaszumiony obraz")


if __name__ == "__main__":
    main()
