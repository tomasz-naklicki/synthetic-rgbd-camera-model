import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


def circular_kernel(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


class PreprocessingManager:
    def __init__(self, params: dict):
        self.params_rgb = params["rgb"]
        self.params_depth = params["depth"]

    def _add_lateral_noise_remap(self, Z, fx, px, cx, cy, max_delta=0.03):
        """
        Dodaje lateralny szum metodą remap + clamp.

        Z         - 2D numpy array (HxW) głębi w metrach
        fx, px    - ogniskowa i pixel pitch [m]
        cx, cy    - współrzędne środka obrazu [px]
        max_delta - maksymalna dopuszczalna zmiana głębi [m]
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

    def _add_axial_noise(self, Z, fx, px, cx, cy):
        """
        Dodaje szum osiowy (axial noise) do mapy głębi.

        Z   - 2D numpy array (HxW) głębi w metrach
        fx  - ogniskowa kamery w tych samych jednostkach co px (np. metry)
        px  - fizyczny rozmiar piksela (pixel pitch) w metrach
        cx, cy - współrzędne środka obrazu w pikselach

        Zwraca:
        Z_noisy - 2D array (HxW) zaszumionej głębi w metrach
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

    def _compute_incidence_with_random_distance(
        self,
        depth,
        fx,
        fy,
        cx,
        cy,
        smoothing_sigma=1.0,
        theta0_deg=75.0,
        slope=50.0,
        depth_min=1.0,
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

    def get_processed_image(self, depth_image: np.ndarray):
        Z_img = depth_image
        W, H = self.params_depth["resolution"]
        fx, fy = self.params_depth["fx"], self.params_depth["fy"]
        cx, cy = (
            self.params_depth["cx"],
            self.params_depth["cy"],
        )
        px = self.params_depth["px"]

        if Z_img.dtype == np.uint16:
            Z = Z_img.astype(np.float32) / 1000.0  # mm -> m
        else:
            Z = Z_img.astype(np.float32)  # expected to be meters

        # Fill max value spots with 0.0
        Z[Z == 65.535] = 0.0
        # Remove pixels above certain angle
        Z_angles, theta = self._compute_incidence_with_random_distance(
            Z, fx, fy, cx, cy
        )

        # Add noise
        Z_noisy = self._add_lateral_noise_remap(Z_angles, fx, px, cx, cy)
        Z_final = self._add_axial_noise(Z_noisy, fx, px, cx, cy)

        # Convert back to mm
        Zmm = np.where(np.isnan(Z_final), 0.0, Z_final)
        Zmm2 = np.clip(Zmm * 1000.0, 0, 65535).astype(np.uint16)

        return Zmm2
