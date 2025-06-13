import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from src.projection import filter_depth_with_local_min_scipy


def circular_kernel(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


class PreprocessingManager:
    def __init__(self, params: dict):
        self.rgb_params = params["rgb"]
        self.depth_params = params["depth"]
        self.T = params["T"]

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

    def _compute_drop_prob_from_angle(
        self,
        depth,
        smoothing_sigma=1.0,
        theta0_deg=75.0,
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
        Nx = -dzdx * self.depth_params["fx"]
        Ny = -dzdy * self.depth_params["fy"]
        Nz = np.ones_like(depth_s)
        N = np.stack((Nx, Ny, Nz), axis=-1)
        N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-8

        # 2) Wektor promienia D
        u = np.arange(w)
        v = np.arange(h)[:, None]
        X = (u - self.depth_params["cx"]) * depth / self.depth_params["fx"]
        Y = (v - self.depth_params["cy"]) * depth / self.depth_params["fy"]
        P = np.stack((X, Y, depth), axis=-1)
        D = P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-8)

        # 3) Kąt padania θ
        cos_t = N[:, :, 2]  # dot product with [0, 0, 1]
        theta = np.arccos(cos_t)  # w radianach
        # 4) Funkcja logistyczna P(return | θ)
        theta0 = np.deg2rad(theta0_deg)

        # p_drop = 1.0 - 1.0 / (1.0 + np.exp(slope * (theta - theta0)))
        def p_drop_quad_thresh(theta, theta0_deg=75.0, theta_min_deg=30.0):
            """
            Kwadratowa rampa z progiem minimalnym:
            - dla θ < θ_min p_drop = 0
            - dla θ ≥ θ_min:
                p_drop = ((θ - θ_min)/(θ0 - θ_min))^2, obcięte do max 1

            Parametry
            ---------
            theta         : float or np.ndarray
                Kąt padania w radianach.
            theta0_deg    : float
                Kąt (w stopniach), przy którym p_drop osiąga 1.
            theta_min_deg : float
                Kąt (w stopniach) od którego zaczyna się wzrost p_drop.

            Zwraca
            ------
            p_drop : float or np.ndarray
                Prawdopodobieństwo odrzucenia w [0,1].
            """
            # konwersja progów do radianów
            theta0 = np.deg2rad(theta0_deg)
            theta_min = np.deg2rad(theta_min_deg)
            # oblicz rampę tylko tam, gdzie theta ≥ theta_min
            ramp = (theta - theta_min) / (theta0 - theta_min)
            # kwadrat z obcięciem
            p = np.clip(ramp**2, 0.0, 1.0)
            # zerujemy poniżej progu
            p = np.where(theta < theta_min, 0.0, p)
            return p

        p_drop = p_drop_quad_thresh(theta, theta0_deg=75.0, theta_min_deg=60.0)
        # 5) Dla punktów bliżej niż depth_min: zawsze wraca
        # p_drop[depth < depth_min] = 0.0
        p_drop[abs(theta - 90) < 4.0] = 0.0
        return p_drop, theta

    # def _compute_drop_prob_from_color(
    #     self,
    #     depth_img: np.ndarray,
    #     rgb_img: np.ndarray,
    #     theta: np.ndarray,
    #     cutoff_v: float = 0.2,
    #     angle_thresh: float = 50.0,
    # ):
    #     colored_depth = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    #     value = colored_depth[..., 2] / 255.0  # Normalize V ∈ [0, 1]

    #     brightness_mask = value < cutoff_v
    #     angle_mask = theta > np.deg2rad(angle_thresh)
    #     combined_mask = brightness_mask & angle_mask

    #     def p_drop_logistic_b(v, cutoff_v=0.2, k=0.05, max_prob=0.65):
    #         """
    #         Funkcja logistyczna wokół cutoff_v:
    #         max_prob / (1 + exp((v−cutoff_v)/k))
    #         """
    #         return max_prob / (1.0 + np.exp((v - cutoff_v) / k))

    #     # prob = (1.0 - (value / cutoff_v)) * 0.65
    #     prob = p_drop_logistic_b(value, cutoff_v)
    #     drop_prob = np.zeros_like(value, dtype=np.float32)
    #     drop_prob[combined_mask] = prob[combined_mask]

    #     return drop_prob

    def _compute_drop_prob_from_color(
        self,
        depth_img: np.ndarray,
        rgb_img: np.ndarray,
        theta: np.ndarray,
        cutoff_v: float = 0.2,
        angle_thresh: float = 50.0,
        angle_exponent: float = 2.0,
        max_prob: float = 0.65,
    ):
        """
        Ciągłe maskowanie na podstawie jasności i kąta, bez ostrego progu kąta.

        Zamiast binary maski θ>angle_thresh, bierzemy:
        w_angle(θ) = clip( (θ/θ_thresh)^angle_exponent , 0, 1 )

        Parametry:
        ---------
        cutoff_v        – próg jasności V w HSV, poniżej którego rośnie p_drop
        angle_thresh    – kąt w stopniach, przy którym w_angle osiąga 1
        angle_exponent  – wykładnik rampy kąta (>1: strome zbocze blisko θ_thresh)
        max_prob        – maksymalne prawdopodobieństwo “drop”
        """

        # 1) Jasność V
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        v = hsv[..., 2] / 255.0  # [0,1]

        # 2) p_brightness: logistyka wokół cutoff_v
        def p_brightness(v):
            return max_prob * np.exp(-v / cutoff_v)

        # 3) w_angle: gładka „rampa” kąta
        #    - normalizujemy θ do [0,1] przez /θ_thresh, potem ^angle_exponent
        #    - obcinamy, by ≥1 dawać 1
        angle_thresh_rad = np.deg2rad(angle_thresh)

        def w_angle(theta):
            ramp = (theta / angle_thresh_rad) ** angle_exponent
            return np.clip(ramp, 0.0, 1.0)

        # 4) Finalne p_drop jako iloczyn
        drop_prob = (p_brightness(v) * w_angle(theta)).astype(np.float32)
        return drop_prob

    def _drop_pixels(self, depth_img: np.ndarray, rgb_img: np.ndarray):
        p_drop_angle, theta = self._compute_drop_prob_from_angle(depth_img)
        p_drop_color = self._compute_drop_prob_from_color(depth_img, rgb_img, theta)
        h, w = depth_img.shape
        p_drop = p_drop_angle + p_drop_color

        u_rand = np.random.rand(h, w)
        mask_drop = u_rand < p_drop

        # 7) Zastosuj maskę
        depth_out = depth_img.copy()
        depth_out[mask_drop] = np.nan
        return depth_out

    def get_processed_image(self, depth_img: np.ndarray, rgb_img: np.ndarray):
        Z_img = depth_img
        colored_depth = self._get_colors_for_depth(depth_img, rgb_img)
        fx, fy = self.depth_params["fx"], self.depth_params["fy"]
        cx, cy = (
            self.depth_params["cx"],
            self.depth_params["cy"],
        )
        px = self.depth_params["px"]

        if Z_img.dtype == np.uint16:
            Z = Z_img.astype(np.float32) / 1000.0  # mm -> m
        else:
            Z = Z_img.astype(np.float32)  # expected to be meters

        # Fill max value spots with 0.0
        Z[Z == 65.535] = 0.0
        # Remove pixels above certain angle
        Z_angles = self._drop_pixels(Z, colored_depth)

        # Add noise
        Z_noisy = self._add_lateral_noise_remap(Z_angles, fx, px, cx, cy)
        Z_final = self._add_axial_noise(Z_noisy, fx, px, cx, cy)

        # Convert back to mm
        Zmm = np.where(np.isnan(Z_final), 0.0, Z_final)
        Zmm2 = np.clip(Zmm * 1000.0, 0, 65535).astype(np.uint16)

        return Zmm2

    def _get_colors_for_depth(
        self, depth_img: np.ndarray, rgb_img: np.ndarray, depth_scale=1000.0
    ):
        H_d, W_d = depth_img.shape
        H_r, W_r = rgb_img.shape[:2]
        K_rgb = np.array(
            [
                [self.rgb_params["fx"], 0, self.rgb_params["cx"]],
                [0, self.rgb_params["fy"], self.rgb_params["cy"]],
                [0, 0, 1],
            ]
        )
        K_depth = np.array(
            [
                [self.depth_params["fx"], 0, self.depth_params["cx"]],
                [0, self.depth_params["fy"], self.depth_params["cy"]],
                [0, 0, 1],
            ]
        )

        # 0. resize & scale intrinsics
        rgb_resized = cv2.resize(rgb_img, (W_d, H_d), interpolation=cv2.INTER_LINEAR)
        scale_x, scale_y = W_d / W_r, H_d / H_r
        Kd = K_depth
        Kr = K_rgb.copy()
        Kr[0, 0] *= scale_x
        Kr[0, 2] *= scale_x
        Kr[1, 1] *= scale_y
        Kr[1, 2] *= scale_y

        # 1. przygotuj output i punkty głębi
        out = np.zeros((H_d, W_d, 3), dtype=rgb_resized.dtype)
        vs, us = np.where(depth_img > 0)
        if not len(us):
            return out

        Z = depth_img[vs, us].astype(np.float32) / depth_scale
        Z = filter_depth_with_local_min_scipy(Z, kernel_size=49)
        pts = np.stack([us, vs], axis=1).reshape(-1, 1, 2).astype(np.float32)

        # 2. undistortPoints → normalized coords
        und = cv2.undistortPoints(
            pts, Kd, np.array(self.depth_params["dist"], np.float32), P=None
        )
        und = und.reshape(-1, 2)
        x_norm, y_norm = und[:, 0], und[:, 1]

        # 3. back-project & transform
        X = x_norm * Z
        Y = y_norm * Z
        ones = np.ones_like(Z)
        pts_d_h = np.stack([X, Y, Z, ones], axis=1).T
        pts_c_h = (self.T @ pts_d_h).T
        Xc, Yc, Zc = pts_c_h[:, 0], pts_c_h[:, 1], pts_c_h[:, 2]

        # 4. project to rgb
        u_c = np.round((Kr[0, 0] * Xc / Zc) + Kr[0, 2]).astype(int)
        v_c = np.round((Kr[1, 1] * Yc / Zc) + Kr[1, 2]).astype(int)

        # 5. mask & fill
        valid = (u_c >= 0) & (u_c < W_d) & (v_c >= 0) & (v_c < H_d) & (Zc > 0)
        out[vs[valid], us[valid]] = rgb_resized[v_c[valid], u_c[valid]]
        return out
