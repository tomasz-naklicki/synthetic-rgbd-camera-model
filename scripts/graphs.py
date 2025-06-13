import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def angle_graph():
    # --- Parametry kwadratowej rampy ---
    theta_min = 60  # stopnie
    theta0 = 75  # stopnie

    # Przedział kątów
    theta = np.linspace(0, 90, 500)

    # Oblicz P_drop
    p_drop = np.zeros_like(theta)
    mask = theta >= theta_min
    p_drop[mask] = ((theta[mask] - theta_min) / (theta0 - theta_min)) ** 2
    p_drop = np.clip(p_drop, 0, 1)

    # Rysuj
    plt.figure(figsize=(8, 4))
    plt.plot(theta, p_drop, color="C0", lw=2)
    plt.axvline(theta_min, ls="--", color="gray")
    plt.axvline(theta0, ls="--", color="gray")
    plt.text(theta_min, 0.05, r"$\theta_{\min}$", ha="center", va="bottom")
    plt.text(theta0, 0.05, r"$\theta_0$", ha="center", va="bottom")
    plt.title(r"Drop probability based on incident angle $\theta$")
    plt.xlabel(r"Incident Angle $\theta$ [°]")
    plt.ylabel(r"Drop Probability $P_{\mathrm{drop}}$")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def color_graph():
    # --- Parametry 3D-surface ---
    p_max = 0.65
    V_thresh = 0.2  # próg jasności
    theta_thresh = 50  # stopnie

    # Siatka jasność V i kąt theta
    V = np.linspace(0, 1, 100)
    theta = np.linspace(0, 90, 100)
    Vg, Tg = np.meshgrid(V, theta)

    # Waga kątowa (kwadratowa rampa, obcięta do [0,1])
    w_angle = (Tg / theta_thresh) ** 2
    w_angle = np.clip(w_angle, 0, 1)

    # Ostateczne P_drop
    P = p_max * np.exp(-Vg / V_thresh) * w_angle

    # Rysuj powierzchnię 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        Vg, Tg, P, rstride=1, cstride=1, cmap="viridis", edgecolor="none", alpha=0.9
    )

    ax.set_title(r"Drop probability based on pixel brightness and incident angle")
    ax.set_xlabel("Brightness $V$")
    ax.set_ylabel(r"Incident Angle $\theta$ (°)")
    ax.set_zlabel(r"$P_{\mathrm{drop}}$")

    fig.colorbar(surf, pad=0.1, label=r"$P_{\mathrm{drop}}$")
    plt.tight_layout()
    plt.show()


def main():
    angle_graph()
    # color_graph()


if __name__ == "__main__":
    main()
