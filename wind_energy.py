"""
HW3 - Part 2, Sub-Part 1: Profile VAD  (STARTER)
------------------------------------------------
Pipeline:
  1. Build a synthetic 3D wind field (log-law speed + Ekman-like veering + random kick)
  2. "Virtual lidar": sample the field on a conical PPI scan -> radial velocities
  3. VAD: fit a sinusoid per range gate -> u, v, w vs height
  4. Compare retrieval to the known "answer"

TODO (later):
  - Load the real MATLAB dataset (HW_3_LiDAR_data) and run vad_fit() on it
  - Mean profiles, movies, error metrics, met summary
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ------------------------------------------------------------------
# 0) PARAMETERS  -> change these to make your group's dataset unique
# ------------------------------------------------------------------
SEED      = 488          # random seed (unique to your group)
U_STAR    = 0.45         # friction velocity [m/s]
Z0        = 0.05         # roughness length [m]
KAPPA     = 0.41         # von Karman constant
DIR_SFC   = 210.0        # wind direction at surface [deg, meteorological "from"]
VEER_TOT  = 25.0         # total clockwise veer from surface to top [deg] (Ekman-like)
KICK      = 0.075        # random kick: 7.5% of local speed (within 5-10%)

# Grid
X = np.linspace(-1500, 1500, 61)   # m
Y = np.linspace(-1500, 1500, 61)   # m
Z = np.linspace(10, 600, 60)       # m (avoid z=0 for log law)

# Lidar scan
ELEV_DEG  = 60.0                     # elevation angle of the PPI cone
AZ_DEG    = np.arange(0, 360, 10)    # azimuths (36 beams)
RANGES    = np.arange(30, 650, 25)   # range gates along beam [m]

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------
# 1) SYNTHETIC WIND FIELD
# ------------------------------------------------------------------
def true_profile(z):
    """Exact profile: speed (log law) and direction (linear veer)."""
    speed = (U_STAR / KAPPA) * np.log(z / Z0)
    wdir = DIR_SFC + VEER_TOT * (z - z.min()) / (z.max() - z.min())
    return speed, wdir


def dir_to_uv(speed, wdir_deg):
    """Meteorological convention: direction wind comes FROM, clockwise from N."""
    d = np.deg2rad(wdir_deg)
    return -speed * np.sin(d), -speed * np.cos(d)


def uv_to_dir(u, v):
    return np.mod(np.rad2deg(np.arctan2(-u, -v)), 360)


def build_field():
    speed, wdir = true_profile(Z)
    u_prof, v_prof = dir_to_uv(speed, wdir)

    # copy the single profile to every (x, y) -> shape (nx, ny, nz)
    shape = (len(X), len(Y), len(Z))
    u = np.broadcast_to(u_prof, shape).copy()
    v = np.broadcast_to(v_prof, shape).copy()
    w = np.zeros(shape)

    # random kick, proportional to local speed
    spd = np.hypot(u, v)
    u += KICK * spd * rng.standard_normal(shape)
    v += KICK * spd * rng.standard_normal(shape)
    return u, v, w, u_prof, v_prof


# ------------------------------------------------------------------
# 2) VIRTUAL LIDAR (conical PPI)
# ------------------------------------------------------------------
def virtual_lidar(u, v, w):
    """Returns radial velocity Vr[gate, azimuth] and gate heights."""
    interp = [RegularGridInterpolator((X, Y, Z), f) for f in (u, v, w)]
    el = np.deg2rad(ELEV_DEG)
    az = np.deg2rad(AZ_DEG)

    R, A = np.meshgrid(RANGES, az, indexing="ij")       # (ngate, naz)
    xs = R * np.cos(el) * np.sin(A)
    ys = R * np.cos(el) * np.cos(A)
    zs = R * np.sin(el)
    pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    ui, vi, wi = (f(pts).reshape(R.shape) for f in interp)
    vr = ui * np.cos(el) * np.sin(A) + vi * np.cos(el) * np.cos(A) + wi * np.sin(el)
    return vr, RANGES * np.sin(el)


# ------------------------------------------------------------------
# 3) VAD RETRIEVAL
# ------------------------------------------------------------------
def vad_fit(vr, az_deg, elev_deg):
    """
    Least-squares fit per gate:  Vr = a0 + a1*cos(az) + b1*sin(az)
      u = b1/cos(el),  v = a1/cos(el),  w = a0/sin(el)
    vr: (ngate, naz), NaNs allowed.
    """
    el = np.deg2rad(elev_deg)
    az = np.deg2rad(az_deg)
    G = np.column_stack([np.ones_like(az), np.cos(az), np.sin(az)])

    n = vr.shape[0]
    u, v, w = (np.full(n, np.nan) for _ in range(3))
    for i in range(n):
        ok = np.isfinite(vr[i])
        if ok.sum() < 6:                      # need enough beams for a fit
            continue
        a0, a1, b1 = np.linalg.lstsq(G[ok], vr[i, ok], rcond=None)[0]
        u[i], v[i], w[i] = b1 / np.cos(el), a1 / np.cos(el), a0 / np.sin(el)
    return u, v, w


# ------------------------------------------------------------------
# 4) RUN + COMPARE
# ------------------------------------------------------------------
if __name__ == "__main__":
    u, v, w, u_true, v_true = build_field()
    vr, z_gate = virtual_lidar(u, v, w)
    u_ret, v_ret, w_ret = vad_fit(vr, AZ_DEG, ELEV_DEG)

    spd_true, dir_true = np.hypot(u_true, v_true), uv_to_dir(u_true, v_true)
    spd_ret, dir_ret = np.hypot(u_ret, v_ret), uv_to_dir(u_ret, v_ret)

    fig, ax = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    # (A) horizontal vectors vs height
    ax[0].quiver(np.zeros_like(Z[::4]), Z[::4], u_true[::4], v_true[::4],
                 color="k", scale=60, width=0.006, label="truth")
    ax[0].quiver(np.full_like(z_gate, 1), z_gate, u_ret, v_ret,
                 color="tab:red", scale=60, width=0.006, label="VAD")
    ax[0].set_xlim(-1, 2); ax[0].set_xticks([0, 1], ["truth", "VAD"])
    ax[0].set_ylabel("Height [m]"); ax[0].set_title("Horizontal vectors")

    # speed
    ax[1].plot(spd_true, Z, "k-", label="truth")
    ax[1].plot(spd_ret, z_gate, "o", color="tab:red", ms=4, label="VAD")
    ax[1].set_xlabel("Speed [m/s]"); ax[1].set_title("Wind speed"); ax[1].legend()

    # (B) direction
    ax[2].plot(dir_true, Z, "k-", label="truth")
    ax[2].plot(dir_ret, z_gate, "o", color="tab:red", ms=4, label="VAD")
    ax[2].set_xlabel("Direction [deg from N]"); ax[2].set_title("Wind direction")
    ax[2].legend()

    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("synthetic_vad_check.png", dpi=150)
    plt.show()

    # quick error check (start of the "how did you do" section)
    st, _ = true_profile(z_gate)
    _, dt = true_profile(z_gate)
    print(f"Speed RMSE:     {np.sqrt(np.nanmean((spd_ret - st)**2)):.3f} m/s")
    print(f"Direction RMSE: {np.sqrt(np.nanmean(((dir_ret - dt + 180) % 360 - 180)**2)):.2f} deg")

    # ------------------------------------------------------------------
    # TODO: real data
    # from scipy.io import loadmat
    # d = loadmat("HW_3_LiDAR_data.mat")
    # print(d.keys())   # inspect variable names, then reshape vr to (gate, az)
    # ------------------------------------------------------------------