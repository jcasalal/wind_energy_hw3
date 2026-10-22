"""
HW3 - Part 2, Sub-Part 1: Profile VAD  (simple version)

The idea in 4 steps:
  STEP 1. Make a fake wind profile where we KNOW the answer
  STEP 2. Copy it everywhere in a 3D box + add random noise ("kick")
  STEP 3. Pretend to be a lidar: scan in a circle and measure only
          the wind ALONG the beam (radial velocity)
  STEP 4. VAD: from those radial velocities, recover the wind
          and compare to the answer
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(488) # change this number -> unique dataset for your group


# STEP 1: The "true" wind profile (only changes with height)

heights = np.arange(10, 610, 10) # 10 m, 20 m, ... 600 m

# Wind SPEED: it follows log law   =>   speed = (u* / k) * ln(z / z0)
u_star = 0.45 # friction velocity [m/s] (typical range 0.1 - 0.8)
k = 0.41 # von Karman constant 
z0 = 0.05 # surface roughness [m] (0.05 m typical for short grass)
true_speed = (u_star / k) * np.log(heights / z0) 

# Wind DIRECTION: turns slowly with height
dir_bottom = 210   # degrees at the ground (wind coming FROM 210 = SSW)
dir_top    = 235   # degrees at the top   -> 25 deg of turning
true_dir = np.linspace(dir_bottom, dir_top, len(heights))


# Helper: speed + direction  <->  u (east) + v (north)
# (meteorology: direction = where the wind comes FROM)
def speed_dir_to_uv(speed, direction):
    rad = np.radians(direction)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v

def uv_to_speed_dir(u, v):
    speed = np.sqrt(u**2 + v**2)
    direction = np.degrees(np.arctan2(-u, -v)) % 360
    return speed, direction


true_u, true_v = speed_dir_to_uv(true_speed, true_dir)

# =====================================================================
# STEP 2: Fill a 3D box with that profile + 7.5% random kick
# =====================================================================
x = np.arange(-1500, 1501, 50) # east-west positions [m]
y = np.arange(-1500, 1501, 50) # north-south positions [m]
kick = 0.075 # 7.5 % noise

# U[i, j, k] = wind at x[i], y[j], heights[k]
U = np.zeros((len(x), len(y), len(heights)))
V = np.zeros((len(x), len(y), len(heights)))

for kz in range(len(heights)):
    noise_u = kick * true_speed[kz] * np.random.randn(len(x), len(y))
    noise_v = kick * true_speed[kz] * np.random.randn(len(x), len(y))
    U[:, :, kz] = true_u[kz] + noise_u
    V[:, :, kz] = true_v[kz] + noise_v



# STEP 3: Virtual lidar - scan a cone and measure radial velocity

elevation = 60                       # beam angle above the ground [deg]
azimuths  = np.arange(0, 360, 10)    # 36 beams around the circle [deg]
ranges    = np.arange(20, 690, 20)   # distances along the beam [m]

el = np.radians(elevation)
gate_heights = ranges * np.sin(el)   # height of each range gate

# radial[g, a] = what the lidar sees at range gate g, azimuth a
radial = np.zeros((len(ranges), len(azimuths)))

for g in range(len(ranges)):
    for a in range(len(azimuths)):
        az = np.radians(azimuths[a])

        # where is this point in space?
        px = ranges[g] * np.cos(el) * np.sin(az)
        py = ranges[g] * np.cos(el) * np.cos(az)
        pz = ranges[g] * np.sin(el)

        # grab the wind at the nearest grid point
        i  = np.argmin(abs(x - px))
        j  = np.argmin(abs(y - py))
        kz = np.argmin(abs(heights - pz))
        u_here = U[i, j, kz]
        v_here = V[i, j, kz]

        # lidar only sees the part of the wind along its beam
        radial[g, a] = (u_here * np.sin(az) + v_here * np.cos(az)) * np.cos(el)


# =====================================================================
# STEP 4: VAD - fit a sine wave at each height to get the wind back
# =====================================================================
# At one height, radial velocity vs azimuth looks like a sine wave:
#     Vr = A + B*cos(az) + C*sin(az)
#   B = v*cos(el)  ->  v = B / cos(el)
#   C = u*cos(el)  ->  u = C / cos(el)
#   (A would be the vertical wind - zero here)

az_rad = np.radians(azimuths)
M = np.column_stack([np.ones(len(az_rad)), np.cos(az_rad), np.sin(az_rad)])

vad_u = np.zeros(len(ranges))
vad_v = np.zeros(len(ranges))

for g in range(len(ranges)):
    A, B, C = np.linalg.lstsq(M, radial[g, :], rcond=None)[0]   # best-fit sine
    vad_u[g] = C / np.cos(el)
    vad_v[g] = B / np.cos(el)

vad_speed, vad_dir = uv_to_speed_dir(vad_u, vad_v)


# =====================================================================
# RESULTS: plots + error
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

ax[0].plot(true_speed, heights, "k-", label="True (answer)")
ax[0].plot(vad_speed, gate_heights, "ro", label="VAD retrieval")
ax[0].set_xlabel("Wind speed [m/s]")
ax[0].set_ylabel("Height [m]")
ax[0].set_title("Wind speed vs height")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(true_dir, heights, "k-", label="True (answer)")
ax[1].plot(vad_dir, gate_heights, "ro", label="VAD retrieval")
ax[1].set_xlabel("Wind direction [deg]")
ax[1].set_title("Wind direction vs height")
ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vad_result.png", dpi=150)
plt.show()

# How far off are we? (compare at the same heights)
speed_at_gates = np.interp(gate_heights, heights, true_speed)
dir_at_gates   = np.interp(gate_heights, heights, true_dir)
print("Speed error (RMSE):     %.2f m/s" % np.sqrt(np.mean((vad_speed - speed_at_gates)**2)))
print("Direction error (RMSE): %.2f deg" % np.sqrt(np.mean((vad_dir - dir_at_gates)**2)))


# =====================================================================
# Real data values
# =====================================================================

"""
HW3 - Part 2, Sub-Part 1, item 3: VAD on the REAL lidar data (Data_for_VAD.mat)

What's in the file:
  Data = 516 scans (11-Oct-2013 to 23-Oct-2013)
  Each scan has:
     name  -> '20131011_214006' = date_time of the scan
     range -> distance along the beam [m]   (83 gates x ~133 beams)
     az    -> azimuth of each beam [deg]
     el    -> elevation = 75 deg (cone scan)
     rv    -> radial velocity [m/s]  (NaN = bad / no signal)

For every scan we do the SAME VAD fit as in vad_simple.py,
then average all scans to get the mean speed + direction profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)   # hides "mean of empty slice" at heights with no data

# =====================================================================
# STEP 1: Load the file
# =====================================================================
mat = loadmat("Data_for_VAD.mat", squeeze_me=True, struct_as_record=False)
scans = mat["Data"]
print("Number of scans:", len(scans))

# Range gates are the same for every scan -> heights are the same too
ranges  = np.asarray(scans[0].range[:, 0], dtype=float)   # 12, 36, 60 ... 1980 m
elev    = 75.0
heights = ranges * np.sin(np.radians(elev))               # ~12 m ... ~1910 m

MIN_GOOD_BEAMS = 0.5    # need at least 50% of the beams valid at a gate to trust the fit
MAX_FIT_ERROR  = 1.0    # m/s: if the sine fit is worse than this, throw the gate away


# =====================================================================
# STEP 2: VAD fit for ONE scan  (same idea as vad_simple.py)
#   rv = A + B*cos(az) + C*sin(az)
#   u = C / cos(el),   v = B / cos(el),   w = A / sin(el)
# =====================================================================
def vad_one_scan(scan):
    az_all = scan.az
    rv_all = scan.rv
    el = np.radians(np.nanmean(scan.el))

    n_gates = rv_all.shape[0]
    u = np.full(n_gates, np.nan)
    v = np.full(n_gates, np.nan)

    for g in range(n_gates):
        rv = rv_all[g, :]
        az = np.radians(az_all[g, :])
        good = np.isfinite(rv)

        if good.mean() < MIN_GOOD_BEAMS:        # too many missing beams -> skip
            continue

        M = np.column_stack([np.ones(good.sum()), np.cos(az[good]), np.sin(az[good])])
        A, B, C = np.linalg.lstsq(M, rv[good], rcond=None)[0]

        # how well does the sine wave match the data?
        fit_error = np.sqrt(np.mean((M @ [A, B, C] - rv[good])**2))
        if fit_error > MAX_FIT_ERROR:
            continue

        u[g] = C / np.cos(el)
        v[g] = B / np.cos(el)

    return u, v


def uv_to_speed_dir(u, v):
    speed = np.sqrt(u**2 + v**2)
    direction = np.degrees(np.arctan2(-u, -v)) % 360
    return speed, direction


# =====================================================================
# STEP 3: Run VAD on every scan
#   U[t, h] = east wind at scan t, height h   (same for V)
# =====================================================================
n_scans = len(scans)
U = np.full((n_scans, len(heights)), np.nan)
V = np.full((n_scans, len(heights)), np.nan)
times = []

for t in range(n_scans):
    U[t, :], V[t, :] = vad_one_scan(scans[t])
    times.append(datetime.strptime(scans[t].name, "%Y%m%d_%H%M%S"))

SPEED, DIR = uv_to_speed_dir(U, V)

# Save results so the movie / summary scripts can reuse them
np.savez("vad_results.npz", times=np.array(times, dtype="datetime64[s]"),
         heights=heights, U=U, V=V, SPEED=SPEED, DIR=DIR)


# =====================================================================
# STEP 4: Mean profiles over the whole dataset
#   Mean speed:     average of the speeds
#   Mean direction: average u and v FIRST, then convert
#                   (you can't just average angles: avg of 350 and 10 is NOT 180!)
# =====================================================================
n_good = np.sum(np.isfinite(SPEED), axis=0)        # how many scans have data at each height
keep = n_good >= 0.25 * n_scans                    # only show heights with decent data

mean_speed = np.nanmean(SPEED, axis=0)
std_speed  = np.nanstd(SPEED, axis=0)
_, mean_dir = uv_to_speed_dir(np.nanmean(U, axis=0), np.nanmean(V, axis=0))

fig, ax = plt.subplots(1, 3, figsize=(13, 6), sharey=True)

ax[0].plot(mean_speed[keep], heights[keep], "b-o", ms=3)
ax[0].fill_betweenx(heights[keep], (mean_speed - std_speed)[keep],
                    (mean_speed + std_speed)[keep], color="b", alpha=0.15, label="±1 std")
ax[0].set_xlabel("Wind speed [m/s]"); ax[0].set_ylabel("Height [m]")
ax[0].set_title("Mean wind speed"); ax[0].legend()

ax[1].plot(mean_dir[keep], heights[keep], "ro", ms=4)   # dots only: a line would jump across 0/360
ax[1].set_xlabel("Wind direction [deg, from]"); ax[1].set_title("Mean wind direction")
ax[1].set_xlim(0, 360); ax[1].set_xticks([0, 90, 180, 270, 360])

ax[2].plot(100 * n_good / n_scans, heights, "k-")
ax[2].set_xlabel("% of scans with good data"); ax[2].set_title("Data availability")

for a in ax:
    a.grid(alpha=0.3)
plt.suptitle(f"VAD mean profiles: {times[0]:%d-%b-%Y} to {times[-1]:%d-%b-%Y}  ({n_scans} scans)")
plt.tight_layout()
plt.savefig("real_mean_profiles.png", dpi=150)
plt.show()

print("Heights with good data: %.0f m to %.0f m" % (heights[keep].min(), heights[keep].max()))