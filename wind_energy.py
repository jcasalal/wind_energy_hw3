"""
HW3 - Part 2, Sub-Part 1: Profile VAD 

Steps of the process:
  1. Make a fake wind profile where we KNOW the answer
  2. Copy it everywhere in a 3D box + add random noise ("kick")
  3. Pretend to be a lidar: scan in a circle and measure only the wind ALONG the beam (radial velocity)
  4. VAD: from those radial velocities, recover the wind and compare to the answer
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

# STEP 2: Fill a 3D box with that profile + 7.5% random kick
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


# STEP 4: VAD - fit a sine wave at each height to get the wind back

# At one height, radial velocity vs azimuth looks like a sine wave:
#   Vr = A + B*cos(az) + C*sin(az)
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

# RESULTS: plots + error

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


# ITEM 2A: horizontal velocity vectors vs height

every = 3
h_true_plot = heights[::every]
u_true_plot = true_u[::every]
v_true_plot = true_v[::every]

# arrow length scale: 20 m/s = 1 inch long arrow (so 10 m/s = half inch)
# using the same number on both panels so the arrow lengths can be compared
ARROW_SCALE = 20

# one figure, 3 panels side by side (figsize is width, height in inches)
# not using sharey since panel 3's y axis is v, not height
fig_vec, ax_vec = plt.subplots(1, 3, figsize=(14, 7))

# panel 1: true vectors vs height (the answer)

# quiver(x, y, u, v) = python's version of matlab quiver, draws arrows
# all arrows go at x = 0 (np.zeros_like just makes a zeros array the same
# size as h_true_plot) and y = their height
#   angles="uv"     -> arrow direction only comes from u and v, so the
#                      height axis doesn't stretch/distort the angle
#   pivot="middle"  -> centers the arrow on its height point
# saving it as q_true so i can add the scale arrow after
q_true = ax_vec[0].quiver(np.zeros_like(h_true_plot), h_true_plot,
                          u_true_plot, v_true_plot,
                          angles="uv", scale_units="inches", scale=ARROW_SCALE,
                          pivot="middle", color="k", width=0.006)

# reference arrow so you can tell how fast the arrows are
# (0.8, 0.03) = position as a fraction of the panel (bottom right corner)
# 10 = the speed the reference arrow stands for, labelpos="N" puts text on top
ax_vec[0].quiverkey(q_true, 0.8, 0.03, 10, "10 m/s",
                    labelpos="N", coordinates="axes")

# x position doesn't mean anything here (everything's at 0) so hide the ticks
ax_vec[0].set_xlim(-1, 1)
ax_vec[0].set_ylim(0, 620)
ax_vec[0].set_xticks([])
ax_vec[0].set_ylabel("Height [m]")
ax_vec[0].set_title("True vectors (the answer)\nup = north, right = east")
ax_vec[0].grid(alpha=0.3)


# panel 2: VAD retrieved vectors vs height
# same thing as panel 1 but with the VAD output (vad_u, vad_v)
# these are at the range gate heights (gate_heights), about every 17 m,
# so every 2nd gate to keep the arrows from overlapping
q_vad = ax_vec[1].quiver(np.zeros_like(gate_heights[::2]), gate_heights[::2],
                         vad_u[::2], vad_v[::2],
                         angles="uv", scale_units="inches", scale=ARROW_SCALE,
                         pivot="middle", color="r", width=0.006)

# same 10 m/s reference arrow as panel 1
ax_vec[1].quiverkey(q_vad, 0.8, 0.03, 10, "10 m/s",
                    labelpos="N", coordinates="axes")

ax_vec[1].set_xlim(-1, 1)
ax_vec[1].set_ylim(0, 620)          # same height range as panel 1 to compare
ax_vec[1].set_xticks([])
ax_vec[1].set_ylabel("Height [m]")
ax_vec[1].set_title("VAD retrieval (virtual lidar)\nup = north, right = east")
ax_vec[1].grid(alpha=0.3)


# panel 3: hodograph

# true profile = black line through all 60 heights
ax_vec[2].plot(true_u, true_v, "k-", label="True (answer)")

# VAD = red dots, one per range gate
ax_vec[2].plot(vad_u, vad_v, "ro", ms=4, label="VAD retrieval")

# label a few heights on the curve so it's clear which end is the bottom
# np.argmin(abs(heights - hl)) = index of the height closest to hl
for hl in [10, 100, 300, 600]:
    i_hl = np.argmin(abs(heights - hl))
    ax_vec[2].annotate(f"{heights[i_hl]:.0f} m",                # the label text
                       (true_u[i_hl], true_v[i_hl]),             # point it goes on
                       textcoords="offset points", xytext=(8, -4))  # nudge it right a bit

# + at the origin (zero wind) for reference
ax_vec[2].plot(0, 0, "k+", ms=10)

# axis equal so 1 m/s is the same length on both axes,
# otherwise the turning angle looks wrong
ax_vec[2].axis("equal")
ax_vec[2].set_xlabel("u (toward east) [m/s]")
ax_vec[2].set_ylabel("v (toward north) [m/s]")
ax_vec[2].set_title("Hodograph")
ax_vec[2].legend(loc="lower right")
ax_vec[2].grid(alpha=0.3)

# tight_layout fixes the spacing so labels don't overlap, then save + show
plt.tight_layout()
plt.savefig("vad_vectors.png", dpi=150)
plt.show()


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

# STEP 1: Load the file

mat = loadmat("Data_for_VAD.mat", squeeze_me=True, struct_as_record=False)
scans = mat["Data"]
print("Number of scans:", len(scans))

# Range gates are the same for every scan -> heights are the same too
ranges  = np.asarray(scans[0].range[:, 0], dtype=float)   # 12, 36, 60 ... 1980 m
elev    = 75.0
heights = ranges * np.sin(np.radians(elev))               # ~12 m ... ~1910 m

MIN_GOOD_BEAMS = 0.5    # need at least 50% of the beams valid at a gate to trust the fit
MAX_FIT_ERROR  = 1.0    # m/s: if the sine fit is worse than this, throw the gate away

# STEP 2: VAD fit for ONE scan  (same idea as vad_simple.py)

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

# STEP 3: Run VAD on every scan
#   U[t, h] = east wind at scan t, height h   (same for V)

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


# STEP 4: Mean profiles over the whole dataset

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

# ITEM 4: movies of the wind speed and wind direction profiles

from matplotlib.animation import FuncAnimation, PillowWriter

FPS      = 4        # frames per second
H_TOP    = 1000     # only show up to 1000 m, almost no good data above ~750 m
show     = heights <= H_TOP   # true/false for each height, used to cut the arrays

# mean direction from STEP 4 is only kept where >25% of scans have data,
# so blank it out (NaN) everywhere else so the gray reference matches
mean_dir_plot = np.where(keep, mean_dir, np.nan)
mean_speed_plot = np.where(keep, mean_speed, np.nan)


def make_profile_movie(values, mean_profile, xlabel, xlim, color, style, fname, title):
    """
    makes one movie. inputs:
      values       -> SPEED or DIR, shape (scans, heights)
      mean_profile -> the mean over all scans, drawn in gray for reference
      xlabel, xlim -> x axis label and limits
      color, style -> line color and style ("o-" = dots + line, "o" = dots only)
      fname        -> name of the gif to save
      title        -> what the movie is of (time gets added each frame)
    """

    # set up the figure once, the frames just update the data in it
    fig_m, ax_m = plt.subplots(figsize=(6, 7))

    # gray mean profile, stays the same the whole movie so you can see
    # when a scan is faster/slower or turned compared to normal
    ax_m.plot(mean_profile[show], heights[show], "o", color="0.6", ms=3,
              label="mean of all scans")

    # empty line for the current scan, gets filled in each frame
    # the comma after prof_line is because plot() returns a list, this
    # pulls out the one line inside it
    prof_line, = ax_m.plot([], [], style, color=color, ms=4,
                           label="current scan")

    ax_m.set_xlim(xlim)
    ax_m.set_ylim(0, H_TOP)
    ax_m.set_xlabel(xlabel)
    ax_m.set_ylabel("Height [m]")
    ax_m.grid(alpha=0.3)
    ax_m.legend(loc="upper right")
    title_text = ax_m.set_title("")    # filled in each frame with the time

    # this runs once for every frame. t = which scan we're on
    def update(t):
        # put this scan's profile into the line
        # NaN heights (bad data) just leave gaps in the line
        prof_line.set_data(values[t, show], heights[show])

        # put the date/time of this scan in the title
        # %d-%b-%Y %H:%M -> e.g. 11-Oct-2013 21:40
        title_text.set_text(f"{title}\n{times[t]:%d-%b-%Y %H:%M}   (scan {t+1} of {n_scans})")

        # FuncAnimation wants back the things that changed
        return prof_line, title_text

    # build the animation: call update() for every scan 0, 1, ..., n_scans-1
    # interval = ms between frames (only matters for live playback)
    anim = FuncAnimation(fig_m, update, frames=n_scans, interval=1000 / FPS)

    # save as a gif. this is the slow part (draws all 516 frames),
    # takes a minute or two so it prints before and after
    print(f"saving {fname} ... (takes a minute or two)")
    anim.save(fname, writer=PillowWriter(fps=FPS))
    print(f"done -> {fname}")

    # close the figure so it doesn't pop up as a still window
    plt.close(fig_m)


# movie 1: wind speed profiles (dots + line since speed is smooth with height)
make_profile_movie(SPEED, mean_speed_plot,
                  xlabel="Wind speed [m/s]", xlim=(0, 13),
                   color="b", style="o-",
                  fname="movie_wind_speed.gif",
                   title="VAD wind speed profile")

# movie 2: wind direction profiles
# dots only, no line: a line would draw a fake jump across the plot every
# time the direction crosses 0/360 (north)
make_profile_movie(DIR, mean_dir_plot,
                   xlabel="Wind direction [deg, from]", xlim=(0, 360),
                   color="r", style="o",
                   fname="movie_wind_direction.gif",
                   title="VAD wind direction profile")


# ITEM 5: evidence for the meteorological summary
from datetime import timedelta

#  local time: file times are UTC, site is UTC-7 (western US)
local_hour = np.array([(tt + timedelta(hours=-7)).hour for tt in times])
night = (local_hour >= 20) | (local_hour < 6) # 8 pm - 6 am
day   = (local_hour >= 10) & (local_hour < 17) # 10 am - 5 pm


# PLOT 1: whole dataset in one picture (speed + direction vs time and height)
# shows jets, the big wind event, and how much the direction jumps around
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax[0].scatter(np.repeat(times, len(heights)), np.tile(heights, n_scans),
              c=SPEED.ravel(), s=4, marker="s", cmap="viridis", vmin=0, vmax=10)
ax[1].scatter(np.repeat(times, len(heights)), np.tile(heights, n_scans),
              c=DIR.ravel(), s=4, marker="s", cmap="twilight", vmin=0, vmax=360)
fig.colorbar(ax[0].collections[0], ax=ax[0], label="Speed [m/s]")
fig.colorbar(ax[1].collections[0], ax=ax[1], label="Direction [deg, from]")
for a in ax:
    a.set_ylim(0, 900); a.set_ylabel("Height [m]")
ax[0].set_title("Wind speed"); ax[1].set_title("Wind direction")
ax[1].set_xlabel("Date (UTC)")
plt.tight_layout(); plt.savefig("item5_time_height.png", dpi=150); plt.show()


# --- PLOT 2: night vs day profile (Stull) + which way the wind blows ---
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].plot(np.nanmean(SPEED[night], axis=0)[keep], heights[keep], "o-", label="night")
ax[0].plot(np.nanmean(SPEED[day], axis=0)[keep], heights[keep], "o-", label="day")
ax[0].set_xlabel("Mean speed [m/s]"); ax[0].set_ylabel("Height [m]")
ax[0].set_title("Night vs day profile"); ax[0].legend(); ax[0].grid(alpha=0.3)

i300 = np.argmin(abs(heights - 300))                 # 300 m has data in every scan
ax[1].hist(DIR[:, i300], bins=np.arange(0, 361, 30), edgecolor="k")
ax[1].set_xlabel("Direction at 300 m [deg, from]"); ax[1].set_ylabel("Number of scans")
ax[1].set_title("Wind direction distribution"); ax[1].set_xticks(range(0, 361, 90))
plt.tight_layout(); plt.savefig("item5_night_day.png", dpi=150); plt.show()


# --- NUMBERS ---
print("\n---- ITEM 5 ----")
# nocturnal jets: fastest wind is below 600 m, at least 4 m/s,
# and the wind above it is at least 2 m/s slower
jet = np.zeros(n_scans, dtype=bool)
for t in range(n_scans):
    prof = SPEED[t, heights <= 800]
    if np.sum(np.isfinite(prof)) < 8:
        continue
    k = np.nanargmax(prof)
    if heights[k] < 600 and prof[k] >= 4 and np.isfinite(prof[k+1:]).any():
        jet[t] = prof[k] - np.nanmin(prof[k+1:]) >= 2
print(f"jets: {100*jet[night].mean():.0f}% of night scans, {100*jet[day].mean():.0f}% of day scans")

# rotor: hub 200 m, diameter 100 m -> 150 to 250 m
# (lidar has almost no data below ~130 m, so a normal 100 m hub can't be checked)
i_bot = np.argmin(abs(heights - 150))
i_top = np.argmin(abs(heights - 250))
dspeed = SPEED[:, i_top] - SPEED[:, i_bot]
veer = (DIR[:, i_top] - DIR[:, i_bot] + 180) % 360 - 180 # wrapped to -180..180
print(f"rotor {heights[i_bot]:.0f}-{heights[i_top]:.0f} m:")
print(f"   speed change top-bottom: median {np.nanmedian(dspeed):+.2f} m/s")
print(f"   direction change top-bottom: median size {np.nanmedian(abs(veer)):.0f} deg")
i_hub = np.argmin(abs(heights - 200))
print(f"   hub (200 m) speed above 3 m/s (cut-in): {100*np.nanmean(SPEED[:, i_hub] > 3):.0f}% of the time")
