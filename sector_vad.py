"""
HW3 - Part 2, Sub-Part 2: Sector VAD

the plan:
  STEP 1. read the 3D wind fields that WindNinja (RANS, momentum solver) made
  STEP 2. plot slices through the domain with vectors       -> deliverable B
  STEP 3. virtual lidar: PPI and RHI scans of the CFD field -> deliverables C, D
  STEP 4. sector VAD on the low-level PPIs                  -> deliverables A, E
  STEP 5. compare the VAD to the CFD "truth"                -> deliverable F

site: Tehachapi Pass, CA (box 35.016-35.124 N, 118.316-118.184 W, ~12 x 12 km)
wind cases from the Mojave (MHV) wind rose, all given at 10 m above ground:
  270 deg @ 10 m/s, 300 @ 7.5, 230 @ 5.5, 90 @ 4
"""

import glob                      # finds files that match a pattern (like dir *.vtk)
import os                        # for file names / folders
import re                        # "regular expressions", used to pull numbers out of text
import numpy as np
import matplotlib.pyplot as plt

# folder where i put the WindNinja output files (.vtk), relative to this script
# look for the windninja folder next to THIS script, no matter where the terminal is
WN_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windninja")
print("looking for WindNinja files in:", WN_FOLDER)


# =====================================================================
# STEP 1: read the WindNinja 3D wind field (.vtk file)
# =====================================================================
# the .vtk file is a "structured grid": 100 x 102 points across the ground,
# 20 levels up. the levels follow the terrain (bottom level = the ground,
# top level = flat lid ~1944 m elevation), and they're squished close
# together near the ground where the wind changes fastest
#
# the file is BINARY (not readable text). after a short text header it has:
#   POINTS  -> x, y, z of every grid point
#   VECTORS -> u, v, w of the wind at every grid point
# all stored as big-endian 8 byte numbers (">f8" in numpy)

def read_windninja_vtk(fname):
    raw = open(fname, "rb").read()          # "rb" = read the file as raw bytes

    # pull the grid size out of the header line "DIMENSIONS 100 102 20"
    nx, ny, nz = [int(d) for d in
                  re.search(rb"DIMENSIONS (\d+) (\d+) (\d+)", raw).groups()]
    n = nx * ny * nz                        # total number of grid points

    # the numbers start right after the end of the "POINTS ..." line
    i = raw.index(b"\n", raw.index(b"POINTS")) + 1
    pts = np.frombuffer(raw[i:i + n*3*8], dtype=">f8")   # n points x 3 coords x 8 bytes

    # same thing for the wind vectors after the "VECTORS ..." line
    j = raw.index(b"\n", raw.index(b"VECTORS")) + 1
    vec = np.frombuffer(raw[j:j + n*3*8], dtype=">f8")

    # reshape the long lists into 3D arrays [level k, row j (y), column i (x)]
    # (vtk stores x changing fastest, then y, then z)
    pts = pts.reshape(nz, ny, nx, 3)
    vec = vec.reshape(nz, ny, nx, 3)

    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]   # [m], x = east, y = north
    U, V, W = vec[..., 0], vec[..., 1], vec[..., 2]   # [m/s], u = east, v = north, w = up
    return X, Y, Z, U, V, W


# helper: get a field at a constant height ABOVE GROUND (AGL)
# the grid levels aren't at the same height above ground everywhere, so for
# each (x, y) column, interpolate up the column to the height we want
# (np.interp = matlab interp1)
def slice_agl(field, AGL, h):
    nz, ny, nx = field.shape
    out = np.zeros((ny, nx))
    for jj in range(ny):
        for ii in range(nx):
            out[jj, ii] = np.interp(h, AGL[:, jj, ii], field[:, jj, ii])
    return out


# find all the 3D wind files (skip the "_surf" ones, those are only the ground)
vtk_files = sorted(f for f in glob.glob(os.path.join(WN_FOLDER, "*.vtk"))
                   if not f.endswith("_surf.vtk"))
print("found", len(vtk_files), "WindNinja runs")

# the speeds i actually typed into WindNinja for each direction.
# WindNinja ROUNDS the speed in the file name (5.5 -> "6", 7.5 -> "8"),
# so the real input speed comes from here instead of the file name
INPUT_SPEED = {270: 10.0, 300: 7.5, 230: 5.5, 90: 4.0}

# read every case into a dictionary, so cases["270"] = that run's arrays
# the direction comes from the file name, e.g. ..._270_10_118m.vtk -> 270
cases = {}
for f in vtk_files:
    wdir = int(re.search(r"_(\d+)_[\d.]+_\d+m\.vtk$", f).group(1))
    X, Y, Z, U, V, W = read_windninja_vtk(f)
    cases[str(wdir)] = dict(X=X, Y=Y, Z=Z, U=U, V=V, W=W,
                            AGL=Z - Z[0],            # height above the ground
                            wdir=float(wdir), wspd=INPUT_SPEED[wdir])
    print(f"  case {wdir} deg @ {INPUT_SPEED[wdir]} m/s: grid {U.shape[2]} x {U.shape[1]} x {U.shape[0]}")


# =====================================================================
# STEP 2: slices through the domain with vectors  (deliverable B)
# =====================================================================
# for each case, 2 figures:
#   fig 1: horizontal slices at 3 heights above ground, colored by wind
#          speed, with arrows, and terrain contour lines on top
#   fig 2: vertical slice west -> east through the middle of the domain,
#          colored by wind speed, with the terrain underneath
# together these show how the wind changes with position AND height

SLICE_HEIGHTS = [10, 50, 150]     # [m] above ground for the horizontal slices
SKIP = 6                          # only draw every 6th arrow so they don't pile up

for name, c in cases.items():
    X, Y, Z, AGL = c["X"], c["Y"], c["Z"], c["AGL"]
    terrain = Z[0]                     # bottom level = ground elevation [m]
    xkm, ykm = X[0] / 1000, Y[0] / 1000   # plot in km, easier to read

    # ---------------- fig 1: horizontal slices -----------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)

    for a, h in zip(ax, SLICE_HEIGHTS):
        # wind at height h above ground
        u_h = slice_agl(c["U"], AGL, h)
        v_h = slice_agl(c["V"], AGL, h)
        spd_h = np.sqrt(u_h**2 + v_h**2)

        # color = speed. same color scale (vmin/vmax) on every panel so
        # the panels can be compared
        pc = a.pcolormesh(xkm, ykm, spd_h, cmap="viridis",
                          vmin=0, vmax=1.6 * c["wspd"], shading="auto")

        # terrain contour lines every 100 m of elevation, thin white
        a.contour(xkm, ykm, terrain, levels=np.arange(800, 1800, 100),
                  colors="w", linewidths=0.5, alpha=0.7)

        # arrows = wind direction + strength (every SKIP-th point)
        a.quiver(xkm[::SKIP, ::SKIP], ykm[::SKIP, ::SKIP],
                 u_h[::SKIP, ::SKIP], v_h[::SKIP, ::SKIP],
                 color="k", scale=25 * c["wspd"], width=0.003)

        a.set_title(f"{h} m above ground")
        a.set_xlabel("x (east) [km]")
        a.set_aspect("equal")          # so 1 km looks the same both directions

    ax[0].set_ylabel("y (north) [km]")
    fig.colorbar(pc, ax=ax, label="Wind speed [m/s]", shrink=0.9)
    fig.suptitle(f"WindNinja RANS, case {c['wdir']:.0f} deg @ {c['wspd']} m/s (at 10 m)"
                 "  -  white lines = terrain every 100 m")
    plt.savefig(f"rans_horizontal_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---------------- fig 2: vertical slice west -> east -------------
    # pick the row (north-south position) where the terrain is bumpiest,
    # so the slice cuts through the ridges and not just a smooth slope.
    # np.std(..., axis=1) = how much the ground height varies along each row
    # skipping the 10 rows nearest the edges (the edges of the domain have
    # boundary effects from the model, see the slow wind at y = 0 and 12 km)
    rough = np.std(terrain, axis=1)
    jm = 10 + np.argmax(rough[10:-10])
    xs = X[:, jm, :] / 1000                  # [levels, x] positions along the slice, km
    zs = Z[:, jm, :]                         # [levels, x] heights (follow the terrain)
    spd_s = np.sqrt(c["U"][:, jm, :]**2 + c["V"][:, jm, :]**2)

    fig, a = plt.subplots(figsize=(13, 5))

    # pcolormesh works on the bent (terrain-following) grid directly
    pc = a.pcolormesh(xs, zs, spd_s, cmap="viridis",
                      vmin=0, vmax=1.6 * c["wspd"], shading="gouraud")

    # fill the ground in gray so it's clear where the terrain is
    a.fill_between(xs[0], 700, zs[0], color="0.4")

    # arrows: horizontal component along the slice (u) and vertical (w)
    # every 4th column and every 2nd level
    a.quiver(xs[1::2, ::4], zs[1::2, ::4], c["U"][1::2, jm, ::4], c["W"][1::2, jm, ::4],
             color="w", scale=30 * c["wspd"], width=0.002)

    a.set_ylim(700, 1950)
    a.set_xlabel("x (east) [km]")
    a.set_ylabel("Elevation [m]")
    a.set_title(f"West-east slice at y = {Y[0, jm, 0]/1000:.1f} km (through the ridges), case "
                f"{c['wdir']:.0f} deg @ {c['wspd']} m/s")
    fig.colorbar(pc, ax=a, label="Wind speed [m/s]")
    plt.savefig(f"rans_vertical_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# STEP 3: virtual lidar  (deliverable C, and the PPI plots for D)
# =====================================================================
# the idea: put a fake lidar in the CFD domain, shoot beams out at
# different azimuths/elevations, and at every range gate along each beam
# grab the CFD wind there and keep only the part ALONG the beam (radial
# velocity). that's all a real Doppler lidar can measure.
#
#   radial velocity  Vr = u*sin(az)*cos(el) + v*cos(az)*cos(el) + w*sin(el)
#   (az measured clockwise from north, positive Vr = wind blowing AWAY from lidar)
#
# PPI = elevation fixed, azimuth sweeps all the way around (cone/flat-ish scan)
# RHI = azimuth fixed, elevation sweeps up (vertical slice)
#
# realistic stuff i included:
#   - beams that hit a hill are BLOCKED: that gate and every gate past it
#     along the beam become NaN (the laser can't see through ground)
#   - gates outside the domain, or too close to the domain edges (boundary
#     effects in the CFD, see the STEP 2 plots), are thrown out
#   - a little random noise on Vr, like a real lidar

LIDAR_X   = 6000.0     # [m] lidar position in the grid (domain center)
LIDAR_Y   = 6000.0     # [m]   picked because ~80% of a 3 deg scan isn't blocked by hills
LIDAR_HAG = 3.0        # [m] lidar height above the ground (on a trailer)

RANGES     = np.arange(100, 4001, 50)     # range gates every 50 m, out to 4 km
PPI_AZ     = np.arange(0, 360, 2)         # PPI beams every 2 deg
PPI_ELEVS  = [2, 3, 5]                    # "low level" PPI elevation angles [deg]
RHI_EL     = np.arange(0.5, 30.1, 0.5)    # RHI elevations [deg]
NOISE_STD  = 0.1                          # [m/s] lidar noise on Vr (typical ~0.1 m/s)
EDGE_CELLS = 8                            # ignore the 8 grid cells (~1 km) nearest each edge

rng = np.random.default_rng(488)          # random numbers for the noise (fixed seed = repeatable)


def sample_cfd(c, gx, gy, gz):
    """
    get the CFD wind (u, v, w) at any points (gx, gy, gz).
    the grid is regular in x and y but the levels bend with the terrain, so:
      1) for each point, blend the 4 surrounding grid COLUMNS (bilinear
         interpolation in x and y) -> one column of heights + winds at that spot
      2) interpolate up that column to the point's height
    also returns which points are under the ground and which are inside the
    usable part of the domain
    """
    X, Y, Z = c["X"], c["Y"], c["Z"]
    nz, ny, nx = Z.shape
    shape = np.shape(gx)
    gx, gy, gz = np.ravel(gx), np.ravel(gy), np.ravel(gz)

    # where is each point in "grid index" units (can be in between grid points)
    x0, dx = X[0, 0, 0], X[0, 0, 1] - X[0, 0, 0]
    y0, dy = Y[0, 0, 0], Y[0, 1, 0] - Y[0, 0, 0]
    fi = (gx - x0) / dx
    fj = (gy - y0) / dy

    # usable region: inside the domain and away from the edges
    inside = ((fi >= EDGE_CELLS) & (fi <= nx - 1 - EDGE_CELLS) &
              (fj >= EDGE_CELLS) & (fj <= ny - 1 - EDGE_CELLS))

    # lower-left grid point of the box each point falls in, and how far
    # across the box it is (0 to 1) in x (wx) and y (wy)
    fi = np.clip(fi, 0, nx - 1.001)
    fj = np.clip(fj, 0, ny - 1.001)
    i0 = fi.astype(int); wx = fi - i0
    j0 = fj.astype(int); wy = fj - j0

    # bilinear blend of the 4 corner columns -> shape (levels, points)
    def blend(F):
        return ((1 - wx) * (1 - wy) * F[:, j0, i0]     + wx * (1 - wy) * F[:, j0, i0 + 1] +
                (1 - wx) * wy       * F[:, j0 + 1, i0] + wx * wy       * F[:, j0 + 1, i0 + 1])

    Zc, Uc, Vc, Wc = blend(Z), blend(c["U"]), blend(c["V"]), blend(c["W"])

    # which level is each point sitting just above (k), and how far up
    # between level k and k+1 (t, 0 to 1)  -> same thing np.interp does,
    # but for all the points at once
    k = np.sum(Zc <= gz[None, :], axis=0) - 1
    k = np.clip(k, 0, nz - 2)
    pts = np.arange(gz.size)
    z_lo, z_hi = Zc[k, pts], Zc[k + 1, pts]
    t = np.clip((gz - z_lo) / (z_hi - z_lo), 0, 1)

    def up_column(Fc):
        return Fc[k, pts] + t * (Fc[k + 1, pts] - Fc[k, pts])

    u, v, w = up_column(Uc), up_column(Vc), up_column(Wc)

    under = gz < Zc[0]                 # below the ground surface
    above = gz > Zc[-1]                # above the model top
    inside = inside & ~above

    return (u.reshape(shape), v.reshape(shape), w.reshape(shape),
            under.reshape(shape), inside.reshape(shape), Zc[0].reshape(shape))


# ground height right under the lidar (sample a point at z = 0, only need the ground)
_, _, _, _, _, LIDAR_GROUND = sample_cfd(next(iter(cases.values())),
                                         np.array([LIDAR_X]), np.array([LIDAR_Y]), np.array([0.0]))
LIDAR_Z = float(LIDAR_GROUND[0]) + LIDAR_HAG
print(f"lidar at x = {LIDAR_X/1000:.1f} km, y = {LIDAR_Y/1000:.1f} km, z = {LIDAR_Z:.0f} m elevation")


def lidar_scan(c, az_deg, el_deg):
    """
    virtual lidar. az_deg and el_deg = one value per beam (same length).
    returns a dictionary with everything at every [beam, range gate]:
      gx, gy, gz = where the gate is
      vr         = radial velocity the lidar "measures" (NaN if blocked/out)
      u, v, w    = the true CFD wind there (to check the VAD against later)
      ok         = True where the gate is good
    """
    az = np.radians(np.asarray(az_deg, float))[:, None]   # [:, None] makes it a column
    el = np.radians(np.asarray(el_deg, float))[:, None]   #   so it broadcasts against RANGES
    r = RANGES[None, :].astype(float)                     # row of ranges

    # where each gate is in space (straight beams, no earth curvature at 4 km)
    gx = LIDAR_X + r * np.cos(el) * np.sin(az)
    gy = LIDAR_Y + r * np.cos(el) * np.cos(az)
    gz = LIDAR_Z + r * np.sin(el)

    u, v, w, under, inside, _ = sample_cfd(c, gx, gy, gz)

    # blocking: once a beam goes under the ground, everything farther out on
    # that beam is gone too. cumsum along the range direction counts how many
    # under-ground gates there have been so far, > 0 means blocked
    blocked = np.cumsum(under, axis=1) > 0
    ok = inside & ~blocked

    # the actual lidar measurement: wind projected on the beam + noise
    vr = (u * np.sin(az) + v * np.cos(az)) * np.cos(el) + w * np.sin(el)
    vr = vr + NOISE_STD * rng.standard_normal(vr.shape)
    vr[~ok] = np.nan

    return dict(az=np.degrees(az[:, 0]), el=np.degrees(el[:, 0]),
                gx=gx, gy=gy, gz=gz, vr=vr, u=u, v=v, w=w, ok=ok)


# run the virtual lidar on every case:
#   ppi[name][el] = PPI scan at that elevation
#   rhi[name]     = RHI scan pointed INTO the wind (looking at where the wind comes from)
ppi, rhi = {}, {}
for name, c in cases.items():
    ppi[name] = {el: lidar_scan(c, PPI_AZ, np.full(PPI_AZ.size, el)) for el in PPI_ELEVS}
    rhi[name] = lidar_scan(c, np.full(RHI_EL.size, c["wdir"]), RHI_EL)
    good = [100 * ppi[name][el]["ok"].mean() for el in PPI_ELEVS]
    print(f"  case {name}: % of PPI gates usable at el {PPI_ELEVS} = "
          + ", ".join(f"{g:.0f}%" for g in good))


# ---------------- PPI plots, colored by radial velocity (deliverable D) ---
# one figure per case, one panel per elevation angle
# red = wind blowing away from the lidar, blue = toward it
# gray background = not seen (blocked by terrain or outside the usable domain)
terrain = next(iter(cases.values()))["Z"][0]
X0km = next(iter(cases.values()))["X"][0] / 1000
Y0km = next(iter(cases.values()))["Y"][0] / 1000

for name, c in cases.items():
    fig, ax = plt.subplots(1, len(PPI_ELEVS), figsize=(17, 5.5), sharey=True)
    vmax = 1.5 * c["wspd"]                     # same color range on all panels

    for a, el in zip(ax, PPI_ELEVS):
        s = ppi[name][el]
        a.set_facecolor("0.85")                # gray = no data

        # np.vstack(... s["gx"][:1]) repeats the first beam at the end so the
        # plot closes the circle (otherwise there's a gap between 358 and 0 deg)
        gxk = np.vstack([s["gx"], s["gx"][:1]]) / 1000
        gyk = np.vstack([s["gy"], s["gy"][:1]]) / 1000
        vr  = np.vstack([s["vr"], s["vr"][:1]])
        pc = a.pcolormesh(gxk, gyk, np.ma.masked_invalid(vr), cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, shading="gouraud")

        a.contour(X0km, Y0km, terrain, levels=np.arange(800, 1800, 100),
                  colors="k", linewidths=0.4, alpha=0.5)
        a.plot(LIDAR_X / 1000, LIDAR_Y / 1000, "k^", ms=9)   # lidar location
        a.set_title(f"PPI, elevation {el} deg")
        a.set_xlabel("x (east) [km]")
        a.set_aspect("equal")
        a.set_xlim(X0km.min(), X0km.max()); a.set_ylim(Y0km.min(), Y0km.max())

    ax[0].set_ylabel("y (north) [km]")
    fig.colorbar(pc, ax=ax, label="Radial velocity [m/s]  (+ = away from lidar)", shrink=0.9)
    fig.suptitle(f"Virtual lidar PPIs, case {c['wdir']:.0f} deg @ {c['wspd']} m/s  "
                 "(triangle = lidar, black lines = terrain every 100 m)")
    plt.savefig(f"ppi_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# ---------------- RHI plots (shows the scanner can do RHIs too) ------------
# beam pointed toward where the wind comes FROM, so Vr is mostly negative
# (wind blowing toward the lidar). x axis = horizontal distance from lidar
fig, ax = plt.subplots(len(cases), 1, figsize=(11, 3.2 * len(cases)), sharex=True)
ax = np.atleast_1d(ax)
for a, (name, c) in zip(ax, cases.items()):
    s = rhi[name]
    hdist = np.hypot(s["gx"] - LIDAR_X, s["gy"] - LIDAR_Y) / 1000   # km from lidar
    vmax = 1.5 * c["wspd"]
    a.set_facecolor("0.85")
    pc = a.pcolormesh(hdist, s["gz"], np.ma.masked_invalid(s["vr"]), cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, shading="gouraud")

    # terrain along the RHI line, so you can see what blocks the low beams
    rr = np.linspace(0, RANGES.max(), 200)
    az = np.radians(c["wdir"])
    _, _, _, _, _, gline = sample_cfd(c, LIDAR_X + rr * np.sin(az),
                                      LIDAR_Y + rr * np.cos(az), np.zeros_like(rr))
    a.fill_between(rr / 1000, 700, gline, color="0.4")
    a.set_ylim(LIDAR_Z - 200, LIDAR_Z + 1200)
    a.set_ylabel("Elevation [m]")
    a.set_title(f"RHI at azimuth {c['wdir']:.0f} deg (into the wind), "
                f"case {c['wdir']:.0f} deg @ {c['wspd']} m/s")
    fig.colorbar(pc, ax=a, label="Vr [m/s]")
ax[-1].set_xlabel("Horizontal distance from lidar [km]")
plt.tight_layout()
plt.savefig("rhi_all_cases.png", dpi=150, bbox_inches="tight")
plt.show()


# =====================================================================
# STEP 4: sector VAD  (deliverables A and E)
# =====================================================================
# regular VAD (Sub-Part 1) needs the full 360 deg circle and assumes the
# wind is the SAME all the way around it. in complex terrain that's not
# true, so instead chop the PPI into pie-slice SECTORS and only assume the
# wind is the same inside each small sector.
#
# inside one sector and range interval ("cell"):
#   Vr / cos(el) = u*sin(az) + v*cos(az)        (w term dropped, at 2-5 deg
#                                                elevation sin(el) is tiny)
#   two unknowns (u, v), lots of gates -> least squares, same as before
#
# trade-off (this is the key thing for the assessment):
#   narrow sector -> wind really is about the same inside it (good) BUT the
#                    beams all point almost the same way, so it's hard to
#                    tell u from v and noise gets blown up (bad)
#   wide sector   -> u and v are easy to separate (good) BUT the wind
#                    changes across the sector in hilly terrain (bad)

SECTOR_W    = 30                                  # [deg] sector width
SECTOR_CTRS = np.arange(0, 360, SECTOR_W)        # sector center azimuths
# range intervals: 600-1100, 1100-1600, ... m
# starting at 600 m, NOT 100 m: on the 2 deg PPI the gates closer than 600 m
# are only ~20 m above the ground, where the wind changes really fast with
# height. on a slope the uphill beams are closer to the ground than the
# downhill ones, so the wind isn't the same around the sector and the fit
# was way off there (vector errors 2-5 m/s vs ~1 m/s farther out).
# real lidars have a near-range blind zone anyway
RANGE_BINS  = np.arange(600, 4101, 500)
MIN_FILL    = 0.5     # need at least half of a cell's gates to be good to trust it


def sector_vad(s, width=SECTOR_W):
    """
    sector VAD on one PPI scan s (from lidar_scan).
    returns one row per cell: center x, y, retrieved u, v, true u, v
    """
    centers = np.arange(0, 360, width)
    el = np.radians(s["el"][0])                    # PPI = same elevation for every beam
    rows = []
    for ac in centers:
        # angle between each beam and the sector center, wrapped to -180..180
        # so sectors that cross north (e.g. 345-15 deg) still work
        daz = (s["az"] - ac + 180) % 360 - 180
        in_sector = (daz >= -width / 2) & (daz < width / 2)

        for r_lo, r_hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
            in_range = (RANGES >= r_lo) & (RANGES < r_hi)
            cell = in_sector[:, None] & in_range[None, :]      # all gates in this cell
            good = cell & np.isfinite(s["vr"])
            if good.sum() < MIN_FILL * cell.sum() or good.sum() < 10:
                continue                                       # too much missing -> skip

            az_g = np.radians(np.broadcast_to(s["az"][:, None], cell.shape)[good])
            vr_g = s["vr"][good] / np.cos(el)

            # least squares: [sin(az)  cos(az)] [u; v] = Vr/cos(el)
            M = np.column_stack([np.sin(az_g), np.cos(az_g)])
            (u_r, v_r), *_ = np.linalg.lstsq(M, vr_g, rcond=None)

            # "truth" = average CFD wind at the same gates the lidar used
            rows.append([s["gx"][good].mean(), s["gy"][good].mean(),
                         u_r, v_r, s["u"][good].mean(), s["v"][good].mean()])
    return np.array(rows)


def uv_to_speed_dir(u, v):
    speed = np.sqrt(u**2 + v**2)
    direction = np.degrees(np.arctan2(-u, -v)) % 360      # direction wind comes FROM
    return speed, direction


# run the sector VAD on every PPI of every case
vad = {name: {el: sector_vad(ppi[name][el]) for el in PPI_ELEVS} for name in cases}

# ---------------- retrieval plots (deliverable E) -------------------------
# lowest PPI (2 deg) for each case: black = CFD truth, red = sector VAD
EL_SHOW = PPI_ELEVS[0]
fig, ax = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 5.5), sharey=True)
ax = np.atleast_1d(ax)
for a, (name, c) in zip(ax, cases.items()):
    R = vad[name][EL_SHOW]
    a.contour(X0km, Y0km, terrain, levels=np.arange(800, 1800, 100),
              colors="0.5", linewidths=0.4)
    sc = 3 * c["wspd"]                  # arrow scale, same for truth + VAD
    a.quiver(R[:, 0] / 1000, R[:, 1] / 1000, R[:, 4], R[:, 5], color="k",
             scale=sc, scale_units="inches", width=0.004, label="CFD truth")
    a.quiver(R[:, 0] / 1000, R[:, 1] / 1000, R[:, 2], R[:, 3], color="r",
             scale=sc, scale_units="inches", width=0.004, alpha=0.8, label="sector VAD")
    a.plot(LIDAR_X / 1000, LIDAR_Y / 1000, "k^", ms=9)
    a.set_title(f"case {c['wdir']:.0f} deg @ {c['wspd']} m/s")
    a.set_xlabel("x (east) [km]"); a.set_aspect("equal")
    a.set_xlim(X0km.min(), X0km.max()); a.set_ylim(Y0km.min(), Y0km.max())
ax[0].set_ylabel("y (north) [km]")
ax[0].legend(loc="lower left")
fig.suptitle(f"Sector VAD on the {EL_SHOW} deg PPI ({SECTOR_W} deg sectors, "
             f"{RANGE_BINS[1]-RANGE_BINS[0]} m range bins)")
plt.savefig("sector_vad_retrievals.png", dpi=150, bbox_inches="tight")
plt.show()


# =====================================================================
# STEP 5: how well does it work?  (deliverable F)
# =====================================================================
# compare each retrieved vector to the CFD truth at the same gates
#   speed bias   = average of (VAD speed - true speed)
#   speed RMSE   = typical size of the speed error
#   dir RMSE     = typical direction error (wrapped so 359 vs 1 = 2 deg, not 358)
#   vector RMSE  = typical length of (VAD vector - true vector), catches both

def error_stats(R):
    s_r, d_r = uv_to_speed_dir(R[:, 2], R[:, 3])
    s_t, d_t = uv_to_speed_dir(R[:, 4], R[:, 5])
    ddir = (d_r - d_t + 180) % 360 - 180
    vec = np.hypot(R[:, 2] - R[:, 4], R[:, 3] - R[:, 5])
    return (len(R), np.mean(s_r - s_t), np.sqrt(np.mean((s_r - s_t)**2)),
            np.sqrt(np.mean(ddir**2)), np.sqrt(np.mean(vec**2)))

print("\nsector VAD vs CFD truth")
print(" case   el   cells   speed bias   speed RMSE   dir RMSE   vector RMSE")
for name, c in cases.items():
    for el in PPI_ELEVS:
        n, b, sr, dr, vr_ = error_stats(vad[name][el])
        print(f"{name:>5} {el:>4} {n:>7}   {b:+8.2f}     {sr:8.2f}    {dr:7.1f}    {vr_:8.2f}")

# ---------------- error vs distance from the lidar ------------------------
# sectors get physically wider farther out (a 30 deg slice is ~300 m wide at
# 600 m but ~2 km wide at 4 km), so they cover more varied terrain -> the
# "same wind in the sector" assumption gets worse with range
print("\nvector RMSE [m/s] vs range (3 deg PPI)")
print(" case  " + "  ".join(f"{lo/1000:.1f}-{hi/1000:.1f}km" for lo, hi
                            in zip(RANGE_BINS[:-1], RANGE_BINS[1:])))
for name in cases:
    R = vad[name][3]
    r = np.hypot(R[:, 0] - LIDAR_X, R[:, 1] - LIDAR_Y)       # distance of each cell
    e = np.hypot(R[:, 2] - R[:, 4], R[:, 3] - R[:, 5])       # vector error of each cell
    vals = []
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        m = (r >= lo) & (r < hi)
        vals.append(f"{np.sqrt(np.mean(e[m]**2)):9.2f}" if m.any() else "      ---")
    print(f"{name:>5}  " + "  ".join(vals))

# ---------------- sector width trade-off ----------------------------------
# rerun with different sector widths to show the narrow vs wide trade-off
WIDTHS = [10, 20, 30, 45, 60, 90]
print("\nvector RMSE [m/s] vs sector width (3 deg PPI)")
print(" case  " + "  ".join(f"{w:>5}" for w in WIDTHS))
width_err = {}
for name in cases:
    width_err[name] = [error_stats(sector_vad(ppi[name][3], width=w))[4] for w in WIDTHS]
    print(f"{name:>5}  " + "  ".join(f"{e:5.2f}" for e in width_err[name]))

# plots: retrieved vs true speed (all cases, all elevations) + the width trade-off
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for name, c in cases.items():
    R = np.vstack([vad[name][el] for el in PPI_ELEVS])
    s_r, _ = uv_to_speed_dir(R[:, 2], R[:, 3])
    s_t, _ = uv_to_speed_dir(R[:, 4], R[:, 5])
    ax[0].plot(s_t, s_r, "o", ms=3, label=f"case {name}")
    ax[1].plot(WIDTHS, width_err[name], "o-", label=f"case {name}")
lim = ax[0].get_xlim()
ax[0].plot([0, 20], [0, 20], "k--", lw=1)            # perfect retrieval line
ax[0].set_xlim(0, None); ax[0].set_ylim(0, None)
ax[0].set_xlabel("True (CFD) speed [m/s]"); ax[0].set_ylabel("Sector VAD speed [m/s]")
ax[0].set_title("Retrieved vs true speed (all PPIs)"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].set_xlabel("Sector width [deg]"); ax[1].set_ylabel("Vector RMSE [m/s]")
ax[1].set_title("Sector width trade-off (3 deg PPI)"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sector_vad_assessment.png", dpi=150, bbox_inches="tight")
plt.show()
