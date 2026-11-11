"""
Single-turn, off-axis injection into the polarizer ring with a septum and a
fast kicker, using the injected positron beam (config.INJECTED_BEAM_FILE).

Physics:
  1. The septum delivers the bunch at an offset X_INJ beyond the blade
     (inj_point, just downstream of the septum exit).
  2. The injection angle is chosen so the bunch centroid crosses the closed
     orbit at the kicker:  x'_s = -x_s (cos dmu + a_s sin dmu) / (b_s sin dmu)
  3. The kicker removes the remaining angle (dpx = -knl[0]), fires once and
     is off on every following turn.
  4. The bunch filaments and radiation-damps towards equilibrium.

Septum geometry (X_BLADE=optimum, the default). At the septum, with pipe
radius R, blade thickness t, clearance n sigma and sigma = sqrt(eps_inj beta_x):
     injected bunch clears the blade:      x_s - n sigma >= b + t
     injected bunch inside the pipe:       x_s + n sigma <= R
     stored beam (the same bunch, back on
     turn 1 on the orbit) clears the blade: n sigma <= b
  -> optimum  b = n sigma,  x_s = 2 n sigma + t,  possible if R >= 3 n sigma + t.
  If it does not fit, the clearance is reduced to the largest n that fits all the
  way from septum to kicker (where beta is larger): n <= (R - A t)/(2 A sigma + sigma(s)),
  with A(s) the linear transport of the centroid, x_c(s) = A(s) x_s.

Lattice elements are installed on the fly with mc.install_apertures and
mc.install_injection (LatticeBuild/misalignments_corrections.py). The ring is
then cycled so s = 0 is the injection point: all turn-by-turn data are
recorded there.

Inputs   lattice from DESIGN/CONFIG/MODE/PHASE/CHANGES (paths.study_from_env)
         cylindrical chamber: radius = widest point of the macroparticles.py
         chamber design, max(chamber_r) in PhysicalAperture/{mode}/
         chamber_per_magnet_{mode}.csv (APERTURE_R overrides; fallback 31.2 mm)
         env: N_PARTICLES, N_TURNS, X_BLADE (default optimum), X_INJ, KICK_ERROR, MATCH_TWISS=0,
              RUN_SCANS=1, SCAN_TURNS, APERTURE_R (or none), N_SIGMA, ON_AXIS=1
Outputs  Results/.../Injection/{mode}/

Run:     python injection.py   (or via run_full_sims.py)
"""
# %%
import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xtrack as xt

import config as cfg
import paths
import LatticeBuild.misalignments_corrections as mc

# %%
# ============================================================
# SETTINGS
# ============================================================
study = paths.study_from_env()
design, config, mode, phase, changes = (study[k] for k in ('design', 'config', 'mode', 'phase', 'changes'))
_on = lambda v: str(v).strip().lower() not in ('0', 'false', 'off', 'no', '')

# --- beam ---
BEAM_FILE = cfg.INJECTED_BEAM_FILE
N_PARTICLES = int(os.environ.get('N_PARTICLES', 500))
SEED = 74                # same sample of the filtered beam as macroparticles.py (n_subset=500, seed 74)
MOMENT_ITERS = 5         # moment-based core filter (filter_by_moment_core), as in macroparticles.py
# Match the beam to the ring Twiss at inj_point (the transfer line delivers a matched beam),
# as match_coordinates in macroparticles.py does. MATCH_TWISS=0 injects the beam file's optics as-is.
MATCH_TWISS = _on(os.environ.get('MATCH_TWISS', '1'))
DELTA_OFFSET = 0.0       # mean momentum offset of the injected bunch w.r.t. the ring

# Gaussian stand-in, ONLY used if BEAM_FILE is missing
FALLBACK_BEAM = dict(nemitt_x=13000e-6, nemitt_y=12000e-6, sigma_z=4.8e-3, sigma_delta=2e-3)

# --- apertures and injection hardware ---
APERTURE_R = os.environ.get('APERTURE_R', 'chamber')   # pipe radius [m], 'chamber' = widest point of the design, 'none' = no pipe
# Reference run: ON_AXIS=1 injects on the closed orbit with the kicker off and no septum blade, i.e. the
# conditions of macroparticles.py's survival study (ON_AXIS=1 APERTURE_R=none reproduces it).
ON_AXIS = _on(os.environ.get('ON_AXIS', '0'))
APERTURE_R_FALLBACK = 0.0312     # [m] if no chamber design is found
INJ = dict(septum_at='DrTripl::1', kicker_at='DrDoub::3',
           septum_length=0.8, kicker_offset=None,      # None = kicker in the centre of its drift
           target_dmu=0.25, septum_exit_offset=None,   # septum placed for exactly 90 deg to the kicker
           x_blade=0.010, side=+1)   # blade face [m]; replaced by the optimum unless X_BLADE is a number
BLADE_THICKNESS = 0.002           # [m]
N_SIGMA = float(os.environ.get('N_SIGMA', 5))   # beam half-width in sigma: injection clearance and envelopes
X_BLADE = os.environ.get('X_BLADE', 'optimum')   # blade face [m], or 'optimum' (see docstring)
if X_BLADE != 'optimum':
    INJ['x_blade'] = float(X_BLADE)
X_INJ = os.environ.get('X_INJ', 'auto')   # bunch centre at septum exit [m]; 'auto' = optimum
                                          # (2 n sigma + t), or blade + t + N_SIGMA sigma for a fixed blade
KICK_ERROR = float(os.environ.get('KICK_ERROR', 0.0))   # relative kicker strength error

# --- tracking ---
N_TURNS = os.environ.get('N_TURNS', 'auto')     # 'auto' = 3 horizontal damping times
CHUNK = 200                                      # turns per monitor chunk (memory)
SNAP_TURNS = [0, 1, 10, 100, 1000, 5000]         # full-coordinate snapshots (+ last turn)
RUN_SCANS = _on(os.environ.get('RUN_SCANS', '0'))
SCAN_PARTICLES = 200
SCAN_TURNS = int(os.environ.get('SCAN_TURNS', 500))

folder = paths.results_dir(design, config, phase, changes, metric='Injection', sub=mode)
rng = np.random.default_rng(SEED)

# %%
# ============================================================
# LATTICE: load, install apertures + injection, cycle to inj_point
# ============================================================
pdr = paths.load_lattice(design, config, mode, phase, changes)
ring = pdr.lines['ring']

if APERTURE_R == 'chamber':
    R_PIPE = APERTURE_R_FALLBACK
    ap_dir = paths.results_dir(design, config, phase, changes, metric='PhysicalAperture')
    for m_ in (mode, 'perfect'):
        f_ = os.path.join(ap_dir, m_, f'chamber_per_magnet_{m_}.csv')
        if os.path.exists(f_):
            ch = pd.read_csv(f_)
            R_PIPE = float(ch['chamber_r'].max())
            print(f'Pipe radius = widest chamber point, {ch["name"][ch["chamber_r"].idxmax()]} '
                  f'({R_PIPE*1e3:.2f} mm), from {f_}')
            break
    else:
        print(f'No chamber_per_magnet csv (run macroparticles.py first): '
              f'pipe radius {R_PIPE*1e3:.1f} mm')
elif APERTURE_R == 'none':
    R_PIPE = None
else:
    R_PIPE = float(APERTURE_R)
R_LIM = R_PIPE if R_PIPE is not None else 1.0     # septum frame when there is no pipe
A_X = A_Y = R_LIM
if ON_AXIS:
    INJ['x_blade'] = R_LIM                       # no blade in the reference run
if R_PIPE is not None:
    mc.install_apertures(pdr, radius=R_PIPE)
else:
    print('APERTURE_R=none: no chamber apertures (losses = dynamic aperture only)')
mc.install_injection(pdr, radius=R_LIM, **INJ)
_t = ring.get_table()
s_lattice = {nn: float(_t['s', nn]) for nn in ['inj_septum_entry', 'inj_septum_exit', 'inj_point', 'inj_kicker']}
ring.cycle(name_first_element='inj_point', inplace=True)   # s = 0 at the injection point

ring.configure_radiation(model='mean')
tw = ring.twiss(method='6d', radiation_analysis=True)

tau_turns = 1 / np.array(tw.damping_constants_turns)       # amplitude damping times [turns]
sig_delta_eq = np.sqrt(tw.eq_gemitt_zeta / tw.bets0)
sig_z_eq = np.sqrt(tw.eq_gemitt_zeta * tw.bets0)
co0 = {k: tw[k][0] for k in ['x', 'px', 'y', 'py', 'zeta', 'delta']}
bx0, ax0, by0, ay0 = tw.betx[0], tw.alfx[0], tw.bety[0], tw.alfy[0]
dx0, dpx0, dy0, dpy0 = tw.dx[0], tw.dpx[0], tw.dy[0], tw.dpy[0]

print(f'D{design} C{config} {phase}deg{"_" + changes if changes else ""} {mode}: '
      f'Q = ({tw.qx:.3f}, {tw.qy:.3f}),  T_rev = {tw.T_rev0*1e9:.1f} ns')
print(f'Damping times [turns]: x {tau_turns[0]:.0f}, y {tau_turns[1]:.0f}, z {tau_turns[2]:.0f}')
print(f'Equilibrium: eps_x = {tw.eq_gemitt_x*1e9:.2f} nm, sigma_delta = {sig_delta_eq:.2e}')
print(f'At inj_point: betx = {bx0:.2f} m, alfx = {ax0:.2f}, Dx = {dx0:.3f} m')

# %%
# ============================================================
# BEAM: injected distribution (energy compression is done upstream), centred, optional Twiss rematch
# ============================================================
# --- beam core: copied unchanged from macroparticles.py so both scripts track the same beam ---
from scipy.optimize import curve_fit

def getrms_gaussfit(xis, maxnmax=50, max_doublings=20):
    """Robust average/variance via histogram + Gaussian fit, adapted from
    the supervisor's getrms(). Doubles nbins until the tallest bin drops
    below maxnmax counts, then fits a Gaussian to the histogram -- the fit
    itself is what makes this robust to a handful of far-out particles,
    since they barely move where a fitted peak sits, unlike a raw
    np.var() which they can dominate."""
    xis = np.asarray(xis, dtype=float)
    minxi = xis.min() - 0.001 * (xis.max() - xis.min())
    maxxi = xis.max() + 0.001 * (xis.max() - xis.min())
    nbins = max(2 ** int(np.floor(np.log2(len(xis) / maxnmax))), 1)
    nmax = 2 * maxnmax
    n_doublings = 0
    counts, edges = None, None
    while nmax > maxnmax and n_doublings < max_doublings:
        nbins *= 2
        counts, edges = np.histogram(xis, nbins, range=[minxi, maxxi])
        nmax = counts.max()
        n_doublings += 1
    centers = (edges[:-1] + edges[1:]) / 2

    def gauss(x, amp, xav, xvar):
        return amp * np.exp(-(x - xav) ** 2 / (2 * xvar))

    p0 = [counts.max(), xis.mean(), xis.var()]
    popt, _ = curve_fit(gauss, centers, counts, p0=p0)
    return popt[1], popt[2]


def moment_based_twiss(df, position, angle, p_col='p[MeV/c]', p0_mev=2860.0,
                        iters=5, maxnmax=50):
    """Iterative moment/projection-based Twiss reconstruction, ported from
    the supervisor's analysedataProj(). At each iteration: normalize the
    (dispersion-subtracted) coordinates using the current beta/alpha
    guess, take three projections at 0/120/240 degrees in that normalized
    frame, get each projection's variance via getrms_gaussfit(), and
    reconstruct an updated beta/alpha/emittance from the three variances.
    Returns (history, Dt, Dpt) where history is a list of
    (iteration, eps, beta, alfa) tuples, one per iteration, so the
    convergence itself can be plotted -- distinct from filter_by_action(),
    which this is meant to be compared against, not replace."""
    pos = df[position].values
    ang = df[angle].values
    delta = (df[p_col].values - p0_mev) / p0_mev

    pos_av, ang_av, delta_av = pos.mean(), ang.mean(), delta.mean()
    var_delta = np.var(delta)

    Dt = np.cov(pos, delta, ddof=0)[0, 1] / var_delta
    Dpt = np.cov(ang, delta, ddof=0)[0, 1] / var_delta

    pos_beta = pos - Dt * (delta - delta_av)
    ang_beta = ang - Dpt * (delta - delta_av)

    xxbet = np.mean((pos - pos_av - Dt * (delta - delta_av)) ** 2)
    xpxpbet = np.mean((ang - ang_av - Dpt * (delta - delta_av)) ** 2)
    xxpbet = np.mean((pos - pos_av - Dt * (delta - delta_av)) * (ang - ang_av - Dpt * (delta - delta_av)))
    eps_ref = np.sqrt(xxbet * xpxpbet - xxpbet ** 2)
    beta_ref, alfa_ref = xxbet / eps_ref, -xxpbet / eps_ref

    history = []
    for it in range(iters):
        xi = pos_beta / np.sqrt(beta_ref)
        xip = np.sqrt(beta_ref) * ang_beta + alfa_ref * pos_beta / np.sqrt(beta_ref)
        xi1 = (-xi + np.sqrt(3) * xip) / 2
        xi2 = (xi + np.sqrt(3) * xip) / 2

        _, xivar = getrms_gaussfit(xi, maxnmax)
        _, xi1var = getrms_gaussfit(xi1, maxnmax)
        _, xi2var = getrms_gaussfit(xi2, maxnmax)

        betnoreps = xivar
        alfnoreps = (xi1var - xi2var) / np.sqrt(3)
        gamnoreps = (2 * (xi1var + xi2var) - xivar) / 3
        eps = np.sqrt(betnoreps * gamnoreps - alfnoreps ** 2)
        betn, alfn = betnoreps / eps, alfnoreps / eps
        alfa_ref = betn * alfa_ref + alfn
        beta_ref = betn * beta_ref

        history.append((it + 1, eps, beta_ref, alfa_ref))
    return history, Dt, Dpt


def filter_by_moment_core(df_in, position, angle, iters=5, max_passes=20,
                          p_col='p[MeV/c]', p0_mev=2860.0):
    df_iter = df_in.copy()
    for _ in range(max_passes):
        history, Dt, Dpt = moment_based_twiss(df_iter, position, angle,
                                              p_col=p_col, p0_mev=p0_mev, iters=iters)
        _, eps, beta, alfa = history[-1]
        delta = (df_iter[p_col].values - p0_mev) / p0_mev
        pos = df_iter[position].values - Dt * (delta - delta.mean())
        ang = df_iter[angle].values - Dpt * (delta - delta.mean())
        pos = pos - pos.mean()
        ang = ang - ang.mean()
        gamma = (1 + alfa**2) / beta
        J = gamma * pos**2 + 2 * alfa * pos * ang + beta * ang**2
        J_max = 2 * eps * np.log(2 * len(df_iter))
        mask = J <= J_max
        if mask.all():
            break
        df_iter = df_iter[mask].copy()
    return df_iter, {'eps': eps, 'beta': beta, 'alfa': alfa, 'J_max': J_max, 'Dt': Dt, 'Dpt': Dpt}


def twiss_from_beam(u, up):
    cov = np.cov(u, up, ddof=0)
    eps = np.sqrt(np.linalg.det(cov))
    return cov[0, 0] / eps, -cov[0, 1] / eps, eps


def rematch(u, up, b_beam, a_beam, b_ring, a_ring):
    """Linear map that keeps the emittance but gives the beam the ring Twiss."""
    un = u / np.sqrt(b_beam)
    upn = (a_beam * u + b_beam * up) / np.sqrt(b_beam)
    return np.sqrt(b_ring) * un, (upn - a_ring * un) / np.sqrt(b_ring)


def load_beam():
    beta0 = ring.particle_ref.beta0[0]
    if os.path.exists(BEAM_FILE):
        df_raw = pd.read_csv(BEAM_FILE, sep=r'\s+')
        # moment-based core in x then y, as macroparticles.py (df_clean), then the same sample
        df_core_x, _ = filter_by_moment_core(df_raw, 'x[mm]', 'xp[mrad]', iters=MOMENT_ITERS)
        df_clean, _ = filter_by_moment_core(df_core_x, 'y[mm]', 'yp[mrad]', iters=MOMENT_ITERS)
        print(f'Moment-based core: kept {len(df_clean)}/{len(df_raw)} particles '
              f'({100*len(df_clean)/len(df_raw):.1f}%)')
        df = df_clean.sample(n=min(N_PARTICLES, len(df_clean)), random_state=SEED)
        x, px = df['x[mm]'].values * 1e-3, df['xp[mrad]'].values * 1e-3
        y, py = df['y[mm]'].values * 1e-3, df['yp[mrad]'].values * 1e-3
        z = df['t[mm/c]'].values * 1e-3                  # c*t [m], larger = later
        p = df['p[MeV/c]'].values
        delta = p / p.mean() - 1
        zeta = -(z - z.mean()) * beta0                  # xsuite: zeta > 0 = ahead
        source = BEAM_FILE
    else:
        print(f'WARNING: {BEAM_FILE} not found -> Gaussian stand-in beam {FALLBACK_BEAM}')
        gamma0 = ring.particle_ref.gamma0[0]
        ex = FALLBACK_BEAM['nemitt_x'] / (beta0 * gamma0)
        ey = FALLBACK_BEAM['nemitt_y'] / (beta0 * gamma0)
        n = N_PARTICLES
        # matched Gaussian, built in normalised coordinates
        xn, pxn, yn, pyn = rng.normal(size=(4, n))
        x, px = np.sqrt(ex * bx0) * xn, np.sqrt(ex / bx0) * (pxn - ax0 * xn)
        y, py = np.sqrt(ey * by0) * yn, np.sqrt(ey / by0) * (pyn - ay0 * yn)
        zeta = FALLBACK_BEAM['sigma_z'] * rng.normal(size=n)
        delta = FALLBACK_BEAM['sigma_delta'] * rng.normal(size=n)
        source = 'Gaussian stand-in'

    # centre the bunch, then remove the correlation with delta (beam dispersion)
    x, px, y, py, delta = [a - a.mean() for a in (x, px, y, py, delta)]
    var_d = np.var(delta)
    for a in (x, px, y, py):
        a -= np.cov(a, delta, ddof=0)[0, 1] / var_d * delta
    return dict(x=x, px=px, y=y, py=py, zeta=zeta, delta=delta, source=source)


beam = load_beam()
bbx, abx, ex_inj = twiss_from_beam(beam['x'], beam['px'])
bby, aby, ey_inj = twiss_from_beam(beam['y'], beam['py'])
print(f'Injected beam ({beam["source"]}): eps_x = {ex_inj*1e9:.1f} nm, eps_y = {ey_inj*1e9:.1f} nm, '
      f'sigma_delta = {beam["delta"].std():.2e}, sigma_z = {beam["zeta"].std()*1e3:.2f} mm')
print(f'  beam Twiss: betx = {bbx:.2f}, alfx = {abx:.2f} | ring: betx = {bx0:.2f}, alfx = {ax0:.2f}')

# Mismatch factor (1 = matched): B = (b_r g_b - 2 a_r a_b + g_r b_b) / 2
def mismatch(b1, a1, b2, a2):
    return 0.5 * (b1 * (1 + a2**2) / b2 - 2 * a1 * a2 + b2 * (1 + a1**2) / b1)
Bmag_x, Bmag_y = mismatch(bx0, ax0, bbx, abx), mismatch(by0, ay0, bby, aby)
print(f'  mismatch factor B of the beam file: x {Bmag_x:.2f}, y {Bmag_y:.2f}'
      f'{" (removed by matching)" if MATCH_TWISS else " (emittance grows by B after filamentation)"}')

if MATCH_TWISS:
    beam['x'], beam['px'] = rematch(beam['x'], beam['px'], bbx, abx, bx0, ax0)
    beam['y'], beam['py'] = rematch(beam['y'], beam['py'], bby, aby, by0, ay0)
    print('  beam rematched to ring Twiss at inj_point')

# %%
# ============================================================
# INJECTION GEOMETRY: offset, angle and kicker strength
# ============================================================
side = INJ['side']
sig_x_inj = np.sqrt(ex_inj * (bx0 if MATCH_TWISS else bbx))
R_required = 3 * N_SIGMA * sig_x_inj + BLADE_THICKNESS      # pipe needed for N_SIGMA clearance
n_clear = N_SIGMA                                          # clearance actually used [sigma]
if ON_AXIS:
    x_inj = 0.0
else:
    if X_BLADE == 'optimum':
        if R_PIPE is not None:
            # Clearance must also hold between septum and kicker, where beta is larger. With the
            # linear optics the centroid there is A(s) x_s (x'_s is proportional to x_s) and the size
            # is sigma(s) = sqrt(eps beta(s)); with x_s = 2 n sigma + t, fitting in the pipe gives
            #     n <= (R - A t) / (2 A sigma + sigma(s))     (A = 1, sigma(s) = sigma at the septum)
            _ik = list(tw.name).index('inj_kicker')
            _phi = 2 * np.pi * (tw.mux[:_ik + 1] - tw.mux[0])
            _b = tw.betx[:_ik + 1]
            _dmu = _phi[-1]
            _xp_over_x = -(np.cos(_dmu) + ax0 * np.sin(_dmu)) / (bx0 * np.sin(_dmu))
            _A = (np.sqrt(_b / bx0) * (np.cos(_phi) + ax0 * np.sin(_phi))
                  + np.sqrt(bx0 * _b) * np.sin(_phi) * _xp_over_x)
            _sig_s = np.sqrt(ex_inj * _b)
            _Ap = np.clip(_A, 0, None)
            _n_s = (R_PIPE - _Ap * BLADE_THICKNESS) / (2 * _Ap * sig_x_inj + _sig_s)
            _i = int(np.argmin(_n_s))
            n_fit = float(_n_s[_i])
            if n_fit < N_SIGMA:
                n_clear = n_fit
                print(f'\nNOTE: {N_SIGMA:g} sigma clearance needs R >= 3 n sigma + t = {R_required*1e3:.1f} mm '
                      f'at the septum; pipe is {R_PIPE*1e3:.2f} mm -> using the largest clearance that fits '
                      f'from septum to kicker, {n_clear:.2f} sigma (limited at s = {tw.s[_i]:.2f} m, '
                      f'{tw.name[_i]})')
        INJ['x_blade'] = n_clear * sig_x_inj
        for nn in ('inj_septum_entry', 'inj_septum_exit'):     # move the blade in the lattice
            if side > 0:
                ring[nn].max_x = INJ['x_blade']
            else:
                ring[nn].min_x = -INJ['x_blade']
        x_auto = 2 * n_clear * sig_x_inj + BLADE_THICKNESS
    else:
        x_auto = INJ['x_blade'] + BLADE_THICKNESS + N_SIGMA * sig_x_inj
    x_inj = x_auto if X_INJ == 'auto' else float(X_INJ)
    print(f"\nSeptum: blade {INJ['x_blade']*1e3:.2f} mm ({'optimum' if X_BLADE == 'optimum' else 'fixed'}), "
          f"thickness {BLADE_THICKNESS*1e3:.1f} mm, sigma_x = {sig_x_inj*1e3:.2f} mm, "
          f"clearance {n_clear:.2f} sigma, bunch centre {x_inj*1e3:.2f} mm, "
          f"outer edge {(x_inj + n_clear*sig_x_inj)*1e3:.2f} mm"
          + (f" (pipe {R_PIPE*1e3:.2f} mm)" if R_PIPE is not None else ''))

i_k = list(tw.name).index('inj_kicker')
dmu = 2 * np.pi * (tw.mux[i_k] - tw.mux[0])
px_inj = -x_inj * (np.cos(dmu) + ax0 * np.sin(dmu)) / (bx0 * np.sin(dmu))

def make_particles(dx_, dpx_, x, px, y, py, zeta, delta):
    """Absolute coordinates = closed orbit + centroid offset + dispersion + beam."""
    d = co0['delta'] + DELTA_OFFSET + delta
    return ring.build_particles(
        x=co0['x'] + side * dx_ + dx0 * (d - co0['delta']) + x,
        px=co0['px'] + side * dpx_ + dpx0 * (d - co0['delta']) + px,
        y=co0['y'] + dy0 * (d - co0['delta']) + y,
        py=co0['py'] + dpy0 * (d - co0['delta']) + py,
        zeta=co0['zeta'] + zeta, delta=d)

# kicker strength from the linear optics between septum exit and kicker (only quadrupoles there):
# angle of the centroid at the kicker, which the kick cancels (xtrack: dpx = -knl[0])
bk, ak = tw.betx[i_k], tw.alfx[i_k]
xp_at_kicker = side * (((ax0 - ak) * np.cos(dmu) - (1 + ax0 * ak) * np.sin(dmu)) / np.sqrt(bx0 * bk) * x_inj
                       + np.sqrt(bx0 / bk) * (np.cos(dmu) - ak * np.sin(dmu)) * px_inj)
k_nominal = xp_at_kicker

# check with one on-momentum particle tracked through the real lattice (mean radiation)
p1 = make_particles(x_inj, px_inj, *np.zeros((6, 1)))
ring.track(p1, ele_stop='inj_kicker')
x_at_kicker = p1.x[0] - tw.x[i_k] if p1.state[0] > 0 else np.nan
if R_PIPE is not None and x_inj >= R_PIPE:
    print(f'\nWARNING: injected centroid x_inj = {x_inj*1e3:.2f} mm is outside the pipe '
          f'(r = {R_PIPE*1e3:.2f} mm). Use X_BLADE=optimum or a smaller X_INJ.')
elif p1.state[0] <= 0:
    print('\nWARNING: the on-momentum centroid is lost before the kicker')

Brho = ring.particle_ref.p0c[0] / 299792458.0
print(f"\nInjection: x_inj = {side*x_inj*1e3:.2f} mm, injection angle = {side*px_inj*1e3:.3f} mrad, "
      f"dmu(septum->kicker) = {dmu/2/np.pi:.4f} [2pi] = {np.degrees(dmu):.1f} deg")
print(f"  centroid at kicker: x = {x_at_kicker*1e3:.3f} mm (should be ~0)" if not ON_AXIS else
      "  ON_AXIS reference run: injected on the closed orbit, kicker off, no blade")
print(f"  kicker deflection = {-k_nominal*1e3:.3f} mrad, int B dl = {abs(k_nominal)*Brho*1e3:.1f} mT m "
      f"(B rho = {Brho:.2f} T m)")

# %%
# ============================================================
# TRACKING
#   turn 0: element-by-element, kicker ON
#   turns 1..N: kicker OFF, chunks of CHUNK turns, moments per turn
# ============================================================
if N_TURNS == 'auto':
    N_TURNS = int(3 * tau_turns[0])
N_TURNS = int(N_TURNS)

def moments(st, x, px, y, py, zeta, delta):
    """Beam moments per column (turn) of surviving particles. Inputs (npart, nturns)."""
    alive = st > 0
    nan = lambda a: np.where(alive, a, np.nan)
    d = nan(delta) - co0['delta']
    xb = nan(x) - co0['x'] - dx0 * d          # betatron part
    pxb = nan(px) - co0['px'] - dpx0 * d
    yb = nan(y) - co0['y'] - dy0 * d
    pyb = nan(py) - co0['py'] - dpy0 * d

    def emit(u, up):
        u_c, up_c = u - np.nanmean(u, 0), up - np.nanmean(up, 0)
        return np.sqrt(np.nanmean(u_c**2, 0) * np.nanmean(up_c**2, 0) - np.nanmean(u_c * up_c, 0)**2)

    with np.errstate(invalid='ignore'), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return dict(
            n_alive=alive.sum(0),
            x_mean=np.nanmean(nan(x) - co0['x'], 0), y_mean=np.nanmean(nan(y) - co0['y'], 0),
            sig_x=np.nanstd(nan(x), 0), sig_y=np.nanstd(nan(y), 0),
            sig_z=np.nanstd(nan(zeta), 0), sig_delta=np.nanstd(nan(delta), 0),
            eps_x=emit(xb, pxb), eps_y=emit(yb, pyb))

particles = make_particles(x_inj, px_inj, beam['x'], beam['px'], beam['y'], beam['py'],
                           beam['zeta'], beam['delta'])
n0 = len(particles.x)

# particles that start inside the blade (field side) never make it through the septum
lost_on_septum = (np.zeros(n0, bool) if ON_AXIS else
                  side * (particles.x - co0['x']) < INJ['x_blade'] + BLADE_THICKNESS)
particles.state[lost_on_septum] = 0
print(f'\n{lost_on_septum.sum()} / {n0} particles hit the septum blade on the way in')

ring.configure_radiation(model='quantum')
pdr['k_inj'] = k_nominal * (1 + KICK_ERROR)

snap = {}
def take_snap(turn, p_or_mon, j=None):
    get = (lambda a: getattr(p_or_mon, a).copy()) if j is None else (lambda a: getattr(p_or_mon, a)[:, j].copy())
    snap[turn] = {a: get(a) for a in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'state']}

take_snap(0, particles)
mom = [moments(*[getattr(particles, a)[:, None] for a in ['state', 'x', 'px', 'y', 'py', 'zeta', 'delta']])]

EBE_KEYS = ['x', 'px', 'y', 'py', 'zeta', 'delta', 'state', 's']
def track_ebe_turn():
    """Track one turn recording every element (for the envelope plots)."""
    ring.track(particles, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
    m_ = ring.record_last_track
    return {a: getattr(m_, a).copy() for a in EBE_KEYS}

def moments_from_ebe(e):
    """Moments at the start of an element-by-element turn (= at inj_point)."""
    return moments(*[e[a][:, :1] for a in ['state', 'x', 'px', 'y', 'py', 'zeta', 'delta']])

# turn 0: injection, kicker ON
t_start = time.time()
ebe = track_ebe_turn()
pdr['k_inj'] = 0.0                                   # kicker off from turn 1 on

# clearance between septum and kicker on turn 0 (higher beta than at the septum)
if not ON_AXIS and R_PIPE is not None:
    with np.errstate(invalid='ignore'):
        _al = ebe['state'] > 0
        _c = np.nanmean(np.where(_al, ebe['x'], np.nan), 0)
        _sg = np.nanstd(np.where(_al, ebe['x'], np.nan), 0)
        _ss = ebe['s'][np.argmax(_al.sum(1))]
        _m = _ss <= tw['s', 'inj_kicker']
        _edge = side * _c[_m] + n_clear * _sg[_m]
    print(f'Septum -> kicker: largest |centroid| + {n_clear:.2f} sigma = {np.nanmax(_edge)*1e3:.2f} mm '
          f'at s = {_ss[_m][np.nanargmax(_edge)]:.2f} m (pipe {R_PIPE*1e3:.2f} mm)')
print(f'Turn 0 done in {time.time()-t_start:.1f}s; tracking {N_TURNS} turns '
      f'(estimated {(time.time()-t_start)*N_TURNS/60:.0f} min)...')

# turn 1: first turn as a stored beam, element by element
ebe_stored = {}
if N_TURNS > 1 and np.any(particles.state > 0):
    ebe_stored[1] = track_ebe_turn()
    mom.append(moments_from_ebe(ebe_stored[1]))
    if 1 in SNAP_TURNS:
        snap[1] = {a: ebe_stored[1][a][:, 0].copy() for a in ['x', 'px', 'y', 'py', 'zeta', 'delta', 'state']}

# turns 2 .. N_TURNS-2 in chunks
turn = 2
while turn < N_TURNS - 1 and np.any(particles.state > 0):
    nt = min(CHUNK, N_TURNS - 1 - turn)
    # explicit monitor: the default one always starts recording at turn 0
    m = xt.ParticlesMonitor(start_at_turn=turn, stop_at_turn=turn + nt, num_particles=n0)
    ring.track(particles, num_turns=nt, turn_by_turn_monitor=m)
    mom.append(moments(m.state, m.x, m.px, m.y, m.py, m.zeta, m.delta))
    for ts in SNAP_TURNS:
        if turn <= ts < turn + nt:
            take_snap(ts, m, ts - turn)
    turn += nt
    print(f'  turn {turn}/{N_TURNS}: {np.sum(particles.state > 0)} alive', end='\r')

# last turn: stored beam after N_TURNS, element by element
if N_TURNS > 2 and np.any(particles.state > 0):
    ebe_stored[turn] = track_ebe_turn()
    mom.append(moments_from_ebe(ebe_stored[turn]))
    turn += 1
take_snap(turn, particles)
print(f'\nTracking finished in {(time.time()-t_start)/60:.1f} min')

mom = {k: np.concatenate([mm[k] for mm in mom]) for k in mom[0]}
turns = np.arange(len(mom['n_alive']))

# %%
# ============================================================
# SAVE
# ============================================================
pd.DataFrame(dict(turn=turns, **mom)).to_csv(f'{folder}/beam_moments_vs_turn.csv', index=False)

survived = int(np.sum(particles.state > 0))
summary = dict(
    beam_source=beam['source'], n_particles=n0, n_turns=int(turns[-1] + 1),
    efficiency_percent=100 * survived / n0, lost_on_septum=int(lost_on_septum.sum()),
    n_sigma=N_SIGMA, x_inj_m=float(side * x_inj),
    injection_angle_mrad=float(side * px_inj * 1e3),          # beam angle at the septum exit
    kicker_deflection_mrad=float(-k_nominal * 1e3),            # dpx given by the kicker (= -knl0)
    kicker_intBdl_Tm=float(abs(k_nominal) * Brho), kick_error=KICK_ERROR,
    s_septum_m=[s_lattice['inj_septum_entry'], s_lattice['inj_septum_exit']],   # s in the lattice file
    s_kicker_m=s_lattice['inj_kicker'],
    dmu_septum_kicker=float(dmu / 2 / np.pi), dmu_septum_kicker_deg=float(np.degrees(dmu)),
    mismatch_x=float(Bmag_x), mismatch_y=float(Bmag_y), matched_twiss=MATCH_TWISS,
    eps_x_inj=float(ex_inj), eps_y_inj=float(ey_inj), eq_gemitt_x=float(tw.eq_gemitt_x),
    damping_turns=[float(t) for t in tau_turns], pipe_radius=R_PIPE, on_axis=ON_AXIS,
    blade_mode=X_BLADE if X_BLADE == 'optimum' else 'fixed', blade_thickness=BLADE_THICKNESS,
    sigma_x_septum=float(sig_x_inj), n_sigma_clearance=float(n_clear), pipe_radius_required_m=float(R_required), **{k: v for k, v in INJ.items()})
with open(f'{folder}/injection_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
print(f'Injection efficiency: {summary["efficiency_percent"]:.1f} %  -> {folder}')

# %%
# ============================================================
# PLOTS
# ============================================================
plt.style.use('default')
s_tw = tw.s
s_kick = tw['s', 'inj_kicker']
C = tw.circumference
# the ring starts at inj_point, so the septum sits at the very end: show it at s < 0
s_sep = (tw['s', 'inj_septum_entry'] - C, tw['s', 'inj_septum_exit'] - C)
s_tw_wrapped = np.where(tw.s > C - 10, tw.s - C, tw.s)
order_tw = np.argsort(s_tw_wrapped)

# chamber: one segment per magnet (ap_<name>_in -> ap_<name>_out) and the septum blade
_tab = ring.get_table()
_s = dict(zip(_tab.name, _tab.s))
ap_seg = []          # (s0, s1, +x, -x, +y, -y)
for nn in _tab.name:
    if nn.startswith('ap_') and nn.endswith('_in') and nn[:-3] + '_out' in _s:
        e = ring[nn]
        ap_seg.append((_s[nn], _s[nn[:-3] + '_out'], e.a, -e.a, e.b, -e.b))
if R_PIPE is not None and not ON_AXIS:
    e = ring['inj_septum_entry']
    ap_seg.append((_s['inj_septum_entry'], _s['inj_septum_exit'], e.max_x, e.min_x, e.max_y, e.min_y))
ap_seg = np.array(ap_seg).reshape(-1, 6)

def sigma_eq(plane):
    if plane == 'x':
        return np.sqrt(tw.eq_gemitt_x * tw.betx + (tw.dx * sig_delta_eq)**2)
    return np.sqrt(max(tw.eq_gemitt_y, 1e-13) * tw.bety)

# ---- 1 & 2: beam envelope at injection (injection region zoom and full ring) ----
import warnings; warnings.simplefilter('ignore', RuntimeWarning)

def envelope(e):
    """Centroid and rms size per element of an element-by-element turn."""
    al = e['state'] > 0
    if al.sum() == 0:
        return None
    with np.errstate(invalid='ignore'):
        out = {k: f(np.where(al, e[k], np.nan), 0) for k, f in [('x', np.nanmean), ('y', np.nanmean)]}
        out.update({f's{k}': np.nanstd(np.where(al, e[k], np.nan), 0) for k in ['x', 'y']})
    out['s'] = e['s'][np.argmax(al.sum(1))]           # s of a surviving particle at each element
    return out

def density_map(e, coord, s_grid, u_edges):
    """Particle density of an element-by-element turn: each particle's coordinate is
    interpolated onto a uniform s grid (linear in drifts) up to where it is lost, then
    histogrammed. Returns counts / (N injected * bin width in mm) = fraction per mm."""
    al = e['state'] > 0
    s_ref = e['s'][np.argmax(al.sum(1))]
    ss_all, uu_all = [], []
    for i in range(al.shape[0]):
        if not al[i].any():
            continue
        s_last = s_ref[al[i]].max()
        ok = s_grid <= s_last
        ss_all.append(s_grid[ok])
        uu_all.append(np.interp(s_grid[ok], s_ref[al[i]], e[coord][i][al[i]]))
    ds = s_grid[1] - s_grid[0]
    s_edges = np.r_[s_grid - ds / 2, s_grid[-1] + ds / 2]
    H, _, _ = np.histogram2d(np.concatenate(ss_all), np.concatenate(uu_all), bins=[s_edges, u_edges])
    return s_edges, H / (al.shape[0] * np.diff(u_edges)[0] * 1e3)

from matplotlib.colors import LogNorm
u_lim = 1.3 * (R_PIPE if R_PIPE is not None else 0.03)          # plotted half range [m]
u_edges = np.linspace(-u_lim, u_lim, 161)

env0 = envelope(ebe)
alive0 = ebe['state'] > 0
cx, sx, cy, sy, s_ebe = env0['x'], env0['sx'], env0['y'], env0['sy'], env0['s']
env_stored = {t: envelope(e) for t, e in ebe_stored.items()}
stored_style = {0: ('C4', '--'), 1: ('C5', '-.')}

for tag, xlim in [('injection_region', (s_sep[0] - 1, s_kick + 6)), ('full_ring', (0, C))]:
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)
    s_st = s_tw_wrapped[order_tw] if tag == 'injection_region' else s_tw
    idx = order_tw if tag == 'injection_region' else slice(None)
    s_grid = np.arange(0, (s_kick + 6) if tag == 'injection_region' else C, 0.02 if tag == 'injection_region' else 0.1)
    for ax_, c, s_, lab, sig_e in [(axs[0], cx, sx, 'x', sigma_eq('x')),
                                   (axs[1], cy, sy, 'y', sigma_eq('y'))]:
        # density of the injected bunch on turn 0 (log scale, so sparse tails stay visible)
        s_edges, D = density_map(ebe, lab, s_grid, u_edges)
        D = np.ma.masked_less_equal(D, 0)
        pcm = ax_.pcolormesh(s_edges, 1e3 * u_edges, D.T, cmap='Blues', shading='flat', rasterized=True,
                             norm=LogNorm(vmin=1 / (n0 * np.diff(u_edges)[0] * 1e3), vmax=D.max()), zorder=0)
        fig.colorbar(pcm, ax=ax_, pad=0.01, label='injected bunch (turn 0):\nfraction of particles per mm')
        for sign in (+1, -1):
            ax_.plot(s_ebe, 1e3 * (c + sign * N_SIGMA * s_), 'C0', lw=1,
                     label=f'injected bunch, $\\pm{N_SIGMA:g}\\sigma$ (turn 0)' if sign > 0 else None)
        ax_.plot(s_ebe, 1e3 * c, 'C0', lw=1.5, label='injected centroid')
        ax_.fill_between(s_st, 1e3 * (tw[lab][idx] - N_SIGMA * sig_e[idx]), 1e3 * (tw[lab][idx] + N_SIGMA * sig_e[idx]),
                         color='C1', alpha=0.5, label=f'stored beam at equilibrium, $\\pm{N_SIGMA:g}\\sigma$')
        # the injected bunch once it is stored: turn 1 and the last tracked turn.
        # In the zoom, the end of each turn (beam coming back to the septum) is drawn at s < 0.
        for k_, (t_, en) in enumerate(env_stored.items()):
            if en is None:
                continue
            ss = en['s'].copy()
            col, ls = stored_style.get(k_, ('C6', ':'))
            # zoom: start of the turn at s >= 0, end of the same turn (back at the septum)
            # at s < 0, drawn as separate pieces
            pieces = ([ss <= C - 10, ss > C - 10] if tag == 'injection_region' else [np.ones_like(ss, bool)])
            for j_, pc in enumerate(pieces):
                sp = ss[pc] - (C if (tag == 'injection_region' and j_ == 1) else 0)
                cc, sg = en[lab][pc], en['s' + lab][pc]
                for sign in (+1, -1):
                    ax_.plot(sp, 1e3 * (cc + sign * N_SIGMA * sg), color=col, ls=ls, lw=1.3,
                             label=(f'stored beam, turn {t_}, $\\pm{N_SIGMA:g}\\sigma$'
                                    if (j_ == 0 and sign > 0) else None))
        cols = (2, 3) if lab == 'x' else (4, 5)
        s0, s1 = ap_seg[:, 0].copy(), ap_seg[:, 1].copy()
        if tag == 'injection_region':
            wrap = s0 > C - 10
            s0[wrap] -= C; s1[wrap] -= C
        for c_ in cols:
            ax_.hlines(1e3 * ap_seg[:, c_], s0, s1, color='k', lw=1.5,
                       label='chamber aperture' if c_ == cols[0] else None)
        ax_.axvline(s_kick, color='C3', ls='--', label='kicker')
        ax_.set_ylabel(f'{lab} [mm]'); ax_.grid(alpha=0.3)
        ax_.set_ylim(-1e3 * u_lim, 1e3 * u_lim)
    if not ON_AXIS:
        axs[0].plot(s_sep, [side * 1e3 * INJ['x_blade']] * 2, color='C2', lw=4, label='septum blade')
        axs[0].plot(s_sep, [side * 1e3 * (INJ['x_blade'] + BLADE_THICKNESS)] * 2, color='C2', lw=4)
    axs[0].legend(fontsize=7.5, loc='lower right', ncol=2)
    axs[1].set_xlabel('s from injection point [m]'); axs[1].set_xlim(*xlim)
    axs[0].set_title(f'Beam envelope at injection (D{design} C{config} {phase}deg{"_" + changes if changes else ""}, {mode})')
    plt.tight_layout(); plt.savefig(f'{folder}/envelope_{tag}.png', dpi=150); plt.close()

# ---- 3: phase space at the septum exit + normalised phase space at the kicker ----
i_k_ebe = int(np.argmin(np.abs(s_ebe - s_kick)))
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
x0 = 1e3 * (snap[0]['x'] - co0['x']); px0 = 1e3 * (snap[0]['px'] - co0['px'])
ok = ~lost_on_septum
axs[0].scatter(x0[ok], px0[ok], s=2, label='injected')
axs[0].scatter(x0[~ok], px0[~ok], s=2, color='C3', label='hits blade')
th = np.linspace(0, 2 * np.pi, 200)
xe = N_SIGMA * np.sqrt(tw.eq_gemitt_x * bx0) * np.cos(th)
axs[0].plot(1e3 * xe, 1e3 * N_SIGMA * np.sqrt(tw.eq_gemitt_x / bx0) * (-ax0 * np.cos(th) - np.sin(th)),
            'C1', label=f'stored beam ${N_SIGMA:g}\\sigma$')
if not ON_AXIS:
    axs[0].axvline(side * 1e3 * INJ['x_blade'], color='C2', lw=3, label='blade')
    axs[0].axvline(side * 1e3 * (INJ['x_blade'] + BLADE_THICKNESS), color='C2', lw=3)
axs[0].set_xlabel('x - x$_{co}$ [mm]'); axs[0].set_ylabel("x' - x'$_{co}$ [mrad]")
axs[0].set_title('Septum exit, turn 0'); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

bk, ak = tw.betx[i_k], tw.alfx[i_k]
for j, lab in [(i_k_ebe, 'before kick'), (i_k_ebe + 1, 'after kick')]:
    a = alive0[:, j]
    u = ebe['x'][a, j] - tw.x[i_k]; up = ebe['px'][a, j] - tw.px[i_k]
    axs[1].scatter(1e3 * u / np.sqrt(bk), 1e3 * (ak * u + bk * up) / np.sqrt(bk), s=2, label=lab)
axs[1].set_xlabel('X [$\\sqrt{\\mathrm{mm\\,mrad}}$]'); axs[1].set_ylabel("X'")
axs[1].set_title('Normalised phase space at the kicker'); axs[1].axis('equal')
axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{folder}/phase_space_septum_kicker.png', dpi=150); plt.close()

# ---- 4: beam sizes vs turn ----
sig_x_eq0, sig_y_eq0 = sigma_eq('x')[0], sigma_eq('y')[0]
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax_, k, eq, lab, sc in [(axs[0, 0], 'sig_x', sig_x_eq0, '$\\sigma_x$ [mm]', 1e3),
                            (axs[0, 1], 'sig_y', sig_y_eq0, '$\\sigma_y$ [mm]', 1e3),
                            (axs[1, 0], 'sig_z', sig_z_eq, '$\\sigma_z$ [mm]', 1e3),
                            (axs[1, 1], 'sig_delta', sig_delta_eq, '$\\sigma_\\delta$ [$10^{-3}$]', 1e3)]:
    ax_.plot(turns + 1, sc * mom[k], lw=1)
    ax_.axhline(sc * eq, color='k', ls='--', label='equilibrium')
    ax_.set_ylabel(lab); ax_.set_xscale('log'); ax_.grid(alpha=0.3, which='both'); ax_.legend(fontsize=8)
for ax_ in axs[1]:
    ax_.set_xlabel('turn + 1')
fig.suptitle('Beam size at the injection point vs turn')
plt.tight_layout(); plt.savefig(f'{folder}/beam_size_vs_turn.png', dpi=150); plt.close()

# ---- 5: emittance vs turn with damping fit ----
from scipy.optimize import curve_fit
def damp(t, e0, tau):
    return tw.eq_gemitt_x + (e0 - tw.eq_gemitt_x) * np.exp(-2 * t / tau)

fig, ax_ = plt.subplots(figsize=(10, 5))
ax_.plot(turns, 1e9 * mom['eps_x'], label='$\\epsilon_x$')
ax_.plot(turns, 1e9 * mom['eps_y'], label='$\\epsilon_y$')
ax_.axhline(1e9 * tw.eq_gemitt_x, color='k', ls='--', label='$\\epsilon_{x,eq}$')
ax_.axhline(1e9 * ex_inj * (1 if MATCH_TWISS else Bmag_x), color='C0', ls=':',
            label='$B_x\\,\\epsilon_{x,inj}$ (filamented)')
fit_ok = np.isfinite(mom['eps_x'])
if turns[fit_ok][-1] > 0.5 * tau_turns[0]:
    try:
        popt, _ = curve_fit(damp, turns[fit_ok][10:], mom['eps_x'][fit_ok][10:], p0=[mom['eps_x'][20], tau_turns[0]])
        ax_.plot(turns, 1e9 * damp(turns, *popt), 'C3--',
                 label=f'fit: $\\tau_x$ = {popt[1]:.0f} turns (twiss {tau_turns[0]:.0f})')
    except RuntimeError:
        print('Damping fit did not converge')
ax_.set_yscale('log'); ax_.set_xlabel('turn'); ax_.set_ylabel('rms emittance [nm]')
ax_.grid(alpha=0.3, which='both'); ax_.legend(fontsize=8)
ax_.set_title('Betatron emittance at the injection point')
plt.tight_layout(); plt.savefig(f'{folder}/emittance_vs_turn.png', dpi=150); plt.close()

# ---- 6: centroid vs turn ----
n_show = min(len(turns), 2000)
fig, ax_ = plt.subplots(figsize=(10, 4))
ax_.plot(turns[:n_show], 1e3 * mom['x_mean'][:n_show], label='$\\langle x \\rangle$')
ax_.plot(turns[:n_show], 1e3 * mom['y_mean'][:n_show], label='$\\langle y \\rangle$')
ax_.set_xlabel('turn'); ax_.set_ylabel('centroid - closed orbit [mm]'); ax_.grid(alpha=0.3); ax_.legend()
ax_.set_title('Residual injection oscillation')
plt.tight_layout(); plt.savefig(f'{folder}/centroid_vs_turn.png', dpi=150); plt.close()

# ---- 7: profiles at selected turns ----
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ts in sorted(snap):
    a = snap[ts]['state'] > 0
    if ts == 0 or a.sum() < 10:
        continue
    for ax_, k, co in [(axs[0], 'x', co0['x']), (axs[1], 'y', co0['y'])]:
        ax_.hist(1e3 * (snap[ts][k][a] - co), bins=60, histtype='step', density=True, label=f'turn {ts}')
for ax_, k in zip(axs, ['x', 'y']):
    ax_.set_xlabel(f'{k} - {k}$_{{co}}$ [mm]'); ax_.set_ylabel('density'); ax_.grid(alpha=0.3)
axs[0].legend(fontsize=8)
fig.suptitle('Beam profiles at the injection point')
plt.tight_layout(); plt.savefig(f'{folder}/profiles_at_turns.png', dpi=150); plt.close()

# ---- 8: losses ----
lost = (particles.state <= 0) & ~lost_on_septum
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(turns, 100 * mom['n_alive'] / n0)
axs[0].set_xlabel('turn'); axs[0].set_ylabel('surviving [%]'); axs[0].set_xscale('symlog')
axs[0].grid(alpha=0.3); axs[0].set_title(f'Efficiency {summary["efficiency_percent"]:.1f} %')
axs[1].hist(particles.s[lost], bins=np.linspace(0, C, 216))
axs[1].axvspan(s_sep[0] + C, s_sep[1] + C, color='C2', alpha=0.4, label='septum'); axs[1].axvline(s_kick, color='C3', ls='--', label='kicker')
axs[1].set_xlabel('s from injection point [m]'); axs[1].set_ylabel('lost particles'); axs[1].legend(fontsize=8)
axs[2].hist(particles.at_turn[lost], bins=50)
axs[2].set_xlabel('turn of loss'); axs[2].set_ylabel('lost particles')
axs[1].set_title(f'{lost.sum()} lost in ring, {lost_on_septum.sum()} on septum')
plt.tight_layout(); plt.savefig(f'{folder}/loss_map.png', dpi=150); plt.close()

# ---- 9: injected actions vs geometric acceptance ----
acc_x = np.min((A_X - np.abs(tw.x))**2 / tw.betx)       # 2J_max, ignoring dispersion
acc_y = np.min((A_Y - np.abs(tw.y))**2 / tw.bety)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax_, (u, up, b, a, acc, lab) in zip(axs, [
        ('x', 'px', bx0, ax0, acc_x, 'x'), ('y', 'py', by0, ay0, acc_y, 'y')]):
    a1 = snap[1]['state'] > 0 if 1 in snap else ok
    s1 = snap[1] if 1 in snap else snap[0]
    d = s1['delta'][a1] - co0['delta']
    uu = s1[u][a1] - co0[u] - (dx0 if u == 'x' else dy0) * d
    vv = s1[up][a1] - co0[up] - (dpx0 if u == 'x' else dpy0) * d
    twoJ = (uu**2 * (1 + a**2) / b + 2 * a * uu * vv + b * vv**2)
    ax_.hist(1e6 * twoJ, bins=60)
    ax_.axvline(1e6 * acc, color='k', ls='--', label='geometric acceptance')
    ax_.set_xlabel(f'$2J_{lab}$ [mm mrad]'); ax_.set_ylabel('particles'); ax_.legend(fontsize=8); ax_.grid(alpha=0.3)
fig.suptitle('Single-particle actions after injection (turn 1)')
plt.tight_layout(); plt.savefig(f'{folder}/actions_vs_acceptance.png', dpi=150); plt.close()

print(f'Plots saved to {folder}')

# %%
# ============================================================
# OPTIONAL SCANS (RUN_SCANS=1): efficiency vs kicker error and energy offset
# ============================================================
def quick_efficiency(kick_error=0.0, delta_offset=0.0):
    sub = rng.choice(n0, size=min(SCAN_PARTICLES, n0), replace=False)
    p = make_particles(x_inj, px_inj, *[beam[k][sub] for k in ['x', 'px', 'y', 'py', 'zeta']],
                       beam['delta'][sub] + delta_offset)
    p.state[side * (p.x - co0['x']) < INJ['x_blade'] + BLADE_THICKNESS] = 0
    pdr['k_inj'] = k_nominal * (1 + kick_error)
    ring.track(p, num_turns=1)
    pdr['k_inj'] = 0.0
    ring.track(p, num_turns=SCAN_TURNS - 1)
    return 100 * np.mean(p.state > 0)

if RUN_SCANS:
    scans = {'kick_error': np.linspace(-0.2, 0.2, 9), 'delta_offset': np.linspace(-0.01, 0.01, 9)}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    out = {}
    for ax_, (k, vals) in zip(axs, scans.items()):
        eff = [quick_efficiency(**{k: v}) for v in vals]
        out[k] = dict(values=vals.tolist(), efficiency=eff)
        ax_.plot(100 * vals, eff, 'o-')
        ax_.set_xlabel(f'{k} [%]'); ax_.set_ylabel(f'efficiency after {SCAN_TURNS} turns [%]'); ax_.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{folder}/scans.png', dpi=150); plt.close()
    with open(f'{folder}/scans.json', 'w') as f:
        json.dump(out, f, indent=4)
