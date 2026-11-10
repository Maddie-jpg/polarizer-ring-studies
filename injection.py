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

Lattice elements are installed on the fly with mc.install_apertures and
mc.install_injection (LatticeBuild/misalignments_corrections.py). The ring is
then cycled so s = 0 is the injection point: all turn-by-turn data are
recorded there.

Inputs   lattice from DESIGN/CONFIG/MODE/PHASE/CHANGES (paths.study_from_env)
         cylindrical chamber: radius = widest point of the macroparticles.py
         chamber design, max(chamber_r) in PhysicalAperture/{mode}/
         chamber_per_magnet_{mode}.csv (APERTURE_R overrides; fallback 31.2 mm)
         env: N_PARTICLES, N_TURNS, X_INJ, KICK_ERROR, MATCH_TWISS=1,
              ENERGY_COMPRESSOR=1, RUN_SCANS=1, SCAN_TURNS, APERTURE_R, N_SIGMA
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
SEED = 74
MAHALANOBIS_R = None     # optional 6-D core cut (as filter_beam_core in macroparticles.py); None = keep main bunch
USE_ECS = _on(os.environ.get('ENERGY_COMPRESSOR', 'false'))   # off by default: the beam file is already compressed
ECS = dict(R56=0.352963, Vdeb=-0.048365, Phasdeb=0.29229, freq=3e9)
MATCH_TWISS = _on(os.environ.get('MATCH_TWISS', '0'))         # rematch beam to ring Twiss at inj_point
DELTA_OFFSET = 0.0       # mean momentum offset of the injected bunch w.r.t. the ring

# Gaussian stand-in, ONLY used if BEAM_FILE is missing
FALLBACK_BEAM = dict(nemitt_x=13000e-6, nemitt_y=12000e-6, sigma_z=4.8e-3, sigma_delta=2e-3)

# --- apertures and injection hardware ---
APERTURE_R = os.environ.get('APERTURE_R', 'chamber')   # pipe radius [m], or 'chamber' = widest point of the design
APERTURE_R_FALLBACK = 0.0312     # [m] if no chamber design is found
INJ = dict(septum_at='DrTripl::1', kicker_at='DrTripl::2',
           septum_offset=0.05, septum_length=0.8, kicker_offset=2.2,
           x_blade=0.010, side=+1)
BLADE_THICKNESS = 0.002           # [m]
N_SIGMA = float(os.environ.get('N_SIGMA', 5))   # beam half-width in sigma: injection clearance and envelopes
X_INJ = os.environ.get('X_INJ', 'auto')   # bunch centre at septum exit [m]; 'auto' = blade + thickness + N_SIGMA sigma_x
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
else:
    R_PIPE = float(APERTURE_R)
A_X = A_Y = R_PIPE
mc.install_apertures(pdr, radius=R_PIPE)
mc.install_injection(pdr, radius=R_PIPE, **INJ)
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
# BEAM: linac distribution (+ ECS), centred, optional Twiss rematch
# ============================================================
def select_main_bunch(df, t_col='t[mm/c]', mad_k=10):
    """Same selection as macroparticles.py: drop satellites far from the median t."""
    t = df[t_col].values
    med = np.median(t)
    mad = np.median(np.abs(t - med))
    if mad == 0:
        return df
    return df[np.abs(t - med) < mad_k * mad * 1.4826]


def filter_beam_core(df, mahalanobis_r):
    cols = ['x[mm]', 'xp[mrad]', 'y[mm]', 'yp[mrad]', 't[mm/c]', 'p[MeV/c]']
    d = (df[cols] - df[cols].mean()).values
    inv = np.linalg.inv(np.cov(d, rowvar=False))
    keep = np.einsum('ij,jk,ik->i', d, inv, d) <= mahalanobis_r**2
    print(f'Core filter (Mahalanobis r = {mahalanobis_r}): kept {keep.sum()} / {len(df)}')
    return df[keep]


def apply_ecs(z, delta, R56, Vdeb, Phasdeb, freq):
    """Energy compressor: chicane (R56) then RF (Vdeb, Phasdeb). z in m."""
    k = 2 * np.pi * freq / 299792458.0
    z_f = z + R56 * delta
    return z_f, delta + Vdeb * np.sin(k * z_f + Phasdeb)


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
        df = select_main_bunch(pd.read_csv(BEAM_FILE, sep=r'\s+'))
        if MAHALANOBIS_R:
            df = filter_beam_core(df, MAHALANOBIS_R)
        df = df.sample(n=min(N_PARTICLES, len(df)), random_state=SEED)
        x, px = df['x[mm]'].values * 1e-3, df['xp[mrad]'].values * 1e-3
        y, py = df['y[mm]'].values * 1e-3, df['yp[mrad]'].values * 1e-3
        z = df['t[mm/c]'].values * 1e-3                  # c*t [m], larger = later
        p = df['p[MeV/c]'].values
        delta = p / p.mean() - 1
        if USE_ECS:
            z, delta = apply_ecs(z, delta, **ECS)
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
print(f'  mismatch factor B: x {Bmag_x:.2f}, y {Bmag_y:.2f} (emittance grows by B after filamentation)')

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
x_inj = (INJ['x_blade'] + BLADE_THICKNESS + N_SIGMA * sig_x_inj) if X_INJ == 'auto' else float(X_INJ)

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

# kicker strength from a single on-momentum particle (mean radiation)
p1 = make_particles(x_inj, px_inj, *np.zeros((6, 1)))
ring.track(p1, ele_stop='inj_kicker')
k_nominal = (p1.px[0] - tw.px[i_k])               # dpx = -knl[0]
x_at_kicker = p1.x[0] - tw.x[i_k]

Brho = ring.particle_ref.p0c[0] / 299792458.0
print(f"\nInjection: x_inj = {side*x_inj*1e3:.2f} mm, px_inj = {side*px_inj*1e3:.3f} mrad, "
      f"dmu(septum->kicker) = {dmu/2/np.pi:.3f} [2pi]")
print(f"  centroid at kicker: x = {x_at_kicker*1e3:.3f} mm (should be ~0)")
print(f"  kicker: knl0 = {k_nominal*1e3:.3f} mrad, int B dl = {abs(k_nominal)*Brho*1e3:.1f} mT m "
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
lost_on_septum = side * (particles.x - co0['x']) < INJ['x_blade'] + BLADE_THICKNESS
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

t_start = time.time()
ring.track(particles, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
ebe = ring.record_last_track
ebe = {a: getattr(ebe, a).copy() for a in ['x', 'px', 'y', 'py', 'delta', 'state', 's']}
pdr['k_inj'] = 0.0                                   # kicker off from turn 1 on
print(f'Turn 0 done in {time.time()-t_start:.1f}s; tracking {N_TURNS} turns '
      f'(estimated {(time.time()-t_start)*N_TURNS/60:.0f} min)...')

turn = 1
while turn < N_TURNS and np.any(particles.state > 0):
    nt = min(CHUNK, N_TURNS - turn)
    # explicit monitor: the default one always starts recording at turn 0
    m = xt.ParticlesMonitor(start_at_turn=turn, stop_at_turn=turn + nt, num_particles=n0)
    ring.track(particles, num_turns=nt, turn_by_turn_monitor=m)
    mom.append(moments(m.state, m.x, m.px, m.y, m.py, m.zeta, m.delta))
    for ts in SNAP_TURNS:
        if turn <= ts < turn + nt:
            take_snap(ts, m, ts - turn)
    turn += nt
    print(f'  turn {turn}/{N_TURNS}: {np.sum(particles.state > 0)} alive', end='\r')
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
    n_sigma=N_SIGMA, x_inj_m=float(side * x_inj), px_inj_rad=float(side * px_inj), dmu_septum_kicker=float(dmu / 2 / np.pi),
    k_inj_rad=float(k_nominal), kick_error=KICK_ERROR, kicker_intBdl_Tm=float(abs(k_nominal) * Brho),
    mismatch_x=float(Bmag_x), mismatch_y=float(Bmag_y), matched_twiss=MATCH_TWISS,
    eps_x_inj=float(ex_inj), eps_y_inj=float(ey_inj), eq_gemitt_x=float(tw.eq_gemitt_x),
    damping_turns=[float(t) for t in tau_turns], pipe_radius=R_PIPE, **{k: v for k, v in INJ.items()},
    use_ecs=USE_ECS, ecs=ECS)
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
e = ring['inj_septum_entry']
ap_seg.append((_s['inj_septum_entry'], _s['inj_septum_exit'], e.max_x, e.min_x, e.max_y, e.min_y))
ap_seg = np.array(ap_seg)

def sigma_eq(plane):
    if plane == 'x':
        return np.sqrt(tw.eq_gemitt_x * tw.betx + (tw.dx * sig_delta_eq)**2)
    return np.sqrt(max(tw.eq_gemitt_y, 1e-13) * tw.bety)

# ---- 1 & 2: first-turn envelope (injection region zoom and full ring) ----
alive0 = ebe['state'] > 0
with np.errstate(invalid='ignore'):
    import warnings; warnings.simplefilter('ignore', RuntimeWarning)
    cx = np.nanmean(np.where(alive0, ebe['x'], np.nan), 0)
    sx = np.nanstd(np.where(alive0, ebe['x'], np.nan), 0)
    cy = np.nanmean(np.where(alive0, ebe['y'], np.nan), 0)
    sy = np.nanstd(np.where(alive0, ebe['y'], np.nan), 0)
s_ebe = ebe['s'][np.argmax(alive0.sum(1))]           # s of a surviving particle at each element

for tag, xlim in [('injection_region', (s_sep[0] - 1, s_kick + 6)), ('full_ring', (0, C))]:
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    s_st = s_tw_wrapped[order_tw] if tag == 'injection_region' else s_tw
    idx = order_tw if tag == 'injection_region' else slice(None)
    for ax_, c, s_, lab, a, sig_e in [(axs[0], cx, sx, 'x', A_X, sigma_eq('x')),
                                      (axs[1], cy, sy, 'y', A_Y, sigma_eq('y'))]:
        ax_.fill_between(s_ebe, 1e3 * (c - N_SIGMA * s_), 1e3 * (c + N_SIGMA * s_), color='C0', alpha=0.3,
                         label=f'injected bunch, $\\pm{N_SIGMA:g}\\sigma$ (turn 0)')
        ax_.plot(s_ebe, 1e3 * c, 'C0', lw=1.5, label='injected centroid')
        ax_.fill_between(s_st, 1e3 * (tw[lab][idx] - N_SIGMA * sig_e[idx]), 1e3 * (tw[lab][idx] + N_SIGMA * sig_e[idx]),
                         color='C1', alpha=0.5, label=f'stored beam at equilibrium, $\\pm{N_SIGMA:g}\\sigma$')
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
    axs[0].plot(s_sep, [side * 1e3 * INJ['x_blade']] * 2, color='C2', lw=4, label='septum blade')
    axs[0].plot(s_sep, [side * 1e3 * (INJ['x_blade'] + BLADE_THICKNESS)] * 2, color='C2', lw=4)
    axs[0].legend(fontsize=8, loc='lower right', ncol=2)
    axs[1].set_xlabel('s from injection point [m]'); axs[1].set_xlim(*xlim)
    axs[0].set_title(f'First-turn beam envelope (D{design} C{config} {phase}deg{"_" + changes if changes else ""}, {mode})')
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
