#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Fill in the two real files.
CURRENT_FILE = 'PositronBeam_2p86GeV_PolarizedEbeam/beam_ECS_04092026.dat'
PREVIOUS_FILE_UNCOMPRESSED = 'PositronBeam_2p86GeV/Beam_3GHzOption_2.86GeV_20260421.dat'

# Same Energy Compressor System parameters as macroparticles.py.
R56 = 0.352963
Vdeb = -0.048365
Phasdeb = 0.29229
C_LIGHT = 299792458
FREQ = 3e9
K = 2 * np.pi * FREQ / C_LIGHT


def apply_energy_compressor(df):
    """Same R56/Vdeb/Phasdeb transform as the ENERGY_COMPRESSOR_ON branch
    in macroparticles.py. That transform works in meters internally
    (z_i = t[mm/c]*1e-3) and stores the result back into a column still
    named 't[mm/c]' -- so it ends up in METERS despite the name. Converted
    back to mm here (*1e3) so this dataset plots on the same t-axis scale
    as the other two, uncompressed, mm-based datasets."""
    df_out = df.copy()
    Eref = np.mean(df_out['p[MeV/c]'].values)
    delta_i = df_out['p[MeV/c]'].values / Eref - 1
    z_i = df_out['t[mm/c]'].values * 1e-3
    z_f = z_i + R56 * delta_i
    delta_f = delta_i + Vdeb * np.sin(K * z_f + Phasdeb)
    p_f = Eref * (1 + delta_f)
    df_out['t[mm/c]'] = z_f * 1e3
    df_out['p[MeV/c]'] = p_f
    return df_out


current_df = pd.read_csv(CURRENT_FILE, sep=r'\s+')
previous_df = pd.read_csv(PREVIOUS_FILE_UNCOMPRESSED, sep=r'\s+')
previous_compressed_df = apply_energy_compressor(previous_df)

p_std_before = previous_df['p[MeV/c]'].std()
p_std_after = previous_compressed_df['p[MeV/c]'].std()
t_std_before = previous_df['t[mm/c]'].std()
t_std_after = previous_compressed_df['t[mm/c]'].std()
print(f"Previous file momentum spread:  before={p_std_before:.3f} MeV/c, "
      f"after={p_std_after:.3f} MeV/c  ({100*(p_std_after/p_std_before-1):+.1f}%)")
print(f"Previous file t-spread:         before={t_std_before:.3f} mm, "
      f"after={t_std_after:.3f} mm  ({100*(t_std_after/t_std_before-1):+.1f}%)")

datasets = [
    ('Current', current_df, 'tab:blue'),
    ('Previous (uncompressed)', previous_df, 'tab:orange'),
    ('Previous (compressed)', previous_compressed_df, 'tab:green'),
]


def CalcEmittanceAuto(df, position, angle):
    """Verbatim from macroparticles.py."""
    pos = df[position].values
    ang = df[angle].values
    cov = np.cov(pos, ang, ddof=0)
    emittance = np.sqrt(np.linalg.det(cov))
    return emittance


def get_twiss(df, position, angle, p_col='p[MeV/c]', p0_mev=2860.0):
    """Verbatim from macroparticles.py."""
    pos = df[position].values
    ang = df[angle].values
    eps = CalcEmittanceAuto(df, position, angle)
    cov_matrix = np.cov(pos, ang, ddof=0)
    beta = cov_matrix[0, 0] / eps
    alpha = -cov_matrix[0, 1] / eps
    gamma = (1 + alpha**2) / beta
    delta = (df[p_col].values - p0_mev) / p0_mev
    dispersion_x = np.cov(pos, delta, ddof=0)[0, 1] / np.var(delta)
    disp_prime_x = np.cov(ang, delta, ddof=0)[0, 1] / np.var(delta)
    return alpha, beta, gamma, dispersion_x, disp_prime_x


def remove_dispersion(df, position, angle, p_col='p[MeV/c]', p0_mev=2860.0):
    """Verbatim from macroparticles.py."""
    df_out = df.copy()
    delta = (df_out[p_col].values - p0_mev) / p0_mev
    disp = np.cov(df_out[position].values, delta, ddof=0)[0, 1] / np.var(delta)
    ddisp = np.cov(df_out[angle].values, delta, ddof=0)[0, 1] / np.var(delta)
    df_out[position] = df_out[position].values - disp * delta
    df_out[angle] = df_out[angle].values - ddisp * delta
    return df_out


def filter_by_action(df, position, angle, n_sigma=3, max_iter=8, min_particles=20):
    """Same as macroparticles.py, PLUS one safety guard not present there
    (macroparticles.py only ever calls this at n_sigma=3, where it
    converges fine -- this script also scans n_sigma=1, which does not):
    at very tight n_sigma the iteration can keep re-trimming every pass
    without ever satisfying the "nothing left to cut" convergence check,
    running all max_iter passes and collapsing toward zero particles
    (confirmed: n_sigma=1 hits N=0 on this dataset without this guard).
    If a cut would leave fewer than min_particles, stop and keep the
    last valid (non-degenerate) population instead of continuing into
    empty/singular covariance territory."""
    df_iter = df.copy()
    for _ in range(max_iter):
        df_iter = remove_dispersion(df_iter, position, angle)
        alpha, beta, gamma, _, _ = get_twiss(df_iter, position, angle)
        eps = CalcEmittanceAuto(df_iter, position, angle)

        x = df_iter[position].values - df_iter[position].values.mean()
        xp = df_iter[angle].values - df_iter[angle].values.mean()
        J = gamma * x**2 + 2 * alpha * x * xp + beta * xp**2

        mask = J <= n_sigma**2 * eps
        if mask.sum() == len(df_iter):
            break
        if mask.sum() < min_particles:
            break
        df_iter = df_iter[mask].copy()
    return df_iter


def filter_by_action_xy(df, n_sigma=3, max_iter=5):
    """Verbatim from macroparticles.py."""
    df_x = filter_by_action(df, 'x[mm]', 'xp[mrad]', n_sigma, max_iter)
    df_xy = filter_by_action(df_x, 'y[mm]', 'yp[mrad]', n_sigma, max_iter)
    return df_xy


def get_twiss_xy(df, pos_col, ang_col):
    """Now matches macroparticles.py's actual reported emittance: runs the
    same filter_by_action() (dispersion removal + iterative action cut,
    n_sigma=3) before computing beta/alpha/eps, instead of using the raw,
    unfiltered dataframe directly."""
    df_clean = filter_by_action(df, pos_col, ang_col, n_sigma=3)
    alpha, beta, gamma, _, _ = get_twiss(df_clean, pos_col, ang_col)
    eps = CalcEmittanceAuto(df_clean, pos_col, ang_col)
    return beta, alpha, eps


def betatron_mismatch(beta1, alpha1, beta2, alpha2):
    """Same formula as in macroparticles.py (Sagan & Rubin; MAD-X/Bmad's
    "Bmag"). H=1.0 when perfectly matched, >1 otherwise. Bmag is the
    emittance growth factor after filamentation: eps_effective = Bmag*eps_true."""
    gamma1 = (1 + alpha1**2) / beta1
    gamma2 = (1 + alpha2**2) / beta2
    H = 0.5 * (beta2*gamma1 - 2*alpha1*alpha2 + beta1*gamma2)
    Bmag = H + np.sqrt(max(H**2 - 1, 0.0))
    return H, Bmag


print("\nTwiss parameters per dataset (what's actually driving the mismatch):")
for plane, pos_col, ang_col in [('horizontal', 'x[mm]', 'xp[mrad]'),
                                 ('vertical', 'y[mm]', 'yp[mrad]')]:
    print(f"  {plane}:")
    for label, df_i, _ in datasets:
        df_clean = filter_by_action(df_i, pos_col, ang_col, n_sigma=3)
        pos = df_clean[pos_col].values
        ang = df_clean[ang_col].values
        alpha, beta, gamma, _, _ = get_twiss(df_clean, pos_col, ang_col)
        eps = CalcEmittanceAuto(df_clean, pos_col, ang_col)
        corr = np.corrcoef(pos, ang)[0, 1]
        print(f"    {label:26s} beta={beta:8.3f}  alpha={alpha:7.3f}  "
              f"eps={eps:7.3f}  std_pos={pos.std():7.3f}  std_ang={ang.std():7.3f}  "
              f"corr={corr:+.3f}  N={len(df_clean)}/{len(df_i)}")

print("\nBetatron mismatch between the three distributions:")
for plane, pos_col, ang_col in [('horizontal', 'x[mm]', 'xp[mrad]'),
                                 ('vertical', 'y[mm]', 'yp[mrad]')]:
    twiss = {label: get_twiss_xy(df_i, pos_col, ang_col) for label, df_i, _ in datasets}
    labels = list(twiss.keys())
    print(f"  {plane}:")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            beta1, alpha1, _ = twiss[labels[i]]
            beta2, alpha2, _ = twiss[labels[j]]
            H, Bmag = betatron_mismatch(beta1, alpha1, beta2, alpha2)
            print(f"    {labels[i]} vs {labels[j]}: H={H:.4f}, Bmag={Bmag:.4f}")


def plot_phase_space_comparison(pos_col, ang_col, xlabel, ylabel, title, out_file):
    """Same scatter + marginal-histogram layout, reused for longitudinal,
    horizontal, and vertical. Note: the energy compressor only touches
    t[mm/c]/p[MeV/c] -- for pos_col/ang_col = x/xp or y/yp,
    'Previous (uncompressed)' and 'Previous (compressed)' will sit exactly
    on top of each other, since apply_energy_compressor() never modifies
    the transverse columns. That's correct (a longitudinal-only
    compressor shouldn't touch x/xp/y/yp), not a plotting bug."""
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(
        4, 5,
        width_ratios=[1, 1, 1, 1, 0.3],
        height_ratios=[0.8, 1, 1, 1],
        hspace=0.08,
        wspace=0.15
    )

    ax_main = fig.add_subplot(gs[1:4, 0:4])
    ax_top = fig.add_subplot(gs[0, 0:4], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 4], sharey=ax_main)

    for label, df_i, color in datasets:
        ax_main.scatter(df_i[pos_col], df_i[ang_col], s=3, alpha=0.35,
                         color=color, label=f'{label} (N={len(df_i)})')
        ax_top.hist(df_i[pos_col], bins=80, histtype='step', color=color, linewidth=1.6)
        ax_right.hist(df_i[ang_col], bins=80, orientation='horizontal',
                      histtype='step', color=color, linewidth=1.6)

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.legend(fontsize='small', loc='best')
    ax_main.grid(True, linestyle=':', alpha=0.4)

    ax_top.set_ylabel('Counts')
    ax_top.tick_params(axis='x', labelbottom=False)

    ax_right.set_xlabel('Counts')
    ax_right.tick_params(axis='y', labelleft=False)

    fig.suptitle(title, fontsize=14)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.show()


plot_phase_space_comparison(
    't[mm/c]', 'p[MeV/c]', r'$t$ [mm/c]', r'$p$ [MeV/c]',
    'Longitudinal Phase Space Comparison', 'longitudinal_phase_space_comparison.png')


sigma_values = [1, 2, 3, 4, 5, None]  # None = no cut, full population
sigma_colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_values)))
theta = np.linspace(0, 2 * np.pi, 200)


def sigma_cut_ellipses(df, pos_col, ang_col):
    """Returns [(label, x_ellipse, xp_ellipse, color), ...] for each
    sigma_values entry, reused by both the clean ellipse-only plot and the
    scatter-background version below. Now uses the SAME filter_by_action()
    pipeline as macroparticles.py (dispersion removal + iterative action
    cut) for each sigma value, instead of a plain single-pass Mahalanobis
    cut with no dispersion removal -- so n_sigma=3 here reproduces exactly
    what macroparticles.py reports as its emittance. Label includes the
    geometric emittance (um), same "mm*mrad reported directly as um"
    convention already used throughout macroparticles.py.

    Note: for n_sigma=None (no cut), dispersion is still removed (a single
    remove_dispersion() pass, no filtering) so it's a fair "no outlier cut,
    but still dispersion-free" baseline -- comparing against the truly raw,
    dispersion-INCLUDED population would conflate two different effects.
    """
    results = []
    for n_sigma, c in zip(sigma_values, sigma_colors):
        if n_sigma is None:
            df_i = remove_dispersion(df, pos_col, ang_col)
            tag = f'No cut (N={len(df_i)})'
        else:
            df_i = filter_by_action(df, pos_col, ang_col, n_sigma=n_sigma)
            tag = f'{n_sigma}\u03c3 (N={len(df_i)})'

        alpha, beta, gamma, _, _ = get_twiss(df_i, pos_col, ang_col)
        eps = CalcEmittanceAuto(df_i, pos_col, ang_col)

        x_ell = np.sqrt(eps * beta) * np.cos(theta)
        xp_ell = -np.sqrt(eps / beta) * (alpha * np.cos(theta) - np.sin(theta))
        label = f'{tag}, \u03b5={eps:.3f} \u03bcm'
        results.append((label, x_ell, xp_ell, c))
    return results


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, pos_col, ang_col, plane_label in [
    (axes[0], 'x[mm]', 'xp[mrad]', 'Horizontal'),
    (axes[1], 'y[mm]', 'yp[mrad]', 'Vertical'),
]:
    pos_full = current_df[pos_col].values
    ang_full = current_df[ang_col].values

    for label, x_ell, xp_ell, c in sigma_cut_ellipses(current_df, pos_col, ang_col):
        ax.plot(x_ell, xp_ell, color=c, label=label)

    ax.set_title(f'{plane_label} Twiss Ellipse vs. Sigma Cut (Current)')
    ax.set_xlabel(pos_col)
    ax.set_ylabel(ang_col)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvline(0, color='black', lw=0.5, ls='--')
    #ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize='small')

plt.tight_layout()
plt.savefig('current_twiss_vs_sigma.png', dpi=150, bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, pos_col, ang_col, plane_label in [
    (axes[0], 'x[mm]', 'xp[mrad]', 'Horizontal'),
    (axes[1], 'y[mm]', 'yp[mrad]', 'Vertical'),
]:
    pos_full = current_df[pos_col].values
    ang_full = current_df[ang_col].values

    ax.scatter(pos_full, ang_full, s=2, alpha=0.15, color='gray', zorder=1,
               label=f'Particles (N={len(pos_full)})')

    for label, x_ell, xp_ell, c in sigma_cut_ellipses(current_df, pos_col, ang_col):
        ax.plot(x_ell, xp_ell, color=c, label=label, linewidth=2, zorder=2)

    # Zoom to the 0.5-99.5 percentile range with margin, rather than the
    # full data range -- a handful of far-out particles (real beam files
    # routinely have these, as established earlier) would otherwise force
    # the axes so wide the ellipses collapse to an unreadable sliver.
    pos_lo, pos_hi = np.percentile(pos_full, [0.5, 99.5])
    ang_lo, ang_hi = np.percentile(ang_full, [0.5, 99.5])
    pos_margin = 0.3 * (pos_hi - pos_lo)
    ang_margin = 0.3 * (ang_hi - ang_lo)
    ax.set_xlim(pos_lo - pos_margin, pos_hi + pos_margin)
    ax.set_ylim(ang_lo - ang_margin, ang_hi + ang_margin)

    ax.set_title(f'{plane_label} Twiss Ellipse vs. Sigma Cut (Current, with particles)')
    ax.set_xlabel(pos_col)
    ax.set_ylabel(ang_col)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvline(0, color='black', lw=0.5, ls='--')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(fontsize='small')

plt.tight_layout()
plt.savefig('current_twiss_vs_sigma_with_particles.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(9, 6))
for label, df_i, color in datasets:
    plt.hist(df_i['p[MeV/c]'], bins=100, histtype='step', color=color,
              linewidth=1.6, label=f'{label} (N={len(df_i)})')
plt.xlabel(r'$p$ [MeV/c]')
plt.ylabel('Counts')
plt.title('Momentum Distribution Comparison')
plt.grid(True, linestyle=':', alpha=0.4)
plt.legend()
plt.savefig('momentum_histogram_comparison.png', dpi=150, bbox_inches='tight')
plt.show()