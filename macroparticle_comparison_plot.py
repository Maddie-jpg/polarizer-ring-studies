#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

# Fill in the real files. Chronological order: First -> First (Compressed)
# -> Second -> Current. "Current" always means the newest file -- when a
# newer one arrives, rename what's currently "Current" down to the next
# ordinal ("Second" -> whatever comes after) and point CURRENT_FILE at the
# new one, same pattern used to add this fourth distribution.
FIRST_FILE = 'PositronBeam_2p86GeV/Beam_3GHzOption_2.86GeV_20260421.dat'
SECOND_FILE = 'PositronBeam_2p86GeV/beam_ECS_04092026.dat'
CURRENT_FILE = '/home/mwatson/Documents/laughing-octo-bassoon/PositronBeam_2p86GeV/beam_ECS_09102026.dat'

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
    as the other, uncompressed, mm-based datasets."""
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


first_df = pd.read_csv(FIRST_FILE, sep=r'\s+')
first_compressed_df = apply_energy_compressor(first_df)
# Second and Current are already ECS output (both filenames carry "_ECS_"),
# so neither gets apply_energy_compressor() -- same reasoning as before,
# just now applying to two files instead of one.
second_df = pd.read_csv(SECOND_FILE, sep=r'\s+')
current_df = pd.read_csv(CURRENT_FILE, sep=r'\s+')

p_std_before = first_df['p[MeV/c]'].std()
p_std_after = first_compressed_df['p[MeV/c]'].std()
t_std_before = first_df['t[mm/c]'].std()
t_std_after = first_compressed_df['t[mm/c]'].std()
print(f"First Distribution momentum spread:  before={p_std_before:.3f} MeV/c, "
      f"after={p_std_after:.3f} MeV/c  ({100*(p_std_after/p_std_before-1):+.1f}%)")
print(f"First Distribution t-spread:         before={t_std_before:.3f} mm, "
      f"after={t_std_after:.3f} mm  ({100*(t_std_after/t_std_before-1):+.1f}%)")

datasets = [
    ('First Distribution', first_df, 'tab:orange'),
    ('First Distribution (Compressed)', first_compressed_df, 'tab:green'),
    ('Second Distribution', second_df, 'tab:blue'),
    ('Current Distribution', current_df, 'tab:red'),
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
    """Matches macroparticles.py's actual reported emittance exactly: runs
    the SAME coupled filter_by_action_xy() cut (x filtered, then y
    filtered on the x-filtered result, n_sigma=3, max_iter=5) that
    macroparticles.py applies once and reuses for BOTH planes, then reads
    off whichever plane's Twiss was asked for from that shared result.
    Previously this filtered pos_col/ang_col on its own, un-coupled from
    the other plane's cut -- close to macroparticles.py's numbers but not
    identical to them, the same issue found in the ellipse-vs-sigma plot."""
    df_clean = filter_by_action_xy(df, n_sigma=3, max_iter=5)
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
        # Coupled x-then-y cut (filter_by_action_xy), same shared
        # df_clean macroparticles.py uses for both planes -- not an
        # independent single-plane filter_by_action() call.
        df_clean = filter_by_action_xy(df_i, n_sigma=3, max_iter=5)
        pos = df_clean[pos_col].values
        ang = df_clean[ang_col].values
        alpha, beta, gamma, _, _ = get_twiss(df_clean, pos_col, ang_col)
        eps = CalcEmittanceAuto(df_clean, pos_col, ang_col)
        corr = np.corrcoef(pos, ang)[0, 1]
        print(f"    {label:32s} beta={beta:8.3f}  alpha={alpha:7.3f}  "
              f"eps={eps:7.3f}  std_pos={pos.std():7.3f}  std_ang={ang.std():7.3f}  "
              f"corr={corr:+.3f}  N={len(df_clean)}/{len(df_i)}")

print("\nBetatron mismatch between all distributions (pairwise):")
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
    t[mm/c]/p[MeV/c] -- for pos_col/ang_col = x/xp or y/yp, 'First
    Distribution' and 'First Distribution (Compressed)' will sit exactly
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


def sigma_cut_ellipses_xy(df):
    """For each sigma_values entry, returns both planes' ellipses, built
    from ONE shared, sequentially-filtered population per n_sigma --
    matching macroparticles.py's actual pipeline exactly:

        df_clean = filter_by_action_xy(df, n_sigma=N_SIGMA)   # x filtered,
                                                                 # then y
                                                                 # filtered
                                                                 # on the
                                                                 # x-filtered
                                                                 # result
        ax, bx, ... = get_twiss(df_clean, 'x[mm]', 'xp[mrad]')  # same df_clean
        ay, by, ... = get_twiss(df_clean, 'y[mm]', 'yp[mrad]')  # same df_clean

    i.e. the horizontal AND vertical Twiss both come from the one combined
    (x-then-y) cut, not from independently filtering each plane. The
    previous version of this function filtered x and y separately
    (filter_by_action per plane, un-coupled), which is why its numbers at
    n_sigma=3 were close to but not identical to macroparticles.py's
    reported 3-sigma emittance (e.g. N=21925/ε=1.261 μm here vs
    N=21921/ε=1.26 μm there) -- the vertical cut was never being applied
    before measuring the horizontal emittance, and vice versa.

    filter_by_action_xy is called with max_iter=5, matching
    macroparticles.py's filter_by_action_xy(df, n_sigma=N_SIGMA) call,
    whose max_iter defaults to 5 (not filter_by_action's own default of 8).

    Returns {'x': [(label, x_ell, xp_ell, color), ...],
             'y': [(label, y_ell, yp_ell, color), ...]}.
    """
    results = {'x': [], 'y': []}
    for n_sigma, c in zip(sigma_values, sigma_colors):
        if n_sigma is None:
            df_i = remove_dispersion(df, 'x[mm]', 'xp[mrad]')
            df_i = remove_dispersion(df_i, 'y[mm]', 'yp[mrad]')
            tag = f'No cut (N={len(df_i)})'
        else:
            df_i = filter_by_action_xy(df, n_sigma=n_sigma, max_iter=5)
            tag = f'{n_sigma}σ (N={len(df_i)})'

        for plane, pos_col, ang_col in [('x', 'x[mm]', 'xp[mrad]'), ('y', 'y[mm]', 'yp[mrad]')]:
            alpha, beta, gamma, _, _ = get_twiss(df_i, pos_col, ang_col)
            eps = CalcEmittanceAuto(df_i, pos_col, ang_col)
            # amp = n_sigma*sqrt(eps), matching draw_action_ellipse() in
            # macroparticles.py exactly. This is the actual cut BOUNDARY --
            # the ellipse where J = gamma*x^2+2*alpha*x*xp+beta*xp^2 equals
            # n_sigma^2*eps, i.e. the edge filter_by_action's mask uses.
            # Previously this used amp=sqrt(eps) (no n_sigma factor at
            # all), which draws the 1-sigma RMS ellipse of the filtered
            # population instead of the n_sigma boundary that produced it
            # -- correct emittance, wrong ellipse, which is why these
            # ellipses looked roughly 3x too small at the "3sigma" label
            # compared to macroparticles.py's plot. n_sigma=None has no
            # boundary to scale to (nothing was cut), so it stays at
            # amp=sqrt(eps): the population's own unscaled RMS ellipse.
            amp = (n_sigma if n_sigma is not None else 1.0) * np.sqrt(eps)
            x_ell = amp * np.sqrt(beta) * np.cos(theta)
            xp_ell = -amp / np.sqrt(beta) * (alpha * np.cos(theta) - np.sin(theta))
            label = f'{tag}, ε={eps:.3f} μm'
            results[plane].append((label, x_ell, xp_ell, c))
    return results


# Twiss-vs-sigma analysis always runs on current_df -- i.e. whichever file
# is CURRENT_FILE above, so this automatically tracks the newest
# distribution without any further changes here when a newer file arrives.
# Computed ONCE per plot (not per-axis) so both planes' ellipses come from
# the same shared x-then-y filtered population at each sigma, per plane.
ellipses_by_plane = sigma_cut_ellipses_xy(current_df)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, pos_col, ang_col, plane, plane_label in [
    (axes[0], 'x[mm]', 'xp[mrad]', 'x', 'Horizontal'),
    (axes[1], 'y[mm]', 'yp[mrad]', 'y', 'Vertical'),
]:
    pos_full = current_df[pos_col].values
    ang_full = current_df[ang_col].values

    for label, x_ell, xp_ell, c in ellipses_by_plane[plane]:
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

for ax, pos_col, ang_col, plane, plane_label in [
    (axes[0], 'x[mm]', 'xp[mrad]', 'x', 'Horizontal'),
    (axes[1], 'y[mm]', 'yp[mrad]', 'y', 'Vertical'),
]:
    pos_full = current_df[pos_col].values
    ang_full = current_df[ang_col].values

    ax.scatter(pos_full, ang_full, s=2, alpha=0.15, color='gray', zorder=1,
               label=f'Particles (N={len(pos_full)})')

    for label, x_ell, xp_ell, c in ellipses_by_plane[plane]:
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


def getrms_gaussfit(xis, maxnmax=50, max_doublings=20):
    """Verbatim from macroparticles.py. Robust average/variance via
    histogram + Gaussian fit -- doubles nbins until the tallest bin drops
    below maxnmax counts, then fits a Gaussian to the histogram. The fit
    is what makes this robust to a handful of far-out particles, since
    they barely move where a fitted peak sits, unlike a raw np.var()
    which they can dominate."""
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
    """Verbatim from macroparticles.py. Iterative moment/projection-based
    Twiss reconstruction: normalize the (dispersion-subtracted)
    coordinates using the current beta/alpha guess, take three
    projections at 0/120/240 degrees in that normalized frame, get each
    projection's variance via getrms_gaussfit(), and reconstruct an
    updated beta/alpha/emittance from the three variances. Has no sigma
    cut at all -- there's nothing to tighten or loosen, it converges by
    iterating, which is why this comparison plot sweeps iteration count
    instead of n_sigma. Returns (history, Dt, Dpt) where history is a
    list of (iteration, eps, beta, alfa) tuples, one per iteration."""
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


# Moment-based Twiss has no n_sigma cut, so it's swept over iteration
# count instead -- same comparison macroparticles.py makes in
# moment_vs_action_emittance.png (that script's version runs on its one
# injected-beam df; here it runs on current_df, the same dataset the
# sigma-cut plots above use, so the two comparison plots line up).
moment_iters = 5
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, pos_col, ang_col, plane, plane_label in [
    (axes[0], 'x[mm]', 'xp[mrad]', 'x', 'Horizontal'),
    (axes[1], 'y[mm]', 'yp[mrad]', 'y', 'Vertical'),
]:
    history, Dt, Dpt = moment_based_twiss(current_df, pos_col, ang_col, iters=moment_iters)

    colors_it = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for (it, eps, beta, alfa), c in zip(history, colors_it):
        x_ell = np.sqrt(eps * beta) * np.cos(theta)
        xp_ell = -np.sqrt(eps / beta) * (alfa * np.cos(theta) - np.sin(theta))
        ax.plot(x_ell, xp_ell, color=c, label=f'Moment-based iter {it} (ε={eps:.3f} μm)')

    # Naive reference: the plain covariance-based RMS ellipse of the FULL,
    # UNFILTERED population (dispersion removed, but no action cut at
    # all) -- not the 3-sigma action-based estimate used elsewhere in
    # this script. This is what moment_based_twiss is actually meant to
    # be compared against: CalcEmittanceAuto on the raw distribution is
    # exactly the quantity that the long skewed tail (traced earlier in
    # this project) drags upward, while the moment-based iterations
    # should settle toward a stable value despite that same tail being
    # present in their input too (moment_based_twiss also runs on
    # current_df, unfiltered -- its robustness comes from the Gaussian-fit
    # variance estimator, not from discarding any particles).
    df_dispfree = remove_dispersion(current_df, pos_col, ang_col)
    alpha_naive, beta_naive, gamma_naive, _, _ = get_twiss(df_dispfree, pos_col, ang_col)
    eps_naive = CalcEmittanceAuto(df_dispfree, pos_col, ang_col)
    x_naive = np.sqrt(eps_naive * beta_naive) * np.cos(theta)
    xp_naive = -np.sqrt(eps_naive / beta_naive) * (alpha_naive * np.cos(theta) - np.sin(theta))
    ax.plot(x_naive, xp_naive, color='red', lw=2.5, linestyle='--',
            label=f'Naive full-distribution covariance (ε={eps_naive:.3f} μm)')

    ax.set_title(f'{plane_label}: Moment-based vs Naive Covariance, Full Distribution (Current)')
    ax.set_xlabel(pos_col)
    ax.set_ylabel(ang_col)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvline(0, color='black', lw=0.5, ls='--')
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize='small')

plt.tight_layout()
plt.savefig('current_moment_vs_action_emittance.png', dpi=150, bbox_inches='tight')
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