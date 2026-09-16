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
 
 
def get_twiss_xy(df, pos_col, ang_col):
    """Beta/alpha from raw position/angle columns -- same formula as
    get_twiss() in macroparticles.py, just without the dispersion part
    (not needed here, only beta/alpha feed betatron_mismatch)."""
    pos = df[pos_col].values
    ang = df[ang_col].values
    cov = np.cov(pos, ang, ddof=0)
    eps = np.sqrt(np.linalg.det(cov))
    beta = cov[0, 0] / eps
    alpha = -cov[0, 1] / eps
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
        pos = df_i[pos_col].values
        ang = df_i[ang_col].values
        beta, alpha, eps = get_twiss_xy(df_i, pos_col, ang_col)
        corr = np.corrcoef(pos, ang)[0, 1]
        print(f"    {label:26s} beta={beta:8.3f}  alpha={alpha:7.3f}  "
              f"eps={eps:7.3f}  std_pos={pos.std():7.3f}  std_ang={ang.std():7.3f}  "
              f"corr={corr:+.3f}")
 
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
 
