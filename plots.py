# %%
# ============================================================
# D1/C9 lattice survey + triplet drift-space / strength survey
# ============================================================
import sys
import os

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import xtrack as xt
import xobjects as xo
import numpy as np
import matplotlib.pyplot as plt

import my_functions as mf

xo.context_cpu.allow_no_prebuilt_kernel = True

# %%
design = int(os.environ.get('DESIGN', 1))
config = int(os.environ.get('CONFIG', 9))
mode = os.environ.get('MODE', 'perfect')
phase = int(os.environ.get('PHASE', 90))
changes = os.environ.get('CHANGES', None)

DRIFT_PAD = 1.0               # metres of padding around the triplet in the zoomed plot

# %%
if changes is not None:
    pdr = xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_{mode}_{phase}_{changes}.json")
else:
    pdr = xt.Environment.from_json(f"JSON Files/D{design}/C{config}/pdr_{mode}_{phase}.json")

ring = pdr.lines['ring']

folder = mf.results_dir(design, config, phase, changes=changes, metric='LatticeOptics', sub=mode)

# %%
# ------------------------------------------------------------
# Plot 1: full-ring survey
# ------------------------------------------------------------
if mode in ['perfect', 'misaligned']:
    ring.survey().plot()
else:
    mf.survey_plot(ring)   # adds BPM/kicker overlay -- only meaningful once correctors exist

plt.title(f'Ring survey -- D{design}C{config}, {mode}')
plt.tight_layout()
plt.savefig(f'{folder}/ring_survey_{mode}.png', dpi=200)
plt.show()
print(f"Saved: {folder}/ring_survey_{mode}.png")

# %%
# ------------------------------------------------------------
# Plot 2: the insertion-region window between Bend2_2R8 and
# Bend2_3L8 (dispersion suppressor -> doublet -> triplet ->
# doublet -> dispersion suppressor), with every quad's K1L and
# every drift's length labeled.
# ------------------------------------------------------------
WINDOW_START_NAME = 'Bend2_2R8'
WINDOW_END_NAME = 'Bend2_3L8'

element_names = ring.element_names
i0 = element_names.index(WINDOW_START_NAME)
i1 = element_names.index(WINDOW_END_NAME)
window_names = element_names[i0:i1 + 1]
print(f"Window: {WINDOW_START_NAME} (idx {i0}) -> {WINDOW_END_NAME} (idx {i1}), "
      f"{len(window_names)} elements:")
print(window_names)

tab_df = ring.get_table(attr=True).to_pandas().reset_index(drop=True)
tab_window = tab_df.iloc[i0:i1 + 1].reset_index(drop=True)

s0_window = tab_window['s'].iloc[0]
s1_window = tab_window['s'].iloc[-1] + tab_window['length'].iloc[-1]
print(f"s-range: [{s0_window:.4f}, {s1_window:.4f}]  (length {s1_window - s0_window:.4f} m)")

# %%
fig, axl = plt.subplots(figsize=(20, 4))
axl.set_ylim(-1.8, 1.8)
axl.set_xlim(s0_window, s1_window)
axl.axhline(0, color='black', linewidth=1, zorder=1)

scK1 = 0.5   # visual height scaling for the quad blocks -- adjust to taste

for ind in range(len(tab_window) - 1):
    row = tab_window.iloc[ind]
    etype = row['element_type']
    name = row['name']
    s0, s1 = row['s'], tab_window.iloc[ind + 1]['s']
    length = s1 - s0

    if 'Quad' in etype:
        kstr = scK1 * row['k1l'] / max(row['length'], 1e-9)
        axl.add_patch(plt.Rectangle((s0, min(0.0, kstr)), length, abs(kstr),
                                     fill=True, color='tab:orange', zorder=3))
        y_label = (0.9 if kstr >= 0 else -0.9)
        axl.annotate(
            f"{name}\n" + r"$K_1L$" + f" = {row['k1l']:+.4f} " + r"m$^{-1}$",
            xy=(s0 + length / 2, y_label), ha='center', va='center',
            fontsize=8, color='saddlebrown',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='tab:orange'),
            zorder=5,
        )

    elif 'Bend' in etype:
        axl.add_patch(plt.Rectangle((s0, -0.08), length, 0.16,
                                     fill=True, color='tab:blue', zorder=3))
        axl.annotate(name, xy=(s0 + length / 2, -0.25), ha='center', va='top',
                     fontsize=7, color='tab:blue', rotation=90, zorder=5)

    elif 'Sext' in etype:
        axl.add_patch(plt.Rectangle((s0, -0.05), length, 0.10,
                                     fill=True, color='tab:green', zorder=3))

    elif 'Drift' in etype and length > 1e-9:
        axl.annotate(
            f"{length:.3f} m",
            xy=(s0 + length / 2, 0), xytext=(0, -22), textcoords='offset points',
            ha='center', va='top', fontsize=7, color='darkgreen', rotation=90,
            zorder=5,
        )
        print(f"Drift {name}: {length:.4f} m")

axl.set_xlabel('Position s [m]')
axl.set_yticks([])
axl.set_title(f'Insertion region ({WINDOW_START_NAME} -> {WINDOW_END_NAME}) -- D{design}C{config}, {mode}')

plt.tight_layout()
plt.savefig(f'{folder}/insertion_region_{mode}.png', dpi=200)
plt.show()
print(f"Saved: {folder}/insertion_region_{mode}.png")