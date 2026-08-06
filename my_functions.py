import xtrack as xt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from TuneDiagram.lib.TuneDiagram.tune_diagram import resonance_lines
from xutil_DA_CC.xsuite_plot_functions import DA_vs_turns
import xpart as xp
from scipy.optimize import curve_fit
import csv

pdr=xt.Environment()

#-----------------------------------
# FUNCTIONS FOR analysis.py SCRIPT
#-----------------------------------

def addSketchBL(acc_tw, acc, lims, limy1, limy2, scK1):
    # Added a 4th row for Chromatic Functions (axw)
    fig, ax = plt.subplots(4, 1, figsize=(12, 12), 
                           height_ratios=(1, 3, 2, 2))
    axl, axp, axw, axt = ax  # Lattice, Optics, W-functions, Text Table
    
    # Link the x-axes
    axp.sharex(axl)
    axw.sharex(axl)
    fig.subplots_adjust(hspace=0.1, top=0.95, bottom=0.05, left=.1)

    #Subplot 2: Beta and Dispersion
    axp.set_ylabel(r'$\beta_x, \beta_y$ [m]')
    indm = np.argmin(np.abs(acc_tw.s - lims[-1])) + 1
    
    axp.plot(acc_tw.s[:indm], acc_tw.betx[:indm], color='red', label=r'$\beta_x$')
    axp.plot(acc_tw.s[:indm], acc_tw.bety[:indm], color='blue', label=r'$\beta_y$')
    axp.set_ylim(limy1)
    
    axp2 = axp.twinx()
    axp2.plot(acc_tw.s[:indm], acc_tw.dx[:indm], color='black', linestyle='--', label='$D_x$')
    axp2.set_ylabel(r'$D_x$ [m]')
    axp2.set_ylim(limy2)
    axp.set_xlim(lims)

    # Chromatic W Functions and second order dispersion
    axw.plot(acc_tw.s[:indm], acc_tw.wx_chrom[:indm], label='$W_x$', color='red')
    axw.plot(acc_tw.s[:indm], acc_tw.wy_chrom[:indm], label='$W_y$', color='blue')
    axw2 = axw.twinx()
    axw2.plot(acc_tw.s[:indm], acc_tw.ddx[:indm], color='black', linestyle='--', label="$D'_x$")
    

    '''tt_sliced = acc.get_table(attr=True)
    tbends = tt_sliced.rows[tt_sliced.element_type == 'Bend']
    
    # Apply shading ONLY to the Chromatic W-functions axis
    for nn in tbends.name:
        axw.axvspan(
            tbends['s', nn], 
            tbends['s', nn] + tbends['length', nn],
            color='tab:blue', 
            alpha=0.15,       # Subtle enough to see the Wx/Wy lines
            linewidth=0
        )'''
    
    axw.set_ylabel('Chromatic $W$')
    axw.set_xlabel('Position s [m]')
    axw.legend(loc='upper right', fontsize=8)
    axw2.legend(loc='upper right', fontsize=8)
    axw2.set_ylabel(r"$D'_x$ [m]")
    
    #Lattice Sketch
    axl.set_ylim(-0.5, 1.5)
    axl.axis('off')
    tab_pan = acc.get_table(attr=True).to_pandas()
    for ind in range( len(tab_pan.T.columns) -1 ):
        if tab_pan['element_type'][ind].find('Drift') >= 0:
           axl.plot( [tab_pan['s'][ind], tab_pan['s'][ind+1]], [0, 0], color='black' )
        if tab_pan['element_type'][ind].find('Bend') >= 0:
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], -0.08), 
               tab_pan['s'][ind+1] - tab_pan['s'][ind], 0.16, fill=True, color='tab:blue' ) ) 
        if tab_pan['element_type'][ind].find('Quad') >= 0:
           kstr = scK1*tab_pan['k1l'][ind]/tab_pan['length'][ind]
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], min(0.0, kstr)),
               tab_pan['s'][ind+1] - tab_pan['s'][ind], abs(kstr), fill=True, color='tab:orange' ) )   
        if tab_pan['element_type'][ind].find('Sext') >= 0:
           max_k2l = max([abs(val) for ind, val in enumerate(tab_pan['k2l']) 
               if 'Sext' in tab_pan['element_type'][ind]])
           kstr = scK1*tab_pan['k2l'][ind]/tab_pan['length'][ind]/max_k2l *3
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], min(0.0, kstr)) ,
               tab_pan['s'][ind+1] - tab_pan['s'][ind], abs(kstr), fill=True, color='tab:green' ) )   
        if tab_pan['element_type'][ind].find('Multipole') >= 0:
           axl.add_patch( patches.Rectangle( (tab_pan['s'][ind], -0.08), 
               tab_pan['s'][ind+1] - tab_pan['s'][ind], 0.16, fill=True, color='tab:pink' ) )   
    axp.plot( [acc.get_length(), acc.get_length()], limy1, color='black', 
              linestyle=(0, (8, 8)), linewidth=.5 )

    # Text Area
    axt.axis('off')
    axt.set_xlim(-0.3, 4.0)
    axt.set_ylim(0, 1.5)
    
    return axt

# Routine generating graphical and text output describing the ring
def SpuckParsAus(line, acc_tw, acc, lims, limy1, limy2, scK1, pdr, grname='NoGraph'):

    element_names = line.element_names
    sectors = set()
    for name in element_names:
        # Looking for structural segment identifiers e.g. '_1R', '_2R', '_3L'
        if "_" in name:
            parts = name.split("_")[-1]
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ["R", "L"]:
                sectors.add(parts[0])

    N_c = len(sectors) if len(sectors) > 0 else 1
    # Fallback default if your specific layout operates on a hardcoded sector configuration:
    if N_c == 1:
        N_c = 2

    axt = addSketchBL(acc_tw, acc, lims, limy1, limy2, scK1)
    axt.text(0.0, .7, f'C ={N_c*acc_tw.circumference:8.4f} m', horizontalalignment='left')
    axt.text(1.0, .7, f'T ={N_c*1e6*acc_tw.T_rev0:8.4f} us', horizontalalignment='left')
    axt.text(2.0, .7, r'($Q_x$, $Q_y$)' + f' = ({N_c*acc_tw.qx:9.5f}, {N_c*acc_tw.qy:9.5f})',
             horizontalalignment='left')

    pos = 0
    for item in pdr.vars.keys():
        if item.startswith('__'):        # internal entries like __vary_default__
            continue
        val = pdr[item]
        if not isinstance(val, (int, float, np.floating, np.integer)):
            continue                     # skip dicts/expressions/anything non-scalar
        if np.abs(val) > 1e-12:
            print("'" + item + f"': {val:8.4f},")
            axt.text(pos % 4, .6 - .1 * int(pos / 4),
                     "'" + item + f"': {val:8.4f},",
                     horizontalalignment='left')
            pos += 1

    file = os.getcwd()
    if grname != 'NoGraph':
        out_path = os.path.join(file, grname)
        if os.path.exists(out_path):
            print(' Error: file ' + grname + ' exists <<<<<<<<<<<================')
        else:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)

    tab_pan = acc.get_table(attr=True).to_pandas()
    print('  Name        Type      L(m)     sin(m)   sout(m)  driftl  driftr    k1(1/m^2)')
    for ind in range(len(tab_pan.T.columns) - 1):
        eltyp = tab_pan['element_type'][ind]
        if eltyp.find('Drift') < 0:
            elnam = tab_pan['name'][ind]
            if ind < 1:
                sin = '  0.0000'
            else:
                sin = f"{tab_pan['length'][ind-1]:8.4f}"
            k1expr = pdr.element_refs[elnam].k1._expr
            if k1expr == None:
                k1expr = ' None'
            else:
                k1expr = str(k1expr)[5:-1]
            print("  " + elnam.ljust(12) + eltyp.ljust(12)[:8] + f" {tab_pan['length'][ind]:7.4f} " +
                  f"{tab_pan['s'][ind]:8.4f} {tab_pan['s'][ind+1]:8.4f} " + sin +
                  f"{tab_pan['length'][ind+1]:8.4f} " + k1expr.ljust(9) +
                  f"= {tab_pan['k1l'][ind]/max(tab_pan['length'][ind],1e-6):7.4f}  ")

    return axt


def survey_plot(ring):
    fig, ax = plt.subplots(figsize=(12, 12))
    sv = ring.survey()
    sv.plot(ax=ax) 
    df = sv.to_pandas()

    bpm_offset = 3.0  # Meters to push BPMs out
    k_height = 2.0    # Transverse thickness of the kicker box 

    for name in df['name']:
        # Get the survey row for this element
        row = df[df['name'] == name].iloc[0]
        
        # Logic for Kickers (Mx and My)
        if name.startswith(('Mx', 'My')):
        
            length = ring.element_dict[name].length
            if length == 0: length = 0.5 # Minimum visible length if it's a marker
            
            color = 'hotpink' if name.startswith('Mx') else 'cyan'
            label = 'H-Kicker' if name.startswith('Mx') else 'V-Kicker'
            
        
            angle_deg = np.degrees(row['theta'])
            
        
            rect = patches.Rectangle(
                (row['Z'] - length/2, row['X'] - k_height/2), 
                length, k_height, 
                angle=angle_deg, rotation_point='center',
                color=color, zorder=10, label=label
            )
            ax.add_patch(rect)

        elif name.startswith(('BPMx', 'BPMy')):

            direction = row['theta'] + np.pi/2
            
            off_z = row['Z'] + bpm_offset * np.cos(direction)
            off_x = row['X'] + bpm_offset * np.sin(direction)
            
            if name.startswith('BPMx'):
                bpm_color = 'hotpink'
                bpm_label = 'BPM (Horizontal)'
            elif name.startswith('BPMy'):
                bpm_color = 'cyan'
                bpm_label = 'BPM (Vertical)'
            else:
                bpm_color = 'black'
                bpm_label = 'BPM (Other)'

            ax.scatter(off_z, off_x, color=bpm_color, s=20, 
                       zorder=11, label=bpm_label)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

    ax.set_aspect('equal')



def plot_resonance_grid_red_blue(ax, qx_range, qy_range, orders, periodicity,
                                 alpha=0.6, lw_systematic=2.2, lw_nonsystematic=1.1,
                                 label_legend=True):
    """
    Classic systematic/non-systematic resonance grid: red = systematic
    (res_sum % periodicity == 0), blue = non-systematic, dashed = skew
    (odd ny). Same logic as TuneDiagram.resonance_lines.plot_resonance,
    reimplemented here so it (a) draws onto a specific `ax` instead of
    global pyplot state, and (b) exposes alpha/linewidth -- the library
    version hardcodes alpha=0.3 with no way to make it more visible.
    """
    res = resonance_lines(qx_range, qy_range, orders, periodicity)
    qmin_x, qmax_x = min(qx_range), max(qx_range)
    qmin_y, qmax_y = min(qy_range), max(qy_range)

    seen_systematic = seen_nonsystematic = False
    for nx_val, ny_val, res_sums in res.resonance_list:
        for res_sum in res_sums:
            if ny_val:
                x_pts = [qmin_x, qmax_x]
                y_pts = [(res_sum - nx_val * qmin_x) / ny_val,
                        (res_sum - nx_val * qmax_x) / ny_val]
            else:
                x_pts = [float(res_sum) / nx_val, float(res_sum) / nx_val]
                y_pts = [qmin_y, qmax_y]

            is_systematic = (res_sum % periodicity == 0)
            color = 'r' if is_systematic else 'b'
            lw = lw_systematic if is_systematic else lw_nonsystematic
            linestyle = '--' if (ny_val % 2) else '-'

            ax.plot(x_pts, y_pts, color=color, linewidth=lw,
                   linestyle=linestyle, alpha=alpha, zorder=1)
            if is_systematic:
                seen_systematic = True
            else:
                seen_nonsystematic = True

    if label_legend:
        extra = []
        if seen_systematic:
            extra.append(Line2D([0], [0], color='r', linewidth=lw_systematic,
                                label='Systematic resonance'))
        if seen_nonsystematic:
            extra.append(Line2D([0], [0], color='b', linewidth=lw_nonsystematic,
                                label='Non-systematic resonance'))
        existing_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=existing_handles + extra, loc='best',
                 fontsize=9, framealpha=0.9)

    return ax


def analyse_verdier_resonances_from_line(line, max_order=5, tune_tolerance=0.02):
    

    # twiss from line
    try:
        twiss = line.twiss()
        qx = twiss.qx
        qy = twiss.qy
        print(f"Current Operating Tunes -> Qx: {qx:.4f}, Qy: {qy:.4f}")
    except Exception as e:
        print(
            f"Could not compute line.twiss(). Please make sure the line is built and closed. Error: {e}"
        )
        return None

    # superperiodicity (N_c) from element names

    element_names = line.element_names
    sectors = set()
    for name in element_names:
        # Looking for structural segment identifiers e.g. '_1R', '_2R', '_3L'
        if "_" in name:
            parts = name.split("_")[-1]
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ["R", "L"]:
                sectors.add(parts[0])

    N_c = len(sectors) if len(sectors) > 0 else 1
    # Fallback default if your specific layout operates on a hardcoded sector configuration:
    if N_c == 1:
        N_c = 3  

    print(f"Identified Structural Superperiodicity (N_c): {N_c}")

    # Phase advances per structural superperiod
    mu_x_fraction = (qx / N_c) % 1.0
    mu_y_fraction = (qy / N_c) % 1.0

    #Check resonance driving terms up to max_order
    dangerous_resonances = []

    for order in range(1, max_order + 1):
        for m in range(-order, order + 1):
            remaining = order - abs(m)
            for n in [remaining, -remaining] if remaining != 0 else [0]:
                if m == 0 and n == 0:
                    continue

                # m * (mu_x / 2pi) + n * (mu_y / 2pi) is close to an integer
                phase_sum = m * mu_x_fraction + n * mu_y_fraction
                distance_to_structural = abs(phase_sum - round(phase_sum))

                #calculate distance to total physical global resonance line:
                global_phase_sum = m * qx + n * qy
                distance_to_global = abs(
                    global_phase_sum - round(global_phase_sum)
                )

                # Flag if the working tune point is dangerously close to a driven structural harmonic
                if distance_to_global < tune_tolerance:
                    # Check if this global resonance is structurally driven or systematically suppressed
                    # If it's a multiple of N_c, it is structurally driven
                    is_structural = (round(global_phase_sum) % N_c) == 0

                    status = (
                        "Dangerous (Structural)"
                        if is_structural
                        else "Suppressed (Non-Structural)"
                    )

                    dangerous_resonances.append(
                        {
                            "Order": order,
                            "m": m,
                            "n": n,
                            "Global Harmonic (k)": int(
                                round(global_phase_sum)
                            ),
                            "Distance to Resonance": f"{distance_to_global:.4f}",
                            "Verdier Status": status,
                            "Resonance Condition": f"{m}*Qx + {n}*Qy = {int(round(global_phase_sum))}",
                        }
                    )

    # Convert to Dataframe for visualization
    df = pd.DataFrame(dangerous_resonances)
    if not df.empty:
        df = df.drop_duplicates(subset=["m", "n"]).reset_index(drop=True)
        # Sort so that the closest dangerous resonances bubble up to the top
        df = df.sort_values(by="Distance to Resonance").reset_index(drop=True)

    return df




def plot_dangerous_resonances(line, qx, qy, max_order=(1, 2, 3, 4, 5),
                              tune_range=0.1, ax=None, qx_range=None,
                              qy_range=None, show_legend=True,
                              legend_tiers=(1, 2, 3, 4), tier_colors=None,
                              draw_background_grid=True):
    """
    Danger levels (Tier 1 to 4):
    1. Structural / Systematic resonances very close to the Working Point.
    2. Non-Structural / Error-driven resonances close to the Working Point.
    3. Resonances that are within the tune box but have lower immediate threat.
    4. Safe background lattice grid (drawn faintly in gray).

    Pass an existing `ax` (and optionally `qx_range`/`qy_range`, e.g. from a
    footprint plot) to overlay the tiered resonance lines on that axis instead
    of creating a new standalone figure. `tune_range` is only used to build
    the default range when `qx_range`/`qy_range` are not supplied.

    `legend_tiers` controls which tier entries get added to the legend --
    e.g. pass (1, 2) when overlaying on a busy combined plot to keep Tier
    3/4 drawn (for context) but out of the legend, since they rarely change
    the actionable read of the plot and just add clutter.

    `tier_colors` optionally overrides the default tier 1/2/3 colors, e.g.
    {1: 'black', 2: 'dimgray'} -- useful when overlaying on a plot that
    already uses red for something else (e.g. a systematic-resonance grid),
    so "close to WP" doesn't collide with a different red-based convention.

    `draw_background_grid` set False skips this function's own faint gray
    background line for every resonance -- turn off when overlaying on a
    plot that already draws its own background/context grid, to avoid a
    redundant, doubled-up faint layer.
    """
    if isinstance(max_order, (int, float)):
        orders_to_check = tuple(range(1, int(max_order) + 1))
    else:
        orders_to_check = tuple(sorted(max_order))

    default_tier_colors = {1: "red", 2: "darkorange", 3: "gold"}
    if tier_colors:
        default_tier_colors.update(tier_colors)
    tier_colors = default_tier_colors

    element_names = line.element_names
    sectors = set()
    for name in element_names:
        if "_" in name:
            parts = name.split("_")[-1]
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ["R", "L"]:
                sectors.add(parts[0])

    N_c = len(sectors) if len(sectors) > 0 else 3  
    print(f"Detected Superperiodicity N_c = {N_c}")
    print(f"Analyzing explicit resonance orders: {orders_to_check}")

    if qx_range is None:
        qx_range = (qx - tune_range, qx + tune_range)
    if qy_range is None:
        qy_range = (qy - tune_range, qy + tune_range)
    qmin_x, qmax_x = qx_range
    qmin_y, qmax_y = qy_range

    made_own_fig = ax is None
    if made_own_fig:
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
    else:
        fig = ax.figure
    
    diagram_object = resonance_lines(qx_range, qy_range, orders_to_check, N_c)

    if made_own_fig:
        ax.set_xlim(qmin_x, qmax_x)
        ax.set_ylim(qmin_y, qmax_y)

    for resonance in diagram_object.resonance_list:
        nx_val = resonance[0]
        ny_val = resonance[1]
        for res_sum in resonance[2]:
            # Plot the standard full faint template background layout first
            if ny_val != 0:
                x_pts = np.array([qmin_x, qmax_x])
                y_pts = (res_sum - nx_val * x_pts) / ny_val
            else:
                x_pts = np.array([float(res_sum) / nx_val, float(res_sum) / nx_val])
                y_pts = np.array([qmin_y, qmax_y])
                
            if draw_background_grid:
                ax.plot(x_pts, y_pts, color="lightgray", linestyle=":", linewidth=1.0, zorder=1, alpha=0.6)

            # Calculate perpendicular distance from working point (qx, qy) to line: nx*x + ny*y = res_sum
            distance = abs(nx_val * qx + ny_val * qy - res_sum) / np.sqrt(nx_val**2 + ny_val**2)
            
            is_structural = (res_sum % N_c == 0)

            if distance <= 0.015:
                if is_structural:
                    
                    color = tier_colors[1]
                    linewidth = 2.5
                    zorder = 4
                else:
                  
                    color = tier_colors[2]
                    linewidth = 2.0
                    zorder = 3
            elif distance <= 0.035:
                color = tier_colors[3]
                linewidth = 1.5
                zorder = 2
            else:
                continue  

            # Plot the colored highlighted overlay on top
            ax.plot(x_pts, y_pts, color=color, linewidth=linewidth, linestyle="-", zorder=zorder)

    # 7. Mark the nominal Active Working Point (skip when overlaying -- the
    #    host plot, e.g. the footprint one, already marks the nominal WP)
    if made_own_fig:
        ax.plot(qx, qy, marker="o", color="black", markersize=10, linestyle="", zorder=5)

    # 8. Axis labeling and Threat Assessment Legend
    ax.set_xlabel(r"$Q_x$", fontsize=12)
    ax.set_ylabel(r"$Q_y$", fontsize=12)
    if made_own_fig:
        ax.set_title(f"Dangerous resonances", fontsize=13)

    tier_legend_specs = {
        1: dict(color=tier_colors[1], linewidth=2.5, label="Tier 1: Most Dangerous (Structural near WP)"),
        2: dict(color=tier_colors[2], linewidth=2.0, label="Tier 2: Medium Danger (Non-Structural near WP)"),
        3: dict(color=tier_colors[3], linewidth=1.5, label="Tier 3: Low Danger (Wider tune tolerance)"),
        4: dict(color="lightgray", linestyle=":", linewidth=1, label="Tier 4: Safe / Faint Background Grid"),
    }
    resonance_legend_elements = [
        Line2D([0], [0], **tier_legend_specs[t]) for t in legend_tiers if t in tier_legend_specs
    ]
    if made_own_fig:
        resonance_legend_elements.append(
            Line2D([0], [0], marker="o", color="black", linestyle="",
                   markersize=10, label=f"Current Operating WP: {qx:.2f},{qy:.2f}"))

    if show_legend:
        if made_own_fig:
            ax.legend(handles=resonance_legend_elements)
        else:
            # merge with whatever handles the host axis already has
            existing_handles, existing_labels = ax.get_legend_handles_labels()
            ax.legend(handles=existing_handles + resonance_legend_elements,
                      loc='best')

    if made_own_fig:
        return fig, ax
    return ax

#---------------------------------------
# FUNCTIONS FOR spin_tracking.py SCRIPT
#---------------------------------------

import LatticeBuild.misalignments_corrections as mc
import xobjects as xo

def spin_tune_resonance_scan(ring, nu_min=5.0, nu_max=6.0, n_points=60,
                              misalign_sigma=None, seed=None):
    
    ring0 = ring
    a_gyro = ring0.particle_ref.anomalous_magnetic_moment[0]
    mass0 = ring0.particle_ref.mass0  # eV

    nu_targets = np.linspace(nu_min, nu_max, n_points)
    gammas = nu_targets / a_gyro
    energies = gammas * mass0             # total energy, eV

    results = {'nu_target': [], 'nu_spin': [], 'qx': [], 'qy': [], 'qs': [],
               'p_bks': [], 'energy': []}

    for nu_target, energy in zip(nu_targets, energies):
        line = ring0.copy()
        line.particle_ref.kinetic_energy0 = energy - mass0
        line.configure_spin('auto')

        if misalign_sigma is not None:
            line.configure_radiation('mean')
            line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
            line = mc.misalignments(line, misalign_sigma, seed=seed)

        try:
            tw = line.twiss(method='6d', radiation_integrals=True,
                             eneloss_and_damping=True, spin=True, polarization=True)
            nu_spin = tw.spin_tune_fractional + np.floor(nu_target)
            p_bks = tw.spin_polarization_inf_no_depol * 100
            qx, qy, qs = tw.qx % 1, tw.qy % 1, abs(tw.qs) % 1
           
        except Exception as e:
            print(f"  nu_target={nu_target:.4f}: twiss/spin failed ({type(e).__name__}: {e})")
            nu_spin = p_bks = qx = qy = qs = np.nan

        results['nu_target'].append(nu_target)
        results['nu_spin'].append(nu_spin)
        results['qx'].append(qx)
        results['qy'].append(qy)
        results['qs'].append(qs)
        results['p_bks'].append(p_bks)
        results['energy'].append(energy)

    return {k: np.array(v) for k, v in results.items()}


def plot_spin_resonance_scan(results, out_path=None):
    nu = results['nu_target']
    p_bks = results['p_bks']
    qx = np.nanmedian(results['qx'])
    qy = np.nanmedian(results['qy'])
    qs = np.nanmedian(results['qs'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nu, p_bks, 'o-', color='tab:blue', markersize=4, label='$P_{BKS}$ (DK limit)')

    n_lo, n_hi = int(np.floor(nu.min())), int(np.ceil(nu.max()))
    seen = set()

    # Imperfection resonances: ν = n (bare integer)
    for n in range(n_lo, n_hi + 1):
        if nu.min() <= n <= nu.max():
            ax.axvline(n, linestyle='-', color='black', alpha=0.5, linewidth=1.5,
                       label=f'Imperfection: $\\nu={n}$')

    # Intrinsic resonances: ν = n ± Q
    for n in range(n_lo, n_hi + 1):
        for Q, label, color in [(qx, 'Q_x', 'tab:red'),
                                 (qy, 'Q_y', 'tab:green'),
                                 (qs, 'Q_s', 'tab:purple')]:
            for sign, tag in [(+1, '+'), (-1, '-')]:
                res = n + sign * Q
                if nu.min() <= res <= nu.max():
                    key = (label, round(res, 4))
                    if key not in seen:
                        seen.add(key)
                        ax.axvline(res, linestyle='--', color=color, alpha=0.6,
                                   label=f'${n}{tag}{label}$ = {res:.3f}')

    ax.set_xlabel(r'Spin tune target $\nu = a\gamma$')
    ax.set_ylabel(r'$P_{BKS}$ (%)')
    ax.set_title('First-order spin resonance scan')
    ax.grid(True, linestyle=':', alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize=8, loc='best')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()

def deep_track_single(base_line, seed_val,long_scan_turns, apply_correction, transient_turns=8000):
    

    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Running deep track for Seed {seed_val} ({branch_label})...")

    line = base_line.copy()

    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.25e-3, seed=seed_val)

    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    if apply_correction:
        orbit_x_rms_before = np.std(tw.x)
        orbit_y_rms_before = np.std(tw.y)
        try:
            mc.orbit_correction(line, tw, threading=False)
        except Exception as e:
            print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
                  f"{type(e).__name__}: {e} -- retrying with threading=True")
            mc.orbit_correction(line, tw, threading=True)
        # Re-twiss after correction so tw reflects the corrected lattice.
        tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                        spin=True, polarization=True)
        orbit_x_rms_after = np.std(tw.x)
        orbit_y_rms_after = np.std(tw.y)
        print(f"  [seed {seed_val}] orbit RMS x: {orbit_x_rms_before:.3e} -> {orbit_x_rms_after:.3e}, "
              f"y: {orbit_y_rms_before:.3e} -> {orbit_y_rms_after:.3e}")
        
    num_particles=300

    particles = xp.generate_matched_gaussian_bunch(
        line=line,
        nemitt_x=tw.eq_nemitt_x,
        nemitt_y=tw.eq_nemitt_y,
        sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
        num_particles=num_particles)

    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]
    particles.spin_x = tw.spin_x[0]
    particles.spin_y = tw.spin_y[0]
    particles.spin_z = tw.spin_z[0]

    '''rng = np.random.default_rng(seed_val)
    fct=0
 
    epsx  = tw.eq_gemitt_x
    epsy  = tw.eq_gemitt_y
    epsl  = tw.eq_gemitt_zeta
 
    betxin, alfxin = tw.betx[0], tw.alfx[0]
    betyin, alfyin = tw.bety[0], tw.alfy[0]
    dxin, dpxin    = tw.dx[0], tw.dpx[0]
    betlin         = tw.bets0
 
    xin, pxin      = tw.x[0], tw.px[0]
    yin, pyin      = tw.y[0], tw.py[0]
    zetain, deltain = tw.zeta[0], tw.delta[0]
 
    sPxin, sPyin, sPzin = tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]
    dsPddx = tw.spin_dn_ddelta_x[0]
    dsPddy = tw.spin_dn_ddelta_y[0]
    dsPddz = tw.spin_dn_ddelta_z[0]
 
    
 
    normat = np.array([
        [(epsx*betxin)**.5,          0,                       0, 0, 0, dxin*(epsl/betlin)**.5],
        [-alfxin*(epsx/betxin)**.5,  (epsx/betxin)**.5,        0, 0, 0, dpxin*(epsl/betlin)**.5],
        [0, 0,  (epsx*betyin)**.5,          0,                       0, 0],
        [0, 0,  -alfyin*(epsx/betyin)**.5,  (epsx/betyin)**.5,  0, 0],
        [0, 0, 0, 0, (epsl*betlin)**.5, 0],
        [0, 0, 0, 0, 0,                 (epsl/betlin)**.5],
    ])
 
    def make_macro_part():
        x, px, y, py, zeta, delta = normat @ rng.standard_normal(6)
        sPx = sPxin + fct*(6.5*px*sPzin + delta*dsPddx)
        sPy = sPyin + fct*(6.5*py*sPzin + delta*dsPddy)
        sPz = sPzin + fct*(-6.5*(px*sPxin + py*sPyin) + delta*dsPddz)
        return (x + xin, px + pxin, y + yin, py + pyin,
                zeta + zetain, delta + deltain, sPx, sPy, sPz)
 
    parts = np.array([make_macro_part() for _ in range(num_particles)]).T
 
    particles = xt.Particles(
        p0c=pdr.lines['ring'].particle_ref.p0c[0],
        mass0=pdr.lines['ring'].particle_ref.mass0,
        q0=pdr.lines['ring'].particle_ref.q0,
        anomalous_magnetic_moment=pdr.lines['ring'].particle_ref.anomalous_magnetic_moment[0],
        x=parts[0], px=parts[1],
        y=parts[2], py=parts[3],
        zeta=parts[4], delta=parts[5],
        spin_x=parts[6], spin_y=parts[7], spin_z=parts[8],
    )'''
    

    line.discard_tracker()
   
    line.configure_radiation('quantum')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    line.configure_spin('auto')

    line.track(particles, num_turns=long_scan_turns, turn_by_turn_monitor=True, with_progress=10)
    mon = line.record_last_track

    # --- Turn-by-turn polarization ---
    mask_alive = mon.state > 0
    n_alive = mask_alive.sum(axis=0)
    pol_x = mon.spin_x.sum(axis=0) / n_alive
    pol_y = mon.spin_y.sum(axis=0) / n_alive
    pol_z = mon.spin_z.sum(axis=0) / n_alive
    pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

    n_turns_rec = len(pol)
    turns = np.arange(n_turns_rec)

    # Two-parameter exponential: free amplitude 'a' and decay rate 'tauinv' (= 1/tau).
    # Letting the amplitude float means we no longer have to force the curve through
    # the first point (as pol[3:]/pol[3] did), so an initial transient does not bias
    # the whole fit.
    def exp_decay(t, a, tauinv):
        return a * np.exp(-tauinv * t)

    # Two-step fit: (1) a linear least-squares fit provides robust starting values
    # for the amplitude and slope, then (2) curve_fit refines the exponential.
    def two_step_exp_fit(t, p):
        t = np.asarray(t, dtype=float)
        p = np.asarray(p, dtype=float)
        # Step 1: straight line P(t) ~ intercept + slope * t  ->  initial guesses.
        A = np.vstack([t, np.ones_like(t)]).T
        slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]
        # a0 = value extrapolated to t=0; tauinv0 = -slope/a0 (positive for decay).
        p0 = [intercept, -slope / intercept]
        popt, pcov = curve_fit(exp_decay, t, p, p0=p0, maxfev=10000)
        return popt, pcov

    # Turn index from which to fit, to drop the initial transient before the spin
    # distribution settles. Tune via the transient_turns argument.
    icTrns = int(np.clip(transient_turns, 0, n_turns_rec - 2))

    # Stage A: fit through *all* turns.
    # Stage B: fit from turn icTrns to the end (transient removed) -> primary result.
    try:
        popt_all, _ = two_step_exp_fit(turns, pol)
        popt_cut, _ = two_step_exp_fit(turns[icTrns:], pol[icTrns:])
        amp_all, tauinv_all = popt_all
        amp_cut, tauinv_cut = popt_cut
        t_dep_turns_all = 1.0 / tauinv_all if tauinv_all > 0 else np.nan
        t_dep_turns_long = 1.0 / tauinv_cut if tauinv_cut > 0 else np.nan
        fit_curve = exp_decay(turns, amp_cut, tauinv_cut)      # primary fit, full range
        fit_curve_all = exp_decay(turns, amp_all, tauinv_all)  # all-turns fit, full range
        fit_ok = True
    except (RuntimeError, ValueError) as e:
        print(f"  Exponential fit failed for seed {seed_val}: {e}")
        amp_cut = np.nan
        t_dep_turns_all = np.nan
        t_dep_turns_long = np.nan
        fit_curve = np.full_like(pol, np.nan)
        fit_curve_all = np.full_like(pol, np.nan)
        fit_ok = False

    n_over_tau = n_turns_rec / t_dep_turns_long if fit_ok and t_dep_turns_long > 0 else np.nan
    print(f"  Seed {seed_val}: tau_fit = {t_dep_turns_all:.1f} turns, "
          f"tau_fit(cut@{icTrns}) = {t_dep_turns_long:.1f} turns, N/tau = {n_over_tau:.2f}")

    p_bks = tw.spin_polarization_inf_no_depol
    t_bks = tw.spin_t_pol_component_s
    t_pol = t_bks
    t_pol_turns = t_bks / tw.T_rev0

    t_dep_analytic_s = tw.spin_t_depol_component_s          # NEW — analytic, from dn/ddelta
    t_dep_turns_analytic = t_dep_analytic_s / tw.T_rev0      # NEW — same units as your tracked one

    p_eq_analytic = (p_bks / (1 + t_pol_turns / t_dep_turns_analytic)) * 100

    return {
        'seed': seed_val,
        'turns': turns,
        'pol': pol,
        'fit': fit_curve,
        'fit_all': fit_curve_all,
        'amp': amp_cut,
        'icTrns': icTrns,
        'p_eq_long': p_eq_analytic,
        'p_bks': p_bks * 100,
        't_pol': t_pol,
        't_dep_turns_long': t_dep_turns_long,
        't_dep_turns_all': t_dep_turns_all,
    }


def plot_seed_with_textbox(data, title_prefix, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"{title_prefix} - Seed {data['seed']}", fontsize=14, fontweight='bold')

    ax.plot(data['turns'], data['pol'], label='Tracking Data', color='blue', alpha=0.7)
    ax.plot(data['turns'], data['fit'], color='red', linestyle='--',
            label=f"Exp. fit")
  

    ax.set_ylabel('$P(t)$')
    ax.set_xlabel('Turns')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    info_text = (
        f"$P_{{BKS}}$ = {data['p_bks']:.2f}%   "
        f"$P_{{eq}}$ (long track) = {data['p_eq_long']:.2f}%\n"
        f"$\\tau_{{depol}}$ = {data['t_dep_turns_long']:.1f} turns   "
        
        f"$t_{{pol}}$ = {data['t_pol']:.4e} s\n"
    )
    fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray', alpha=0.9))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved to {out_path}")

def plot_invariant_spin_vector(base_line,seed_val, apply_correction,out_path):
    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Plotting invariant spin vector for Seed {seed_val} ({branch_label})...")

    line = base_line.copy()

    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.26e-3, seed=seed_val)

    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    if apply_correction:
        try:
            mc.orbit_correction(line, tw, threading=False)
        except Exception as e:
            print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
                  f"{type(e).__name__}: {e} -- retrying with threading=True")
            mc.orbit_correction(line, tw, threading=True)
        # Orbit correction changes the closed orbit/optics, so re-twiss to get
        # the n0 vector consistent with the corrected lattice.
        tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                        spin=True, polarization=True)

    s = tw.s
    sx = tw.spin_x
    sy = tw.spin_y
    sz = tw.spin_z

    n0_mag = np.sqrt(sx**2 + sy**2 + sz**2)
    print(f"  |n0| range: [{n0_mag.min():.6f}, {n0_mag.max():.6f}] (should be ~1.0)")
    print(f"  spin tune = {tw.spin_tune_fractional:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, sx * 100, label='$100(n_{0,x})$', color='tab:blue')
    ax.plot(s, sy, label='$n_{0,y}$', color='tab:green')
    ax.plot(s, (sz* 100)+0.2, label='$100(n_{0,z})+0.2$', color='tab:red')

    ax.set_xlabel('s (m)')
    ax.set_ylabel('Invariant spin vector $n_0$ component')
    branch_tag = 'Corrected' if apply_correction else 'Misaligned'
    ax.set_title(f'Invariant Spin Vector Along the Ring — Seed {seed_val} ({branch_tag})')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved to {out_path}")




#%%


def track_single_particle_nx1(base_line,seed_val, apply_correction, out_path):
    branch_label = 'corrected' if apply_correction else 'misaligned'
    print(f"Plotting spin vector (nx=1 init) for Seed {seed_val} ({branch_label})...")
    line = base_line.copy()

    tw_0 = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)

    
    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.26e-3, seed=seed_val)
 
    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                    spin=True, polarization=True)
 
    if apply_correction:
        try:
            mc.orbit_correction(line, tw, threading=False)
        except Exception as e:
            print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
                  f"{type(e).__name__}: {e} -- retrying with threading=True")
            mc.orbit_correction(line, tw, threading=True)
 
    tw_nx1 = line.twiss(start=tw_0.name[0], end=tw_0.name[-2], init_at=tw_0.name[0],
                        x=tw_0.x[0],       px=tw_0.px[0],
                        y=tw_0.y[0],       py=tw_0.py[0],
                        zeta=tw_0.zeta[0], delta=tw_0.delta[0],
                        alfx=tw_0.alfx[0], betx=tw_0.betx[0],
                        alfy=tw_0.alfy[0], bety=tw_0.bety[0],
                        dx=tw_0.dx[0],     dpx=tw_0.dpx[0],
                        spin=True, spin_x=1, spin_y=0, spin_z=0,
                        _continue_if_lost=True)



 
    s  = tw_nx1.s
    sx = tw_nx1.spin_x
    sy = tw_nx1.spin_y
    sz = tw_nx1.spin_z
 
    n0_mag = np.sqrt(sx**2 + sy**2 + sz**2)
    print(f"  |s| range: [{n0_mag.min():.6f}, {n0_mag.max():.6f}]")
 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, sx , label='$s_x$',        color='tab:blue')
    ax.plot(s, sy,       label='$s_y$',              color='tab:green')
    ax.plot(s, sz , label='$s_z$', color='tab:red')
 
    ax.set_xlabel('s (m)')
    ax.set_ylabel('Spin vector component')
    branch_tag = 'Corrected' if apply_correction else 'Misaligned'
    ax.set_title(f'Spin Vector Along the Ring $(s_x(0)=1)$ — Seed {seed_val} ({branch_tag})')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved to {out_path}")



# %%


def n0_vs_spin_tune_scan(base_line,seed_val, nu_min, nu_max, n_points=60,apply_correction=True, at_element=None):

    line = base_line.copy()
    
    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, 0.26e-3, seed=seed_val)
    
    tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                        spin=True, polarization=True)
    
    if apply_correction:
            try:
                mc.orbit_correction(line, tw, threading=False)
            except Exception as e:
                print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
                      f"{type(e).__name__}: {e} -- retrying with threading=True")
                mc.orbit_correction(line, tw, threading=True)
            # Orbit correction changes the closed orbit/optics, so re-twiss to get
            # the n0 vector consistent with the corrected lattice.
            tw = line.twiss(method='6d', radiation_integrals=True, eneloss_and_damping=True,
                            spin=True, polarization=True)
    ring0 = line
    a_gyro = ring0.particle_ref.anomalous_magnetic_moment[0]
    mass0 = ring0.particle_ref.mass0  # eV

    nu_targets = np.linspace(nu_min, nu_max, n_points)
    gammas = nu_targets / a_gyro
    energies = gammas * mass0

    nu_spin, n0x, n0y, n0z = [], [], [], []

    for nu_target, energy in zip(nu_targets, energies):
        line = ring0.copy()
        line.particle_ref.kinetic_energy0 = energy - mass0
        line.configure_spin('auto')
        try:
            tw = line.twiss(method='6d', radiation_integrals=True,
                            eneloss_and_damping=True, spin=True, polarization=True)
            idx = 0 if at_element is None else tw.rows.indices[at_element]
            nu_spin.append(tw.spin_tune_fractional + np.floor(nu_target))
            n0x.append(tw.spin_x[idx])
            n0y.append(tw.spin_y[idx])
            n0z.append(tw.spin_z[idx])
        except Exception as e:
            print(f"  nu_target={nu_target:.4f}: failed ({type(e).__name__}: {e})")
            nu_spin.append(np.nan); n0x.append(np.nan)
            n0y.append(np.nan);     n0z.append(np.nan)

    return {k: np.array(v) for k, v in
            dict(nu_spin=nu_spin, n0x=n0x, n0y=n0y, n0z=n0z).items()}


def plot_n0_vs_spin_tune(results, out_path=None):
    nu = results['nu_spin']
    order = np.argsort(nu)  # spin tune isn't monotonic in nu_target near a
                            # resonance crossing, so sort by the x-axis itself
    nu = nu[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(nu, results['n0x'][order], 'o-', ms=4, label='$n_{0,x}$', color='tab:blue')
    ax.plot(nu, results['n0y'][order], 'o-', ms=4, label='$n_{0,y}$', color='tab:green')
    ax.plot(nu, results['n0z'][order], 'o-', ms=4, label='$n_{0,z}$', color='tab:red')

    ax.set_xlabel(r'Spin tune $\nu_{spin}$')
    ax.set_ylabel(r'Invariant spin vector $n_0$ component')
    ax.set_title(r'Invariant spin vector vs spin tune')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    return fig, ax



def check_qy_spin_coupling(line, dqy=1e-3, qy_knobs=('kQFarc', 'kQDarc'),
                           max_order=5, verbose=True):


    tw0 = line.twiss(method='6d', radiation_integrals=True,
                     eneloss_and_damping=True, spin=True, polarization=True)
    Qx0, Qy0 = tw0.qx, tw0.qy
    nu0 = tw0.spin_tune_fractional
    n0_vec0 = np.array([tw0.spin_x[0], tw0.spin_y[0], tw0.spin_z[0]])

    n_center = int(round(nu0))
    candidates = []
    for n in range(n_center - max_order, n_center + max_order + 1):
        for sign in (+1, -1):
            res = (n + sign * Qy0)
            candidates.append((abs(res - nu0), n, sign, res))
    candidates.sort(key=lambda c: c[0])
    dist, n_near, sign_near, res_near = candidates[0]

    if verbose:
        sign_str = '+' if sign_near > 0 else '-'
        print(f"[coupling] Qx={Qx0:.5f}  Qy={Qy0:.5f}  nu_spin={nu0:.5f}")
        print(f"[coupling] nearest vertical intrinsic resonance: "
              f"nu = {n_near} {sign_str} Qy = {res_near:.5f}  "
              f"(distance {dist:.5f})")

    line2 = line.copy()
    k0 = {kn: line2.vars[kn]._value for kn in qy_knobs}
    limits = {kn: (v - 0.05 * abs(v) - 1e-6, v + 0.05 * abs(v) + 1e-6)
             for kn, v in k0.items()}
    try:
        opt = line2.match(
            method='6d', solve=False, verbose=False,
            vary=[xt.Vary(kn, step=1e-6, limits=limits[kn]) for kn in qy_knobs],
            targets=[
                xt.Target('qx', Qx0, tol=1e-7),
                xt.Target('qy', Qy0 + dqy, tol=1e-7),
            ],
        )
        try:
            opt.solve(n_steps=20)
        except (RuntimeError, np.linalg.LinAlgError):
            for kn in qy_knobs:
                line2.vars[kn] = k0[kn]
            opt.step(20, broyden=True, rcond=1e-3)
        tw1 = line2.twiss(method='6d', radiation_integrals=True,
                          eneloss_and_damping=True, spin=True, polarization=True)
        Qx1, Qy1 = tw1.qx, tw1.qy
        nu1 = tw1.spin_tune_fractional
        n0_vec1 = np.array([tw1.spin_x[0], tw1.spin_y[0], tw1.spin_z[0]])

        dQy_actual = Qy1 - Qy0
        dQx_actual = Qx1 - Qx0
        dnu = nu1 - nu0
        dn0 = np.linalg.norm(n0_vec1 - n0_vec0)

        dnu_dQy = dnu / dQy_actual if abs(dQy_actual) > 1e-12 else np.nan
        dn0_dQy = dn0 / abs(dQy_actual) if abs(dQy_actual) > 1e-12 else np.nan

        if verbose:
            print(f"[coupling] Qy nudged by {dQy_actual:+.2e} "
                  f"(Qx held to within {dQx_actual:+.2e})")
            print(f"[coupling] d(nu_spin)/d(Qy) ~= {dnu_dQy:.4f}")
            print(f"[coupling] |d(n0)|/d(Qy)    ~= {dn0_dQy:.4f}  "
                  f"(tilt of invariant spin axis per unit Qy)")
    except Exception as e:
        print(f"[coupling] could not isolate Qy via {qy_knobs} "
              f"({type(e).__name__}: {e}); reporting resonance proximity only")
        dnu_dQy = dn0_dQy = np.nan

    return dict(Qx=Qx0, Qy=Qy0, nu_spin=nu0,
                nearest_resonance=res_near, nearest_order=n_near,
                nearest_sign=sign_near, distance_to_resonance=dist,
                dnu_dQy=dnu_dQy, dn0_dQy=dn0_dQy)


def compare_qy_spin_coupling_across_branches(base_line, seed_val,
                                             misalign_sigma=0.25e-3,
                                             dqy=1e-3, qy_knobs=('kQFarc', 'kQDarc'),
                                             max_order=5, results_dir=None):
    rows = {}

    line_perfect = base_line.copy()
    print("=== Coupling check: PERFECT lattice ===")
    rows['perfect'] = check_qy_spin_coupling(
        line_perfect, dqy=dqy, qy_knobs=qy_knobs, max_order=max_order)

    line_mis = base_line.copy()
    line_mis.configure_radiation('mean')
    line_mis.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line_mis = mc.misalignments(line_mis, misalign_sigma, seed=seed_val)
    print(f"\n=== Coupling check: MISALIGNED lattice (seed {seed_val}) ===")
    rows['misaligned'] = check_qy_spin_coupling(
        line_mis, dqy=dqy, qy_knobs=qy_knobs, max_order=max_order)

    line_cor = base_line.copy()
    line_cor.configure_radiation('mean')
    line_cor.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line_cor = mc.misalignments(line_cor, misalign_sigma, seed=seed_val)
    tw_cor = line_cor.twiss(method='6d', radiation_integrals=True,
                            eneloss_and_damping=True, spin=True, polarization=True)
    try:
        mc.orbit_correction(line_cor, tw_cor, threading=False)
    except Exception as e:
        print(f"  [seed {seed_val}] orbit_correction(threading=False) raised: "
              f"{type(e).__name__}: {e} -- retrying with threading=True")
        mc.orbit_correction(line_cor, tw_cor, threading=True)
    print(f"\n=== Coupling check: CORRECTED lattice (seed {seed_val}) ===")
    rows['corrected'] = check_qy_spin_coupling(
        line_cor, dqy=dqy, qy_knobs=qy_knobs, max_order=max_order)

    print("\n=== Summary: Qy / nu_spin across branches ===")
    header = f"{'branch':<12}{'Qy':>12}{'nu_spin':>12}{'dist_to_res':>14}{'dnu_dQy':>12}{'dn0_dQy':>12}"
    print(header)
    for branch, r in rows.items():
        print(f"{branch:<12}{r['Qy']:>12.6f}{r['nu_spin']:>12.6f}"
              f"{r['distance_to_resonance']:>14.6f}{r['dnu_dQy']:>12.4f}{r['dn0_dQy']:>12.4f}")

    dQy_mis = rows['misaligned']['Qy'] - rows['perfect']['Qy']
    dnu_mis = rows['misaligned']['nu_spin'] - rows['perfect']['nu_spin']
    dQy_cor = rows['corrected']['Qy'] - rows['perfect']['Qy']
    dnu_cor = rows['corrected']['nu_spin'] - rows['perfect']['nu_spin']
    print(f"\nMisalignment shift:  dQy = {dQy_mis:+.2e}   d(nu_spin) = {dnu_mis:+.2e}")
    print(f"Residual after correction: dQy = {dQy_cor:+.2e}   d(nu_spin) = {dnu_cor:+.2e}")

    if results_dir is not None:
        import csv
        out_path = f'{results_dir}/qy_spin_coupling_seed{seed_val}.csv'
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['branch', 'Qx', 'Qy', 'nu_spin', 'nearest_resonance',
                       'distance_to_resonance', 'dnu_dQy', 'dn0_dQy'])
            for branch, r in rows.items():
                w.writerow([branch, r['Qx'], r['Qy'], r['nu_spin'],
                           r['nearest_resonance'], r['distance_to_resonance'],
                           r['dnu_dQy'], r['dn0_dQy']])
        print(f"\nSaved summary to {out_path}")

    return rows



def assess_seed_resonance_excitation(seed_val, apply_correction, base_line,long_scan_turns,
                                     misalign_sigma=0.26e-3, p_eq_suppression_flag=0.9,
                                     results_dir=None):
    branch_label = 'corrected' if apply_correction else 'misaligned'

    line = base_line.copy()
    line.configure_radiation('mean')
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))
    line = mc.misalignments(line, misalign_sigma, seed=seed_val)

    if apply_correction:
        tw = line.twiss(method='6d', radiation_integrals=True,
                        eneloss_and_damping=True, spin=True, polarization=True)
        try:
            mc.orbit_correction(line, tw, threading=False)
        except Exception as e:
            print(f"orbit_correction(threading=False) raised: {type(e).__name__}: {e}"
                  f" -- retrying with threading=True")
            mc.orbit_correction(line, tw, threading=True)

    coupling = check_qy_spin_coupling(line)

    track_result = deep_track_single(base_line,seed_val,long_scan_turns, apply_correction)

    p_bks = track_result['p_bks']
    p_eq_long = track_result['p_eq_long']
    t_dep_turns = track_result['t_dep_turns_long']
    fit_reliable = track_result['fit_reliable']

    suppression_ratio = p_eq_long / p_bks if p_bks > 0 else np.nan
    excited = bool(np.isfinite(suppression_ratio) and
                  suppression_ratio < p_eq_suppression_flag)

    print(f"\n=== Resonance excitation assessment: seed {seed_val} ({branch_label}) ===")
    print(f"Qx={coupling['Qx']:.5f}  Qy={coupling['Qy']:.5f}  "
          f"nu_spin={coupling['nu_spin_full']:.5f}")
    print(f"nearest resonance: nu = {coupling['nearest_order']} "
          f"{'+' if coupling['nearest_sign']>0 else '-'} Qy = {coupling['nearest_resonance']:.5f}  "
          f"(distance {coupling['distance_to_resonance']:.5f})")
    print(f"tau_depol = {t_dep_turns:.1f} turns (fit_reliable={fit_reliable})")
    print(f"p_bks = {p_bks:.2f}%   p_eq_long = {p_eq_long:.2f}%   "
          f"suppression_ratio = {suppression_ratio:.3f}")
    print(f"excited (p_eq_long < {p_eq_suppression_flag} * p_bks): {excited}")

    row = dict(seed=seed_val, branch=branch_label, Qx=coupling['Qx'], Qy=coupling['Qy'],
              nu_spin_full=coupling['nu_spin_full'],
              nearest_order=coupling['nearest_order'],
              nearest_sign=coupling['nearest_sign'],
              nearest_resonance=coupling['nearest_resonance'],
              distance_to_resonance=coupling['distance_to_resonance'],
              t_dep_turns=t_dep_turns, fit_reliable=fit_reliable,
              p_bks=p_bks, p_eq_long=p_eq_long,
              suppression_ratio=suppression_ratio, excited=excited)

    if results_dir is not None:
        out_path = f'{results_dir}/resonance_excitation_seed{seed_val}_{branch_label}.csv'
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"Saved to {out_path}")

    return row


def assess_resonance_excitation_multi_seed(seed_list, apply_correction, deep_track_single_fn,
                                           check_qy_spin_coupling_fn, base_line,
                                           misalign_sigma=0.26e-3, p_eq_suppression_flag=0.9,
                                           results_dir=None, out_name='resonance_excitation_summary.csv'):
    rows = []
    for seed_val in seed_list:
        row = assess_seed_resonance_excitation(
            seed_val, apply_correction, deep_track_single_fn, check_qy_spin_coupling_fn,
            base_line, mc, xo, misalign_sigma=misalign_sigma,
            p_eq_suppression_flag=p_eq_suppression_flag, results_dir=None)
        rows.append(row)

    if results_dir is not None:
        out_path = f'{results_dir}/{out_name}'
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved summary of {len(rows)} seeds to {out_path}")

    n_excited = sum(r['excited'] for r in rows)
    print(f"\n{n_excited}/{len(rows)} seeds show resonance excitation "
          f"(suppression_ratio < {p_eq_suppression_flag})")

    return rows