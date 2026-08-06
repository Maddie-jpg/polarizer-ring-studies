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
