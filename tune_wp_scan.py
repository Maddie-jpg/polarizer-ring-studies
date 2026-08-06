import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from TuneDiagram.lib.TuneDiagram.tune_diagram import resonance_lines
from xutil_DA_CC.xsuite_plot_functions import DA_vs_turns

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


def plot_DA_tune_scan(qx_range, qy_range, particles, num_r_steps, num_theta_steps, x_norm, y_norm, delta_initial):
    """
    qx_range: 1D array of qx values to scan
    qy_range: 1D array of qy values to scan
    """
    # Create grids for tunes
    QX, QY = np.meshgrid(qx_range, qy_range)
    
    # Initialize a 2D array to store the minimum DA for each working point
    min_DA_grid = np.zeros_like(QX)

    # Loop through the grid
    for i in range(len(qy_range)):
        for j in range(len(qx_range)):
            current_qx = qx_range[j]
            current_qy = qy_range[i]
            
            # --- 1. SIMULATION STEP ---
            # You need to update your accelerator lattice tunes to (current_qx, current_qy)
            # and rerun your tracking simulation here to get the new 'particles' object.
            # Example: particles = run_tracking(current_qx, current_qy)
            
            # For demonstration, let's assume 'particles' is generated here:
            
            # --- 2. CALCULATE DA ---
            # Call your function (disable plotting inside the loop to avoid 1000s of windows)
            _, _, _, min_DA = DA_vs_turns(
                particles, num_r_steps, num_theta_steps, 
                x_norm, y_norm, delta_initial, delta_plots=False
            )
            
            # Store the result
            min_DA_grid[i, j] = min_DA

    # --- 3. PLOT THE HEATMAP ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Using pcolormesh for a clean heatmap grid
    c = ax.pcolormesh(QX, QY, min_DA_grid, cmap='viridis', shading='auto')
    
    # Add labels and styling
    ax.set_xlabel(r'Horizontal Tune $q_x$')
    ax.set_ylabel(r'Vertical Tune $q_y$')
    ax.set_title(r'Minimum Dynamic Aperture [$\sigma$]')
    
    # Colorbar
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r'Minimum DA ($\sigma$)')
    
    plt.show()