import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_tune_diagram(df, qx, qy, tune_window=0.05, save_path="tune_diagram.png"):
    """Plots a localized 2D Tune Diagram centered around the working point

    showing nearby resonance lines color-coded by order and styled by Verdier
    status.
    """
    if df.empty:
        print("The resonance DataFrame is empty. Nothing to plot.")
        return

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 7))

    # Define the viewing window around the working point
    qx_min, qx_max = qx - tune_window, qx + tune_window
    qy_min, qy_max = qy - tune_window, qy + tune_window

    ax.set_xlim(qx_min, qx_max)
    ax.set_ylim(qy_min, qy_max)

    # Plot the current Operating Working Point
    ax.plot(
        qx,
        qy,
        "ro",
        markersize=10,
        label=f"Working Point ({qx:.4f}, {qy:.4f})",
        zorder=5,
    )

    # Styling properties for resonance orders
    # Distinct colors for orders 1 through 5+
    order_colors = {
        1: "blue",
        2: "green",
        3: "darkorange",
        4: "purple",
        5: "brown",
    }

    # Plot each resonance line found in the dataframe
    for _, row in df.iterrows():
        m = int(row["m"])
        n = int(row["n"])
        k = int(row["Global Harmonic (k)"])
        order = int(row["Order"])
        status = row["Verdier Status"]

        # Solid lines for Dangerous/Structural, dashed for Suppressed/Non-Structural
        linestyle = "-" if "Structural" in status else "--"
        # Thicker lines for lower-order resonances (which are usually stronger)
        linewidth = max(0.8, 3.5 - 0.5 * order)
        color = order_colors.get(order, "black")
        alpha = 0.8 if "Structural" in status else 0.35

        # Handle different line orientations safely
        if n == 0:  # Vertical line: m*Qx = k -> Qx = k/m
            if m != 0:
                qx_val = k / m
                if qx_min <= qx_val <= qx_max:
                    ax.axvline(
                        x=qx_val,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha,
                    )
        elif m == 0:  # Horizontal line: n*Qy = k -> Qy = k/n
            if n != 0:
                qy_val = k / n
                if qy_min <= qy_val <= qy_max:
                    ax.axhline(
                        y=qy_val,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha,
                    )
        else:  # Sloped resonance line: Qy = (k - m*Qx) / n
            qx_vals = np.array([qx_min, qx_max])
            qy_vals = (k - m * qx_vals) / n
            # Only plot if it intersects or passes near our window bounds
            ax.plot(
                qx_vals,
                qy_vals,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

    # Build a clean, clear custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="red",
            marker="o",
            linestyle="",
            markersize=10,
            label=f"Working Point ({qx:.3f}, {qy:.3f})",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Dangerous (Structural)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            label="Suppressed (Non-Structural)",
        ),
    ]

    # Add indicators for each active resonance order present in data
    for o in sorted(df["Order"].unique()):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=order_colors.get(o, "black"),
                linestyle="-",
                linewidth=2,
                label=f"Order {o} Resonance",
            )
        )

    # Position the legend outside the plot area to prevent overlapping data
    ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1)
    )

    # Labels and Titles
    ax.set_xlabel(r"Horizontal Tune $Q_x$", fontsize=12)
    ax.set_ylabel(r"Vertical Tune $Q_y$", fontsize=12)
    ax.set_title(
        "Local Tune Diagram & Verdier Resonance Analysis",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
