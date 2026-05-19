import numpy as np
import matplotlib.pyplot as plt
from TuneDiagram.lib.TuneDiagram.tune_diagram import resonance_lines


class TuneFootprintAnalysis:

    # =====================================================
    # Init
    # =====================================================

    def __init__(
        self,
        line,
        twiss=None,
        resonance_orders=(1,2,3,4),
        Qx_range=None,
        Qy_range=None
    ):

        self.line = line
        self.twiss = twiss

        self.resonance_orders = resonance_orders

        self.Qx_range = Qx_range
        self.Qy_range = Qy_range

    # =====================================================
    # Generic plotting backend
    # =====================================================

    def _base_plot(
        self,
        Qx,
        Qy,
        values=None,
        label='',
        title='',
        cmap='viridis',
        nominal_point=None
    ):

        fig, ax = plt.subplots(figsize=(8,8))

        # Resonance diagram
        resonances = resonance_lines(
            self.Qx_range,
            self.Qy_range,
            self.resonance_orders,
            3
        )

        resonances.plot_resonance(fig)

        # Scatter
        sc = ax.scatter(
            Qx,
            Qy,
            c=values,
            cmap=cmap,
            s=10
        )

        cb = plt.colorbar(sc)
        cb.set_label(label)

        # Nominal WP
        if nominal_point is not None:

            ax.plot(
                nominal_point[0],
                nominal_point[1],
                'ro',
                label='Nominal WP'
            )

        ax.set_xlabel(r'$Q_x$')
        ax.set_ylabel(r'$Q_y$')

        ax.set_title(title)

        ax.grid(True, alpha=0.3)

        if nominal_point is not None:
            ax.legend()

        plt.tight_layout()

        return fig, ax

    # =====================================================
    # Chromatic footprint
    # =====================================================

    def chromatic_footprint(
        self,
        Qx,
        Qy,
        delta
    ):

        return self._base_plot(
            Qx,
            Qy,
            values=delta,
            label=r'$\delta$',
            title='Chromatic Footprint'
        )

    # =====================================================
    # Amplitude footprint
    # =====================================================

    def amplitude_footprint(
        self,
        Qx,
        Qy,
        amplitude
    ):

        return self._base_plot(
            Qx,
            Qy,
            values=amplitude,
            label='Amplitude',
            title='Amplitude Footprint'
        )

    # =====================================================
    # Frequency map analysis
    # =====================================================

    def diffusion_map(
        self,
        Qx_start,
        Qx_end,
        Qy_start,
        Qy_end
    ):

        diffusion = np.sqrt(
            (Qx_start - Qx_end)**2 +
            (Qy_start - Qy_end)**2
        )

        diffusion = np.log10(diffusion + 1e-20)

        return self._base_plot(
            Qx_start,
            Qy_start,
            values=diffusion,
            label=r'$\log_{10}(D)$',
            title='Frequency Map Analysis'
        )

    # =====================================================
    # Resonance finder
    # =====================================================

    def find_resonances(
        self,
        Qx,
        Qy,
        max_order=6,
        tolerance=1e-3
    ):

        resonances = []

        for m in range(-max_order, max_order+1):

            for n in range(-max_order, max_order+1):

                if m == 0 and n == 0:
                    continue

                order = abs(m) + abs(n)

                if order > max_order:
                    continue

                value = m*Qx + n*Qy

                p = int(round(value))

                distance = abs(value - p)

                if distance < tolerance:

                    resonances.append({
                        'm': m,
                        'n': n,
                        'p': p,
                        'order': order,
                        'distance': distance
                    })

        return resonances