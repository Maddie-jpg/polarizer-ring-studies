import numpy as np
import matplotlib.pyplot as plt


class TuneWPScanner:
    """
    Multi-objective working point scanner for (Qx, Qy) space.

    Produces heatmaps for:
        - tune stability
        - chromatic footprint
        - amplitude footprint
        - space charge proxy
        - dynamic aperture
        - combined objective
    """

    def __init__(
        self,
        line,
        qx_range,
        qy_range,
        nq=25,
        weights=None
    ):
        self.line = line

        self.qx_vals = np.linspace(*qx_range, nq)
        self.qy_vals = np.linspace(*qy_range, nq)

        self.QX, self.QY = np.meshgrid(self.qx_vals, self.qy_vals)
        self.shape = self.QX.shape

        # default weights
        self.weights = weights or {
            "tune": 0.2,
            "chrom": 0.2,
            "amp": 0.2,
            "sc": 0.1,
            "da": 0.3
        }

        # storage
        self.maps = {
            "tune": np.zeros(self.shape),
            "chrom": np.zeros(self.shape),
            "amp": np.zeros(self.shape),
            "sc": np.zeros(self.shape),
            "da": np.zeros(self.shape),
            "combined": np.zeros(self.shape),
        }

        self.best_wp = None

    # ---------------------------------------------------------
    # USER OVERRIDABLE PHYSICS METHODS
    # ---------------------------------------------------------

    def evaluate_metrics(self, qx, qy):
        """
        Replace THIS if you want full physics control.
        """

        line = self.line

        tw = line.twiss(method="6d")

        # --- tune stability ---
        tune = -((tw.qx - qx)**2 + (tw.qy - qy)**2)

        # --- chromatic footprint ---
        deltas = np.linspace(-0.03, 0.03, 5)
        qx_d, qy_d = [], []

        for d in deltas:
            try:
                t = line.twiss4d(delta0=d)
                qx_d.append(t.qx)
                qy_d.append(t.qy)
            except:
                pass

        chrom = -(np.std(qx_d) + np.std(qy_d))

        # --- amplitude footprint ---
        try:
            fp = line.get_footprint(
                nemitt_x=1e-6,
                nemitt_y=1e-6,
                r_range=(0.1, 3),
                n_r=6
            )
            amp = -(np.std(fp.qx) + np.std(fp.qy))
        except:
            amp = -1e3

        # --- space charge proxy ---
        sc = -abs(qx - qy)

        # --- DA proxy ---
        try:
            da = line.get_da(nemitt_x=1e-6, nemitt_y=1e-6)
            da_score = np.min(np.sqrt(da.x**2 + da.y**2))
        except:
            da_score = 0.0

        return tune, chrom, amp, sc, da_score

    # ---------------------------------------------------------
    # SCAN ENGINE
    # ---------------------------------------------------------

    def run_scan(self, verbose=True):

        for i, qx in enumerate(self.qx_vals):
            for j, qy in enumerate(self.qy_vals):

                if verbose:
                    print(f"Scanning Qx={qx:.4f}, Qy={qy:.4f}")

                t, c, a, s, d = self.evaluate_metrics(qx, qy)

                self.maps["tune"][i, j] = t
                self.maps["chrom"][i, j] = c
                self.maps["amp"][i, j] = a
                self.maps["sc"][i, j] = s
                self.maps["da"][i, j] = d

        self._build_combined()
        self._find_best_wp()

    # ---------------------------------------------------------
    # NORMALISATION + COMBINATION
    # ---------------------------------------------------------

    def _norm(self, M):
        return (M - M.min()) / (M.max() - M.min() + 1e-12)

    def _build_combined(self):

        n = self._norm

        self.maps["combined"] = (
            self.weights["tune"] * n(self.maps["tune"]) +
            self.weights["chrom"] * n(self.maps["chrom"]) +
            self.weights["amp"] * n(self.maps["amp"]) +
            self.weights["sc"] * n(self.maps["sc"]) +
            self.weights["da"] * n(self.maps["da"])
        )

    def _find_best_wp(self):
        idx = np.unravel_index(
            np.argmax(self.maps["combined"]),
            self.shape
        )

        self.best_wp = {
            "qx": self.qx_vals[idx[0]],
            "qy": self.qy_vals[idx[1]],
            "idx": idx
        }

    # ---------------------------------------------------------
    # PLOTTING (ALL HEATMAPS)
    # ---------------------------------------------------------

    def plot(self, show=True):

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))

        plots = [
            ("tune", "Tune footprint"),
            ("chrom", "Chromatic footprint"),
            ("amp", "Amplitude footprint"),
            ("sc", "Space charge"),
            ("da", "Minimum DA"),
            ("combined", "Combined score"),
        ]

        for ax, (key, title) in zip(axs.flat, plots):

            M = self._norm(self.maps[key])

            im = ax.imshow(
                M.T,
                origin="lower",
                extent=[
                    self.qx_vals.min(), self.qx_vals.max(),
                    self.qy_vals.min(), self.qy_vals.max()
                ],
                aspect="auto"
            )

            ax.scatter(
                self.best_wp["qx"],
                self.best_wp["qy"],
                c="red",
                s=80,
                edgecolors="black"
            )

            ax.set_title(title)
            ax.set_xlabel("Qx")
            ax.set_ylabel("Qy")

        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
        plt.tight_layout()

        if show:
            plt.show()

        return fig