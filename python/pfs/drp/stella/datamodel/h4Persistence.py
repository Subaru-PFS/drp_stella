from collections.abc import Sequence
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

import pfs.datamodel.h4Persistence
from pfs.datamodel.h4Persistence import TrapComponent

type FloatArray0D = np.ndarray[tuple[()], np.dtype[np.floating]]
type FloatArray1D = np.ndarray[tuple[int], np.dtype[np.floating]]
type FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type FloatArray3D = np.ndarray[tuple[int, int, int], np.dtype[np.floating]]

type IntArray1D = np.ndarray[tuple[int], np.dtype[np.integer]]


def _bounded_lsq_pgd(
    A: np.ndarray,
    b: np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
    max_iter: int = 20_000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Solve  min ||Ax - b||^2  s.t.  lo <= x_i <= hi  via FISTA.

    Uses the Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle
    2009) with box-projection, which converges at O(1/k^2) rate.
    """
    AtA = A.T @ A
    Atb = A.T @ b
    L = float(np.linalg.eigvalsh(AtA).max())
    if L == 0.0:
        return np.zeros(A.shape[1])
    alpha = 1.0 / L

    n = A.shape[1]
    x = np.clip(np.zeros(n), lo, hi)
    y = x.copy()
    t = 1.0

    for _ in range(max_iter):
        x_prev = x.copy()
        grad = AtA @ y - Atb
        x = np.clip(y - alpha * grad, lo, hi)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x + ((t - 1.0) / t_new) * (x - x_prev)
        t = t_new
        if np.linalg.norm(x - x_prev) < tol:
            break

    return x


@dataclasses.dataclass
class Illumination:
    """Illumination start time, duration, and flux.

    Parameters
    ----------
    t0 : `float`
        Illumination start time [s].
    texp : `float`
        Illumination duration [s].
    flux : `float`
        Photon flux during illumination [electrons/s].
    """

    t0: float
    texp: float
    flux: float

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_arrays(
        cls,
        t0: FloatArray1D,
        texp: FloatArray1D,
        flux: FloatArray1D,
    ) -> "list[Illumination]":
        """Construct list of instances from arrays.

        Parameters
        ----------
        t0 : `np.ndarray` of `float`
            Illumination start time [s].
        texp : `np.ndarray` of `float`
            Illumination duration [s].
        flux : `np.ndarray` of `float`
            Photon flux during each illumination [electrons/s].

        Returns
        -------
        illums : `list` [`Illumination`]
            List of `Illumination` instances.
        """
        return [Illumination(t0=float(a), texp=float(b), flux=float(c)) for a, b, c in zip(t0, texp, flux)]

    def validate(self) -> None:
        """Validate this `Illumination` instance.

        Raises
        ------
        ValueError
            Raised if any member is invalid.
        """
        if self.texp <= 0:
            raise ValueError(f"texp must be > 0, got {self.texp}")
        if self.flux < 0:
            raise ValueError(f"flux must be >= 0, got {self.flux}")

    @staticmethod
    def make_time_mask(
        illums: "Sequence[Illumination]",
        t_arr: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.bool]]:
        """Make a boolean array: True for times within any illumination window.

        Parameters
        ----------
        illums : `Sequence` [`Illumination`]
            List of illumination.
        t_arr : `np.ndarray` of `float`
            Array of times.

        Returns
        -------
        mask : `np.ndarray` of `bool`
            True for times within any illumination window.
        """
        mask = np.zeros(len(t_arr), dtype=bool)
        for illum in illums:
            mask |= (t_arr >= illum.t0) & (t_arr <= illum.t0 + illum.texp)
        return mask


class H4PersistenceModel(pfs.datamodel.h4Persistence.H4PersistenceModel):
    def update_fractions(
        self,
        fractions: FloatArray1D,
        fraction_errors: FloatArray1D,
        name: str | None = None,
    ) -> "H4PersistenceModel":
        """Return a new H4PersistenceModel with updated fractions; taus, and f_mids unchanged.

        Parameters
        ----------
        fractions : `np.array` of `float`
            New fractions, one per component.
        fraction_errors : `np.array` of `float`
            New fraction errors, one per component.
        name : `str` or None, optional
            Name for the new H4PersistenceModel.  Defaults to the current name.

        Returns
        -------
        newmodel : `H4PersistenceModel`
            New model.
        """
        if not (len(fractions) == len(fraction_errors) == self.n_components):
            raise ValueError(
                f"Expected {self.n_components} fractions"
                f" (len(fractions)={len(fractions)}, len(fraction_errors)={len(fraction_errors)})"
            )
        new_comps = [
            TrapComponent(fraction=float(f), tau=c.tau, f_mid=c.f_mid, fraction_error=float(e))
            for f, e, c in zip(fractions, fraction_errors, self._components)
        ]
        return H4PersistenceModel(
            new_comps,
            name=name if name is not None else self.name,
            spatialProfile=self.spatialProfile,
        )

    def releasedCharge(self, q_state: np.ndarray, dt: float) -> np.ndarray:
        """Get charge released during ``dt`` if there is no incoming flux.

        Parameters
        ----------
        q_state : `np.ndarray`, shape (..., K)
            Trapped charge.
        dt : `float`
            Delta time in seconds.

        Returns
        -------
        released : `np.ndarray`, shape (...)
            Released charge.
        """
        decay = np.exp(-dt / self.taus).astype(q_state.dtype)
        return q_state @ (1.0 - decay)

    def step(
        self,
        q_state: np.ndarray,
        flux_rate: np.ndarray,
        texp: float,
        dt_gap: float = 0.0,
        new_shape: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate trap state through one exposure and an optional trailing gap.

        ``q_state`` holds charges from the previous exposure that begin decaying
        at the start of this exposure.  New charges from this exposure start
        decaying at ``t_mid = f_mid * texp`` into the exposure.

        Parameters
        ----------
        q_state : `np.ndarray`, shape (..., K)
            Trapped charge that has been decaying since the start of this
            exposure (from the previous exposure).
        flux_rate : `np.ndarray`, shape (...)
            Photon flux rate during the exposure [e-/s].
        texp : `float`
            Exposure duration [s].
        dt_gap : `float`
            Time between the end of this exposure and the next evaluation
            point [s].
        new_shape : `np.ndarray`, optional
            Multiplicative spatial profile applied **only** to the charge newly
            trapped during this exposure, before it is added to the carried-over
            (decayed) state.  Must broadcast against the newly-trapped charge of
            shape ``(..., K)`` (e.g. ``(..., 1)`` to apply the same profile to
            every component).  Because the shape is baked into each charge packet
            exactly once at trapping time and is *not* re-applied to carried-over
            charge, it does not compound across successive :meth:`step` calls.
            If None (default), no shaping is applied.

        Returns
        -------
        q_end : `np.ndarray`, shape (..., K)
            Total trapped charge immediately after the exposure ends.
        q_next : `np.ndarray`, shape (..., K)
            State to pass as ``q_state`` to the next :meth:`step` call.
        persistence_released : `np.ndarray`, shape (...)
            Total charge released from traps during this exposure
            (from q_state only; new charges start decaying from t_mid).
        """
        t_delays = self.f_mids * texp  # (K,) no-decay duration per component
        # (K,) decay-active duration per component
        dt_2nds = texp - t_delays

        decay_full = np.exp(-texp / self.taus)  # (K,)
        decay_2nd = np.exp(-dt_2nds / self.taus)  # (K,)

        # Per-component coefficient for the charge newly trapped this exposure:
        #   q_new = flux_rate * fraction * (t_delay*decay_2nd + tau*(1-decay_2nd))
        # combining the old Phase 1 + Phase 2 terms.  This is a tiny (K,) array,
        # so the expensive broadcast over (..., K) happens only once below.
        coeff_new = self.fractions * (t_delays * decay_2nd + self.taus * (1.0 - decay_2nd))

        # Match the input dtype (typically float32) so the large (..., K)
        # arrays below are not silently promoted to float64.
        dtype = q_state.dtype
        decay_full = decay_full.astype(dtype, copy=False)
        coeff_new = coeff_new.astype(dtype, copy=False)

        # Charge released from carried-over traps: a matrix-vector product over
        # the component axis avoids materialising a (..., K) temporary.
        persistence_released = q_state @ (1.0 - decay_full)

        # Total trapped charge after the exposure: new charge + decayed old charge.
        # The spatial profile is applied to the newly-trapped charge only, so it
        # is imprinted on each charge packet once and never re-applied to the
        # carried-over (already-shaped) state --- avoiding runaway compounding.
        q_end = flux_rate[..., np.newaxis] * coeff_new
        if new_shape is not None:
            q_end *= np.asarray(new_shape, dtype=dtype)
        q_end += q_state * decay_full

        if dt_gap > 0:
            decay_gap = np.exp(-dt_gap / self.taus).astype(dtype, copy=False)
            q_next = q_end * decay_gap
        else:
            q_next = q_end.copy()

        return q_end, q_next, persistence_released

    def signal(self, t: float | np.ndarray, illums_quartz: Sequence[Illumination]) -> np.ndarray:
        """Total trapped charge at time(s) ``t``.

        Parameters
        ----------
        t : `float` or array-like
            Evaluation time(s) [s].
        illums_quartz : `Sequence` [`Illumination`]
            List of quartz illumination.

        Returns
        -------
        total : `np.ndarray`
            Total trapped charge [electrons].  Scalar when ``t`` is scalar.
        """
        t_arr = np.asarray(t, dtype=float)
        t_arr1d = t_arr.reshape(-1)
        total = np.zeros_like(t_arr1d)
        for comp in self._components:
            for illum in illums_quartz:
                total += comp.charge(t_arr1d, illum.t0, illum.texp, illum.flux)
        return total.reshape(t_arr.shape)

    def signal_by_component(
        self, t: float | np.ndarray, illums_quartz: Sequence[Illumination]
    ) -> dict[str, np.ndarray]:
        """Trapped charge broken down by component.

        Parameters
        ----------
        t : `float` or array-like
            Evaluation time(s) [s].
        illums_quartz : `Sequence` [`Illumination`]
            List of quartz illumination.

        Returns
        -------
        charges : `dict` [`str`, `np.ndarray`]
            Keys are component labels plus ``"total"``.
        """
        t_arr = np.asarray(t, dtype=float)
        t_arr1d = t_arr.reshape(-1)
        result: dict[str, np.ndarray] = {}
        total = np.zeros_like(t_arr1d)
        for comp in self._components:
            q = np.zeros_like(t_arr1d)
            for illum in illums_quartz:
                q += comp.charge(t_arr1d, illum.t0, illum.texp, illum.flux)
            result[comp.label] = q.reshape(t_arr.shape)
            total += q
        result["total"] = total.reshape(t_arr.shape)
        return result

    def persistence(
        self, t: float | FloatArray1D, illums_quartz: Sequence[Illumination], texp_dark: float = 300.0
    ) -> np.ndarray:
        """Charge released during a dark exposure starting at time ``t``.

        Persistence is defined as the total trapped charge that decays
        during the dark exposure window [t, t + texp_dark]:

            P(t) = Q(t) - Q(t + texp_dark)

        Parameters
        ----------
        t : `float` or array-like
            Start time(s) of the dark exposure [s].
        illums_quartz : `Sequence` [`Illumination`]
            List of quartz illumination.
        texp_dark : `float`, optional
            Dark exposure duration [s].  Default: 300.

        Returns
        -------
        total : `np.ndarray`
            Persistence signal [electrons] at each time in ``t``.
        """
        if texp_dark <= 0:
            raise ValueError(f"texp_dark must be > 0, got {texp_dark}")
        t_arr = np.asarray(t, dtype=float)
        t_arr1d = t_arr.reshape(-1)
        result = np.zeros_like(t_arr1d)
        for comp in self._components:
            for illum in illums_quartz:
                mask = t_arr1d > illum.t0 + illum.texp
                if not np.any(mask):
                    continue
                result[mask] += comp.charge(t_arr1d[mask], illum.t0, illum.texp, illum.flux) - comp.charge(
                    t_arr1d[mask] + texp_dark, illum.t0, illum.texp, illum.flux
                )
        np.maximum(result, 0.0, out=result)
        return result.reshape(t_arr.shape)

    def persistence_by_component(
        self, t: float | FloatArray1D, illums_quartz: Sequence[Illumination], texp_dark: float = 300.0
    ) -> dict[str, np.ndarray]:
        """Persistence broken down by component.

        Parameters
        ----------
        t : `float` or array-like
            Start time(s) of the dark exposure [s].
        illums_quartz : `Sequence` [`Illumination`]
            List of quartz illumination.
        texp_dark : `float`, optional
            Dark exposure duration [s].  Default: 300.

        Returns
        -------
        persistences : `dict` [`str`, `np.ndarray`]
            Keys are component labels plus ``"total"``.
        """
        if texp_dark <= 0:
            raise ValueError(f"texp_dark must be > 0, got {texp_dark}")
        t_arr = np.asarray(t, dtype=float)
        t_arr1d = t_arr.reshape(-1)
        result: dict[str, np.ndarray] = {}
        total = np.zeros_like(t_arr1d)
        for comp in self._components:
            q = np.zeros_like(t_arr1d)
            for illum in illums_quartz:
                mask = t_arr1d > illum.t0 + illum.texp
                if not np.any(mask):
                    continue
                q[mask] += comp.charge(t_arr1d[mask], illum.t0, illum.texp, illum.flux) - comp.charge(
                    t_arr1d[mask] + texp_dark, illum.t0, illum.texp, illum.flux
                )
            np.maximum(q, 0.0, out=q)
            result[comp.label] = q.reshape(t_arr.shape)
            total += q
        result["total"] = total.reshape(t_arr.shape)
        return result

    def fit_fractions(
        self,
        t_obs: np.ndarray,
        p_obs: np.ndarray,
        illums_quartz: Sequence[Illumination],
        texp_dark: float = 300.0,
        dp_obs: np.ndarray | None = None,
        fixed_indices: Sequence[int] | None = None,
    ) -> tuple["H4PersistenceModel", dict]:
        r"""Fit trap fractions to observed persistence data.

        The model persistence is linear in fractions, so the problem reduces
        to bounded linear least squares:

            min  ||A @ f - p_obs||^2    subject to   0 <= f_i <= 1

        Normalization convention
        ------------------------
        p_obs must be normalized by (\sum_k flux_k * texp_k) * texp_dark:

            p_obs = P_actual [e^-] / (\sum_k flux_k \cdot texp_k [e^-] * texp_dark [s])

        Parameters
        ----------
        t_obs : array-like
            Observation times [s].  Should be > t0 + texp_quartz.
        p_obs : array-like
            Observed persistence normalized by (\sum flux_k \cdot texp_k) * texp_dark.
        illums_quartz : `Sequence` [`Illumination`]
            List of quartz illumination.
        texp_dark : `float`, optional
            Dark exposure duration [s].  Default: 300.
        dp_obs : array-like, optional
            1-sigma errors on ``p_obs``.  When given, the fit minimises
            ``((model - p_obs) / dp_obs)^2``.
        fixed_indices : `Sequence` [`int`], optional
            Indices of components whose fractions are held fixed.

        Returns
        -------
        fitted_det : `H4PersistenceModel`
            New H4PersistenceModel with fitted fractions; taus and f_mids unchanged.
        info : `dict`
            Fitting diagnostics: fractions, fraction_errors, residuals, rms,
            cost, success, solver, dataframe, metadata.
        """
        t_arr = np.asarray(t_obs, dtype=float).reshape(-1)
        p_arr = np.asarray(p_obs, dtype=float).reshape(-1)
        norm = sum(illum.flux * illum.texp for illum in illums_quartz) * texp_dark
        mask_illuminated = Illumination.make_time_mask(illums_quartz, t_arr)

        A = np.zeros((len(t_arr), self.n_components))
        for j, comp in enumerate(self._components):
            unit = TrapComponent(
                fraction=1.0, tau=comp.tau, f_mid=comp.f_mid, fraction_error=comp.fraction_error
            )
            col = np.zeros(len(t_arr))
            for illum in illums_quartz:
                col += unit.charge(t_arr, illum.t0, illum.texp, illum.flux) - unit.charge(
                    t_arr + texp_dark, illum.t0, illum.texp, illum.flux
                )
            col[mask_illuminated] = 0.0
            A[:, j] = col / norm

        fixed_set = set(fixed_indices) if fixed_indices is not None else set()
        free_idx = [j for j in range(self.n_components) if j not in fixed_set]

        p_adj = p_arr.copy()
        for j in fixed_set:
            p_adj -= A[:, j] * self._components[j].fraction

        if dp_obs is not None:
            w = 1.0 / np.asarray(dp_obs, dtype=float).reshape(-1)
            A_fit = A[:, free_idx] * w[:, None]
            p_fit = p_adj * w
        else:
            A_fit = A[:, free_idx]
            p_fit = p_adj

        finite_mask = np.isfinite(p_fit) & np.all(np.isfinite(A_fit), axis=1)
        if not np.any(finite_mask):
            raise ValueError("No finite data points after weighting. " "Check dp_obs for zero or NaN values.")
        A_solve = A_fit[finite_mask]
        p_solve = p_fit[finite_mask]

        fitted_f_free = None
        solver = ""
        try:
            try:
                result = lsq_linear(A_solve, p_solve, bounds=(0.0, 1.0), method="bvls")
                if np.all(np.isfinite(result.x)):
                    fitted_f_free = result.x
                    solver = "scipy lsq_linear (bvls)"
            except np.linalg.LinAlgError:
                pass
            if fitted_f_free is None:
                result = lsq_linear(
                    A_solve,
                    p_solve,
                    bounds=(0.0, 1.0),
                    method="trf",
                    lsq_solver="lsmr",
                )
                if np.all(np.isfinite(result.x)):
                    fitted_f_free = result.x
                    solver = "scipy lsq_linear (trf+lsmr)"
        except ImportError:
            pass
        if fitted_f_free is None:
            fitted_f_free = _bounded_lsq_pgd(A_solve, p_solve, lo=0.0, hi=1.0)
            solver = "numpy projected gradient descent"

        fitted_f = np.array([comp.fraction for comp in self._components])
        for k, j in enumerate(free_idx):
            fitted_f[j] = fitted_f_free[k]

        residuals = A @ fitted_f - p_arr
        rms = np.sqrt(np.mean(residuals**2))

        AtA_fit = A_solve.T @ A_solve
        n_free = len(free_idx)
        try:
            cov_free = np.linalg.lstsq(AtA_fit, np.eye(n_free), rcond=None)[0]
        except np.linalg.LinAlgError:
            cov_free = np.zeros((n_free, n_free))
        if dp_obs is None:
            dof = max(1, len(p_arr) - n_free)
            cov_free = cov_free * (np.dot(residuals, residuals) / dof)
        df_free = np.sqrt(np.maximum(0.0, np.diag(cov_free)))

        df = np.zeros(self.n_components)
        for k, j in enumerate(free_idx):
            df[j] = df_free[k]

        fitted_det = self.update_fractions(
            fractions=fitted_f, fraction_errors=df, name=self.name + " (fitted)"
        )

        result_df = pd.DataFrame(
            {
                "tau": [c.tau for c in fitted_det.components],
                "f_mid": [c.f_mid for c in fitted_det.components],
                "fraction": fitted_f,
                "fraction_error": df,
            }
        )

        info = {
            "fractions": fitted_f,
            "fraction_errors": df,
            "dataframe": result_df,
            "metadata": {"name": self.name},
            "residuals": residuals,
            "rms": rms,
            "cost": float(0.5 * np.dot(residuals, residuals)),
            "success": True,
            "solver": solver,
        }
        return fitted_det, info

    def plot(
        self,
        illums_quartz: Sequence[Illumination],
        *,
        texp_dark: float = 300.0,
        t_extra: float = 3600.0,
        n_points: int = 2000,
        log_scale: bool = False,
        t_obs: np.ndarray | None = None,
        p_obs: np.ndarray | None = None,
        dp_obs: np.ndarray | None = None,
        xlim: tuple[float, float] | None = None,
        ylim1: tuple[float, float] | None = None,
        ylim2: tuple[float, float] | None = None,
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot illumination timeline, trapped charge, and persistence.

        Parameters
        ----------
        illums_quartz : `Sequence` [`Illumination`]
            List of quartz illumination.
        texp_dark : `float`, optional
            Dark exposure duration [s].  Default: 300.
        t_extra : `float`, optional
            Time window after the end of the last illumination [s].  Default: 3600.
            Ignored when ``xlim`` is provided.
        n_points : `int`, optional
            Number of sample points on the time axis.  Default: 2000.
        log_scale : `bool`, optional
            If True, use log scale on both axes of the signal panels.
        t_obs, p_obs, dp_obs : array-like or None
            Observed data to overlay on the persistence panel.
        xlim, ylim1, ylim2 : `tuple` or None
            Axis limits.
        show : `bool`, optional
            Call ``plt.show()`` if True.
        save_path : `str` or None, optional
            Save figure to this path (dpi=150) if given.

        Returns
        -------
        figure : `matplotlib.figure.Figure`
            Figure.
        """
        t_start = min(illum.t0 for illum in illums_quartz)
        t_end_last = max(illum.t0 + illum.texp for illum in illums_quartz)

        if xlim is not None:
            t_plot_start, t_stop = float(xlim[0]), float(xlim[1])
        else:
            t_plot_start, t_stop = t_start, t_end_last + t_extra

        if log_scale:
            t = np.logspace(np.log10(max(t_plot_start, 1e-3)), np.log10(t_stop), n_points)
        else:
            t = np.linspace(t_plot_start, t_stop, n_points)

        data_signal = self.signal_by_component(t, illums_quartz)
        data_pers = self.persistence_by_component(t, illums_quartz, texp_dark)
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        fig, (ax0, ax1, ax2) = plt.subplots(
            3,
            1,
            figsize=(10, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 3, 3]},
        )
        fig.suptitle(
            f"PFS H4RG Persistence Model --- {self.name}",
            fontsize=14,
        )

        # --- Panel 0: illumination timeline with t_mid markers ---
        for k, illum in enumerate(illums_quartz):
            t_end_k = illum.t0 + illum.texp
            label = "Illumination" if k == 0 else None
            ax0.axvspan(illum.t0, t_end_k, ymin=0.2, ymax=0.8, color="gold", alpha=0.8, label=label)
            ax0.axvline(t_end_k, color="steelblue", lw=1.5, ls="--")
            # Mark t_mid for each component (use mean f_mid if they differ)
            mean_f_mid = float(np.mean(self.f_mids))
            t_mid_k = illum.t0 + mean_f_mid * illum.texp
            mid_label = "t_mid (mean)" if k == 0 else None
            ax0.axvline(t_mid_k, color="darkorange", lw=1.2, ls=":", label=mid_label)
            ax0.text(
                (illum.t0 + t_end_k) / 2,
                0.50,
                f"#{k+1}  $\\Phi$={illum.flux:.2e}\ntexp={illum.texp:.1f} s",
                ha="center",
                va="center",
                fontsize=7,
                color="saddlebrown",
                transform=ax0.get_xaxis_transform(),
            )
        ax0.set_yticks([])
        ax0.set_ylabel("Timeline", fontsize=9)
        ax0.set_title(f"Illumination timeline  ({len(illums_quartz)} event(s))", fontsize=10)
        ax0.legend(loc="upper right", fontsize=8)
        ax0.set_ylim(0, 1)

        # --- Panel 1: individual components + total (trapped charge) ---
        for i, comp in enumerate(self._components):
            ax1.plot(
                t,
                data_signal[comp.label],
                color=colors[i % 10],
                label=f"{comp.label}  (f={comp.fraction:.4f}, f_mid={comp.f_mid:.2f})",
            )
        ax1.plot(t, data_signal["total"], color="black", lw=2.5, label="Total")
        for illum in illums_quartz:
            ax1.axvline(illum.t0 + illum.texp, color="steelblue", lw=1.5, ls="--")
            mean_f_mid = float(np.mean(self.f_mids))
            ax1.axvline(illum.t0 + mean_f_mid * illum.texp, color="darkorange", lw=1.2, ls=":")
        ax1.set_ylabel("Trapped charge [e${}^-$]")
        ax1.set_title("Trapped charge --- components & total")
        ax1.legend(fontsize=8)
        ax1.grid(True, which="both", alpha=0.3)
        if log_scale:
            ax1.set_yscale("log")
            ax1.set_xscale("log")

        # --- Panel 2: persistence normalized by \sum(flux \cdot texp) * texp_dark ---
        norm = sum(illum.flux * illum.texp for illum in illums_quartz) * texp_dark
        for i, comp in enumerate(self._components):
            ax2.plot(
                t,
                data_pers[comp.label] / norm,
                color=colors[i % 10],
                label=f"{comp.label}  (f={comp.fraction:.4f}, f_mid={comp.f_mid:.2f})",
            )
        ax2.plot(t, data_pers["total"] / norm, color="black", lw=2.5, label="Total")
        if t_obs is not None and p_obs is not None:
            if dp_obs is not None:
                ax2.errorbar(
                    np.asarray(t_obs, dtype=float),
                    np.asarray(p_obs, dtype=float),
                    yerr=np.asarray(dp_obs, dtype=float),
                    fmt="o",
                    ms=4,
                    color="red",
                    ecolor="red",
                    elinewidth=1,
                    capsize=3,
                    zorder=5,
                    label="Observed",
                    alpha=0.7,
                )
            else:
                ax2.scatter(
                    np.asarray(t_obs, dtype=float),
                    np.asarray(p_obs, dtype=float),
                    s=15,
                    color="red",
                    zorder=5,
                    label="Observed",
                    alpha=0.7,
                )
        for illum in illums_quartz:
            ax2.axvline(illum.t0 + illum.texp, color="steelblue", lw=1.5, ls="--")
            mean_f_mid = float(np.mean(self.f_mids))
            ax2.axvline(illum.t0 + mean_f_mid * illum.texp, color="darkorange", lw=1.2, ls=":")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel(r"Persistence / ($\sum$ flux $\cdot$ texp $\times$ t_exp,dark)")
        ax2.set_title(f"Persistence  (t_exp,dark = {texp_dark:.0f} s)")
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", alpha=0.3)
        if log_scale:
            ax2.set_yscale("log")
            ax2.set_xscale("log")

        if xlim is not None:
            ax0.set_xlim(xlim)
        if ylim1 is not None:
            ax1.set_ylim(ylim1)
        if ylim2 is not None:
            ax2.set_ylim(ylim2)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def summary(self, illum_quartz: Illumination | None = None) -> None:
        """Print a formatted summary of trap components.

        If illum_quartz is provided, also prints the trapped charge at key times
        relative to the end of illumination.

        Parameters
        ----------
        illum_quartz : `Illumination`, optional
            Quartz illumination.
        """
        print(f"H4PersistenceModel: {self.name}")
        print(f"  Components   : {self.n_components}")
        print(f"  Total fraction: {self.total_fraction:.4f}")
        print()

        df_arr = self.fraction_errors
        print(
            f"  {'#':>3}  {'label':<22}  {'fraction':>10}  {'+- error':>10}"
            f"  {'f_mid':>6}  {'tau [s]':>10}"
        )
        print("  " + "-" * 74)
        for i, comp in enumerate(self._components):
            print(
                f"  {i:>3}  {comp.label:<22}  {comp.fraction:>10.5f}"
                f"  {df_arr[i]:>10.5f}  {comp.f_mid:>6.3f}  {comp.tau:>10.1f}"
            )

        if illum_quartz is not None:
            t_end = illum_quartz.t0 + illum_quartz.texp
            total_in = illum_quartz.flux * illum_quartz.texp
            check_dt = [0.0, illum_quartz.texp, 10 * illum_quartz.texp, 100 * illum_quartz.texp]
            labels = [
                "end of illumination (t_end)",
                f"+{illum_quartz.texp:.0f} s after t_end",
                f"+{10*illum_quartz.texp:.0f} s after t_end",
                f"+{100*illum_quartz.texp:.0f} s after t_end",
            ]
            print()
            print(
                f"  t0={illum_quartz.t0} s,"
                f" texp_quartz={illum_quartz.texp} s,"
                f" flux={illum_quartz.flux:.2e} e-/s"
            )
            print(f"  Total incoming charge = {total_in:.3e} e-")
            print()
            print(f"  {'label':<30} {'time [s]':>10} {'trapped [e-]':>14} {'fraction':>10}")
            print("  " + "-" * 69)
            for dt, lbl in zip(check_dt, labels):
                tc = t_end + dt
                q = self.signal(tc, [illum_quartz])
                print(f"  {lbl:<30} {tc:>10.1f} {float(q):>14.2f}" f" {float(q)/total_in:>10.2e}")
