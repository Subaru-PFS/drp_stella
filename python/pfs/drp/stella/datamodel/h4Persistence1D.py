from collections.abc import Sequence
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

import pfs.datamodel.h4Persistence1D
from pfs.datamodel.h4Persistence1D import TrapComponent

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
    tExp : `float`
        Illumination duration [s].
    flux : `float`
        Photon flux during illumination [electrons/s].
    """

    t0: float
    tExp: float
    flux: float

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_arrays(
        cls,
        t0: FloatArray1D,
        tExp: FloatArray1D,
        flux: FloatArray1D,
    ) -> "list[Illumination]":
        """Construct list of instances from arrays.

        Parameters
        ----------
        t0 : `np.ndarray` of `float`
            Illumination start time [s].
        tExp : `np.ndarray` of `float`
            Illumination duration [s].
        flux : `np.ndarray` of `float`
            Photon flux during each illumination [electrons/s].

        Returns
        -------
        illums : `list` [`Illumination`]
            List of `Illumination` instances.
        """
        return [Illumination(t0=float(a), tExp=float(b), flux=float(c)) for a, b, c in zip(t0, tExp, flux)]

    def validate(self) -> None:
        """Validate this `Illumination` instance.

        Raises
        ------
        ValueError
            Raised if any member is invalid.
        """
        if self.tExp <= 0:
            raise ValueError(f"tExp must be > 0, got {self.tExp}")
        if self.flux < 0:
            raise ValueError(f"flux must be >= 0, got {self.flux}")

    @staticmethod
    def makeTimeMask(
        illums: "Sequence[Illumination]",
        tArr: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.bool]]:
        """Make a boolean array: True for times within any illumination window.

        Parameters
        ----------
        illums : `Sequence` [`Illumination`]
            List of illumination.
        tArr : `np.ndarray` of `float`
            Array of times.

        Returns
        -------
        mask : `np.ndarray` of `bool`
            True for times within any illumination window.
        """
        mask = np.zeros(len(tArr), dtype=bool)
        for illum in illums:
            mask |= (tArr >= illum.t0) & (tArr <= illum.t0 + illum.tExp)
        return mask


class H4Persistence1DBasicModel(pfs.datamodel.h4Persistence1D.H4Persistence1DBasicModel):
    def updateFractions(
        self,
        fractions: FloatArray1D,
        fractionErrs: FloatArray1D,
        name: str | None = None,
    ) -> "H4Persistence1DBasicModel":
        """Return a new H4Persistence1DBasicModel with updated fractions; taus, and fMids unchanged.

        Parameters
        ----------
        fractions : `np.array` of `float`
            New fractions, one per component.
        fractionErrs : `np.array` of `float`
            New fraction errors, one per component.
        name : `str` or None, optional
            Name for the new H4Persistence1DBasicModel.  Defaults to the current name.

        Returns
        -------
        newmodel : `H4Persistence1DBasicModel`
            New model.
        """
        if not (len(fractions) == len(fractionErrs) == self.nComponents):
            raise ValueError(
                f"Expected {self.nComponents} fractions"
                f" (len(fractions)={len(fractions)}, len(fractionErrs)={len(fractionErrs)})"
            )
        new_comps = [
            TrapComponent(fraction=float(f), tau=c.tau, fMid=c.fMid, fractionErr=float(e))
            for f, e, c in zip(fractions, fractionErrs, self.components)
        ]
        return H4Persistence1DBasicModel(
            new_comps,
            name=name if name is not None else self.name,
            spatialProfile=self.spatialProfile,
        )

    def releasedCharge(self, qState: np.ndarray, dt: float) -> np.ndarray:
        """Get charge released during ``dt`` if there is no incoming flux.

        Parameters
        ----------
        qState : `np.ndarray`, shape (..., K)
            Trapped charge.
        dt : `float`
            Delta time in seconds.

        Returns
        -------
        released : `np.ndarray`, shape (...)
            Released charge.
        """
        decay = np.exp(-dt / self.taus).astype(qState.dtype)
        return qState @ (1.0 - decay)

    def step(
        self,
        qState: np.ndarray,
        fluxRate: np.ndarray,
        tExp: float,
        dtGap: float = 0.0,
        newShape: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate trap state through one exposure and an optional trailing gap.

        ``qState`` holds charges from the previous exposure that begin decaying
        at the start of this exposure.  New charges from this exposure start
        decaying at ``tMid = fMid * tExp`` into the exposure.

        Parameters
        ----------
        qState : `np.ndarray`, shape (..., K)
            Trapped charge that has been decaying since the start of this
            exposure (from the previous exposure).
        fluxRate : `np.ndarray`, shape (...)
            Photon flux rate during the exposure [e-/s].
        tExp : `float`
            Exposure duration [s].
        dtGap : `float`
            Time between the end of this exposure and the next evaluation
            point [s].
        newShape : `np.ndarray`, optional
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
        qEnd : `np.ndarray`, shape (..., K)
            Total trapped charge immediately after the exposure ends.
        qNext : `np.ndarray`, shape (..., K)
            State to pass as ``qState`` to the next :meth:`step` call.
        releasedPersistence : `np.ndarray`, shape (...)
            Total charge released from traps during this exposure
            (from qState only; new charges start decaying from tMid).
        """
        tDelays = self.fMids * tExp  # (K,) no-decay duration per component
        # (K,) decay-active duration per component
        dt2nds = tExp - tDelays

        decayFull = np.exp(-tExp / self.taus)  # (K,)
        decay2nd = np.exp(-dt2nds / self.taus)  # (K,)

        # Per-component coefficient for the charge newly trapped this exposure:
        #   qNew = fluxRate * fraction * (tDelay*decay2nd + tau*(1-decay2nd))
        # combining the old Phase 1 + Phase 2 terms.  This is a tiny (K,) array,
        # so the expensive broadcast over (..., K) happens only once below.
        newCoeff = self.fractions * (tDelays * decay2nd + self.taus * (1.0 - decay2nd))

        # Match the input dtype (typically float32) so the large (..., K)
        # arrays below are not silently promoted to float64.
        dtype = qState.dtype
        decayFull = decayFull.astype(dtype, copy=False)
        newCoeff = newCoeff.astype(dtype, copy=False)

        # Charge released from carried-over traps: a matrix-vector product over
        # the component axis avoids materialising a (..., K) temporary.
        releasedPersistence = qState @ (1.0 - decayFull)

        # Total trapped charge after the exposure: new charge + decayed old charge.
        # The spatial profile is applied to the newly-trapped charge only, so it
        # is imprinted on each charge packet once and never re-applied to the
        # carried-over (already-shaped) state --- avoiding runaway compounding.
        qEnd = fluxRate[..., np.newaxis] * newCoeff
        if newShape is not None:
            qEnd *= np.asarray(newShape, dtype=dtype)
        qEnd += qState * decayFull

        if dtGap > 0:
            decayGap = np.exp(-dtGap / self.taus).astype(dtype, copy=False)
            qNext = qEnd * decayGap
        else:
            qNext = qEnd.copy()

        return qEnd, qNext, releasedPersistence

    def signal(self, t: float | np.ndarray, quartzIllums: Sequence[Illumination]) -> np.ndarray:
        """Total trapped charge at time(s) ``t``.

        Parameters
        ----------
        t : `float` or array-like
            Evaluation time(s) [s].
        quartzIllums : `Sequence` [`Illumination`]
            List of quartz illumination.

        Returns
        -------
        total : `np.ndarray`
            Total trapped charge [electrons].  Scalar when ``t`` is scalar.
        """
        tArr = np.asarray(t, dtype=float)
        tArr1d = tArr.reshape(-1)
        total = np.zeros_like(tArr1d)
        for comp in self.components:
            for illum in quartzIllums:
                total += comp.charge(tArr1d, illum.t0, illum.tExp, illum.flux)
        return total.reshape(tArr.shape)

    def signalComponents(
        self, t: float | np.ndarray, quartzIllums: Sequence[Illumination]
    ) -> dict[str, np.ndarray]:
        """Trapped charge broken down by component.

        Parameters
        ----------
        t : `float` or array-like
            Evaluation time(s) [s].
        quartzIllums : `Sequence` [`Illumination`]
            List of quartz illumination.

        Returns
        -------
        charges : `dict` [`str`, `np.ndarray`]
            Keys are component labels plus ``"total"``.
        """
        tArr = np.asarray(t, dtype=float)
        tArr1d = tArr.reshape(-1)
        result: dict[str, np.ndarray] = {}
        total = np.zeros_like(tArr1d)
        for comp in self.components:
            q = np.zeros_like(tArr1d)
            for illum in quartzIllums:
                q += comp.charge(tArr1d, illum.t0, illum.tExp, illum.flux)
            result[comp.label] = q.reshape(tArr.shape)
            total += q
        result["total"] = total.reshape(tArr.shape)
        return result

    def persistence(
        self, t: float | FloatArray1D, quartzIllums: Sequence[Illumination], tExpDark: float = 300.0
    ) -> np.ndarray:
        """Charge released during a dark exposure starting at time ``t``.

        Persistence is defined as the total trapped charge that decays
        during the dark exposure window [t, t + tExpDark]:

            P(t) = Q(t) - Q(t + tExpDark)

        Parameters
        ----------
        t : `float` or array-like
            Start time(s) of the dark exposure [s].
        quartzIllums : `Sequence` [`Illumination`]
            List of quartz illumination.
        tExpDark : `float`, optional
            Dark exposure duration [s].  Default: 300.

        Returns
        -------
        total : `np.ndarray`
            Persistence signal [electrons] at each time in ``t``.
        """
        if tExpDark <= 0:
            raise ValueError(f"tExpDark must be > 0, got {tExpDark}")
        tArr = np.asarray(t, dtype=float)
        tArr1d = tArr.reshape(-1)
        result = np.zeros_like(tArr1d)
        for comp in self.components:
            for illum in quartzIllums:
                mask = tArr1d > illum.t0 + illum.tExp
                if not np.any(mask):
                    continue
                result[mask] += comp.charge(tArr1d[mask], illum.t0, illum.tExp, illum.flux) - comp.charge(
                    tArr1d[mask] + tExpDark, illum.t0, illum.tExp, illum.flux
                )
        np.maximum(result, 0.0, out=result)
        return result.reshape(tArr.shape)

    def persistenceComponents(
        self, t: float | FloatArray1D, quartzIllums: Sequence[Illumination], tExpDark: float = 300.0
    ) -> dict[str, np.ndarray]:
        """Persistence broken down by component.

        Parameters
        ----------
        t : `float` or array-like
            Start time(s) of the dark exposure [s].
        quartzIllums : `Sequence` [`Illumination`]
            List of quartz illumination.
        tExpDark : `float`, optional
            Dark exposure duration [s].  Default: 300.

        Returns
        -------
        persistences : `dict` [`str`, `np.ndarray`]
            Keys are component labels plus ``"total"``.
        """
        if tExpDark <= 0:
            raise ValueError(f"tExpDark must be > 0, got {tExpDark}")
        tArr = np.asarray(t, dtype=float)
        tArr1d = tArr.reshape(-1)
        result: dict[str, np.ndarray] = {}
        total = np.zeros_like(tArr1d)
        for comp in self.components:
            q = np.zeros_like(tArr1d)
            for illum in quartzIllums:
                mask = tArr1d > illum.t0 + illum.tExp
                if not np.any(mask):
                    continue
                q[mask] += comp.charge(tArr1d[mask], illum.t0, illum.tExp, illum.flux) - comp.charge(
                    tArr1d[mask] + tExpDark, illum.t0, illum.tExp, illum.flux
                )
            np.maximum(q, 0.0, out=q)
            result[comp.label] = q.reshape(tArr.shape)
            total += q
        result["total"] = total.reshape(tArr.shape)
        return result

    def fitFractions(
        self,
        tObs: np.ndarray,
        pObs: np.ndarray,
        quartzIllums: Sequence[Illumination],
        tExpDark: float = 300.0,
        dpObs: np.ndarray | None = None,
        fixedIndices: Sequence[int] | None = None,
    ) -> tuple["H4Persistence1DBasicModel", dict]:
        r"""Fit trap fractions to observed persistence data.

        The model persistence is linear in fractions, so the problem reduces
        to bounded linear least squares:

            min  ||A @ f - pObs||^2    subject to   0 <= f_i <= 1

        Normalization convention
        ------------------------
        pObs must be normalized by (\sum_k flux_k * tExp_k) * tExpDark:

            pObs = P_actual [e^-] / (\sum_k flux_k \cdot tExp_k [e^-] * tExpDark [s])

        Parameters
        ----------
        tObs : array-like
            Observation times [s].  Should be > t0 + tExpQuartz.
        pObs : array-like
            Observed persistence normalized by (\sum flux_k \cdot tExp_k) * tExpDark.
        quartzIllums : `Sequence` [`Illumination`]
            List of quartz illumination.
        tExpDark : `float`, optional
            Dark exposure duration [s].  Default: 300.
        dpObs : array-like, optional
            1-sigma errors on ``pObs``.  When given, the fit minimises
            ``((model - pObs) / dpObs)^2``.
        fixedIndices : `Sequence` [`int`], optional
            Indices of components whose fractions are held fixed.

        Returns
        -------
        persistModel : `H4Persistence1DBasicModel`
            New H4Persistence1DBasicModel with fitted fractions; taus and fMids unchanged.
        info : `dict`
            Fitting diagnostics: fractions, fractionErrs, residuals, rms,
            cost, success, solver, dataframe, metadata.
        """
        tArr = np.asarray(tObs, dtype=float).reshape(-1)
        pArr = np.asarray(pObs, dtype=float).reshape(-1)
        norm = sum(illum.flux * illum.tExp for illum in quartzIllums) * tExpDark
        illumMask = Illumination.makeTimeMask(quartzIllums, tArr)

        A = np.zeros((len(tArr), self.nComponents))
        for j, comp in enumerate(self.components):
            unit = TrapComponent(
                fraction=1.0, tau=comp.tau, fMid=comp.fMid, fractionErr=comp.fractionErr
            )
            col = np.zeros(len(tArr))
            for illum in quartzIllums:
                col += unit.charge(tArr, illum.t0, illum.tExp, illum.flux) - unit.charge(
                    tArr + tExpDark, illum.t0, illum.tExp, illum.flux
                )
            col[illumMask] = 0.0
            A[:, j] = col / norm

        fixedSet = set(fixedIndices) if fixedIndices is not None else set()
        freeIdx = [j for j in range(self.nComponents) if j not in fixedSet]

        pAdj = pArr.copy()
        for j in fixedSet:
            pAdj -= A[:, j] * self.components[j].fraction

        if dpObs is not None:
            w = 1.0 / np.asarray(dpObs, dtype=float).reshape(-1)
            AFit = A[:, freeIdx] * w[:, None]
            pFit = pAdj * w
        else:
            AFit = A[:, freeIdx]
            pFit = pAdj

        finiteMask = np.isfinite(pFit) & np.all(np.isfinite(AFit), axis=1)
        if not np.any(finiteMask):
            raise ValueError("No finite data points after weighting. " "Check dpObs for zero or NaN values.")
        ASolve = AFit[finiteMask]
        pSolve = pFit[finiteMask]

        fitted_f_free = None
        solver = ""
        try:
            try:
                result = lsq_linear(ASolve, pSolve, bounds=(0.0, 1.0), method="bvls")
                if np.all(np.isfinite(result.x)):
                    fitted_f_free = result.x
                    solver = "scipy lsq_linear (bvls)"
            except np.linalg.LinAlgError:
                pass
            if fitted_f_free is None:
                result = lsq_linear(
                    ASolve,
                    pSolve,
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
            fitted_f_free = _bounded_lsq_pgd(ASolve, pSolve, lo=0.0, hi=1.0)
            solver = "numpy projected gradient descent"

        fitted_f = np.array([comp.fraction for comp in self.components])
        for k, j in enumerate(freeIdx):
            fitted_f[j] = fitted_f_free[k]

        residuals = A @ fitted_f - pArr
        rms = np.sqrt(np.mean(residuals**2))

        AtA_fit = ASolve.T @ ASolve
        nFree = len(freeIdx)
        try:
            covFree = np.linalg.lstsq(AtA_fit, np.eye(nFree), rcond=None)[0]
        except np.linalg.LinAlgError:
            covFree = np.zeros((nFree, nFree))
        if dpObs is None:
            dof = max(1, len(pArr) - nFree)
            covFree = covFree * (np.dot(residuals, residuals) / dof)
        df_free = np.sqrt(np.maximum(0.0, np.diag(covFree)))

        df = np.zeros(self.nComponents)
        for k, j in enumerate(freeIdx):
            df[j] = df_free[k]

        persistModel = self.updateFractions(
            fractions=fitted_f, fractionErrs=df, name=self.name + " (fitted)"
        )

        result_df = pd.DataFrame(
            {
                "tau": [c.tau for c in persistModel.components],
                "f_mid": [c.fMid for c in persistModel.components],
                "fraction": fitted_f,
                "fractionErr": df,
            }
        )

        info = {
            "fractions": fitted_f,
            "fractionErrs": df,
            "dataframe": result_df,
            "metadata": {"name": self.name},
            "residuals": residuals,
            "rms": rms,
            "cost": float(0.5 * np.dot(residuals, residuals)),
            "success": True,
            "solver": solver,
        }
        return persistModel, info

    def plot(
        self,
        quartzIllums: Sequence[Illumination],
        *,
        tExpDark: float = 300.0,
        tExtra: float = 3600.0,
        nPoints: int = 2000,
        logScale: bool = False,
        tObs: np.ndarray | None = None,
        pObs: np.ndarray | None = None,
        dpObs: np.ndarray | None = None,
        xlim: tuple[float, float] | None = None,
        ylim1: tuple[float, float] | None = None,
        ylim2: tuple[float, float] | None = None,
        show: bool = True,
        savePath: str | None = None,
    ) -> plt.Figure:
        """Plot illumination timeline, trapped charge, and persistence.

        Parameters
        ----------
        quartzIllums : `Sequence` [`Illumination`]
            List of quartz illumination.
        tExpDark : `float`, optional
            Dark exposure duration [s].  Default: 300.
        tExtra : `float`, optional
            Time window after the end of the last illumination [s].  Default: 3600.
            Ignored when ``xlim`` is provided.
        nPoints : `int`, optional
            Number of sample points on the time axis.  Default: 2000.
        logScale : `bool`, optional
            If True, use log scale on both axes of the signal panels.
        tObs, pObs, dpObs : array-like or None
            Observed data to overlay on the persistence panel.
        xlim, ylim1, ylim2 : `tuple` or None
            Axis limits.
        show : `bool`, optional
            Call ``plt.show()`` if True.
        savePath : `str` or None, optional
            Save figure to this path (dpi=150) if given.

        Returns
        -------
        figure : `matplotlib.figure.Figure`
            Figure.
        """
        tStart = min(illum.t0 for illum in quartzIllums)
        tEndLast = max(illum.t0 + illum.tExp for illum in quartzIllums)

        if xlim is not None:
            tPlotStart, tStop = float(xlim[0]), float(xlim[1])
        else:
            tPlotStart, tStop = tStart, tEndLast + tExtra

        if logScale:
            t = np.logspace(np.log10(max(tPlotStart, 1e-3)), np.log10(tStop), nPoints)
        else:
            t = np.linspace(tPlotStart, tStop, nPoints)

        signalData = self.signalComponents(t, quartzIllums)
        persData = self.persistenceComponents(t, quartzIllums, tExpDark)
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

        # --- Panel 0: illumination timeline with tMid markers ---
        for k, illum in enumerate(quartzIllums):
            tEnd_k = illum.t0 + illum.tExp
            label = "Illumination" if k == 0 else None
            ax0.axvspan(illum.t0, tEnd_k, ymin=0.2, ymax=0.8, color="gold", alpha=0.8, label=label)
            ax0.axvline(tEnd_k, color="steelblue", lw=1.5, ls="--")
            # Mark tMid for each component (use mean fMid if they differ)
            mean_fMid = float(np.mean(self.fMids))
            tMid_k = illum.t0 + mean_fMid * illum.tExp
            midLabel = "tMid (mean)" if k == 0 else None
            ax0.axvline(tMid_k, color="darkorange", lw=1.2, ls=":", label=midLabel)
            ax0.text(
                (illum.t0 + tEnd_k) / 2,
                0.50,
                f"#{k+1}  $\\Phi$={illum.flux:.2e}\ntExp={illum.tExp:.1f} s",
                ha="center",
                va="center",
                fontsize=7,
                color="saddlebrown",
                transform=ax0.get_xaxis_transform(),
            )
        ax0.set_yticks([])
        ax0.set_ylabel("Timeline", fontsize=9)
        ax0.set_title(f"Illumination timeline  ({len(quartzIllums)} event(s))", fontsize=10)
        ax0.legend(loc="upper right", fontsize=8)
        ax0.set_ylim(0, 1)

        # --- Panel 1: individual components + total (trapped charge) ---
        for i, comp in enumerate(self.components):
            ax1.plot(
                t,
                signalData[comp.label],
                color=colors[i % 10],
                label=f"{comp.label}  (f={comp.fraction:.4f}, fMid={comp.fMid:.2f})",
            )
        ax1.plot(t, signalData["total"], color="black", lw=2.5, label="Total")
        for illum in quartzIllums:
            ax1.axvline(illum.t0 + illum.tExp, color="steelblue", lw=1.5, ls="--")
            mean_fMid = float(np.mean(self.fMids))
            ax1.axvline(illum.t0 + mean_fMid * illum.tExp, color="darkorange", lw=1.2, ls=":")
        ax1.set_ylabel("Trapped charge [e${}^-$]")
        ax1.set_title("Trapped charge --- components & total")
        ax1.legend(fontsize=8)
        ax1.grid(True, which="both", alpha=0.3)
        if logScale:
            ax1.set_yscale("log")
            ax1.set_xscale("log")

        # --- Panel 2: persistence normalized by \sum(flux \cdot tExp) * tExpDark ---
        norm = sum(illum.flux * illum.tExp for illum in quartzIllums) * tExpDark
        for i, comp in enumerate(self.components):
            ax2.plot(
                t,
                persData[comp.label] / norm,
                color=colors[i % 10],
                label=f"{comp.label}  (f={comp.fraction:.4f}, fMid={comp.fMid:.2f})",
            )
        ax2.plot(t, persData["total"] / norm, color="black", lw=2.5, label="Total")
        if tObs is not None and pObs is not None:
            if dpObs is not None:
                ax2.errorbar(
                    np.asarray(tObs, dtype=float),
                    np.asarray(pObs, dtype=float),
                    yerr=np.asarray(dpObs, dtype=float),
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
                    np.asarray(tObs, dtype=float),
                    np.asarray(pObs, dtype=float),
                    s=15,
                    color="red",
                    zorder=5,
                    label="Observed",
                    alpha=0.7,
                )
        for illum in quartzIllums:
            ax2.axvline(illum.t0 + illum.tExp, color="steelblue", lw=1.5, ls="--")
            mean_fMid = float(np.mean(self.fMids))
            ax2.axvline(illum.t0 + mean_fMid * illum.tExp, color="darkorange", lw=1.2, ls=":")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel(r"Persistence / ($\sum$ flux $\cdot$ tExp $\times$ t_exp,dark)")
        ax2.set_title(f"Persistence  (t_exp,dark = {tExpDark:.0f} s)")
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", alpha=0.3)
        if logScale:
            ax2.set_yscale("log")
            ax2.set_xscale("log")

        if xlim is not None:
            ax0.set_xlim(xlim)
        if ylim1 is not None:
            ax1.set_ylim(ylim1)
        if ylim2 is not None:
            ax2.set_ylim(ylim2)
        plt.tight_layout()

        if savePath:
            fig.savefig(savePath, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def summary(self, quartzIllum: Illumination | None = None) -> None:
        """Print a formatted summary of trap components.

        If quartzIllum is provided, also prints the trapped charge at key times
        relative to the end of illumination.

        Parameters
        ----------
        quartzIllum : `Illumination`, optional
            Quartz illumination.
        """
        print(f"H4Persistence1DBasicModel: {self.name}")
        print(f"  Components   : {self.nComponents}")
        print(f"  Total fraction: {self.totalFraction:.4f}")
        print()

        dfArr = self.fractionErrs
        print(
            f"  {'#':>3}  {'label':<22}  {'fraction':>10}  {'+- error':>10}"
            f"  {'fMid':>6}  {'tau [s]':>10}"
        )
        print("  " + "-" * 74)
        for i, comp in enumerate(self.components):
            print(
                f"  {i:>3}  {comp.label:<22}  {comp.fraction:>10.5f}"
                f"  {dfArr[i]:>10.5f}  {comp.fMid:>6.3f}  {comp.tau:>10.1f}"
            )

        if quartzIllum is not None:
            tEnd = quartzIllum.t0 + quartzIllum.tExp
            total_in = quartzIllum.flux * quartzIllum.tExp
            check_dt = [0.0, quartzIllum.tExp, 10 * quartzIllum.tExp, 100 * quartzIllum.tExp]
            labels = [
                "end of illumination (tEnd)",
                f"+{quartzIllum.tExp:.0f} s after tEnd",
                f"+{10*quartzIllum.tExp:.0f} s after tEnd",
                f"+{100*quartzIllum.tExp:.0f} s after tEnd",
            ]
            print()
            print(
                f"  t0={quartzIllum.t0} s,"
                f" tExpQuartz={quartzIllum.tExp} s,"
                f" flux={quartzIllum.flux:.2e} e-/s"
            )
            print(f"  Total incoming charge = {total_in:.3e} e-")
            print()
            print(f"  {'label':<30} {'time [s]':>10} {'trapped [e-]':>14} {'fraction':>10}")
            print("  " + "-" * 69)
            for dt, lbl in zip(check_dt, labels):
                tc = tEnd + dt
                q = self.signal(tc, [quartzIllum])
                print(f"  {lbl:<30} {tc:>10.1f} {float(q):>14.2f}" f" {float(q)/total_in:>10.2e}")
