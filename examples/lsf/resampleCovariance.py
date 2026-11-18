"""Claude: resampling white noise

The beating of the diagonal variance, and the information (1^T C^-1 1) with the full covariance vs
diagonal-only weights.
"""
import numpy as np
from pfs.drp.stella.interpolate import Resampler
n = 400
for name, dIn in (("b-like (downsample 0.066->0.08)", 0.066), ("r-like (upsample 0.083->0.08)", 0.083)):
    inWl = 600 + dIn*np.arange(n)
    outWl = np.arange(602, inWl[-1] - 2, 0.08)
    for order in (1, 3):
        rs = Resampler(inWl, outWl, order=order)
        # Full resampling matrix R (columns = resampled unit vectors); C = R R^T for unit white noise
        R = np.array([rs.resampleFlux(np.eye(n)[k]) for k in range(n)]).T
        C = R @ R.T
        diag = np.diag(C)
        ev = np.linalg.eigvalsh(C)
        # Information about a constant: 1^T C^-1 1 per pixel (GLS) vs sum of diagonal weights
        one = np.ones(len(outWl))
        gls = one @ np.linalg.lstsq(C, one, rcond=1e-12)[0]
        print(f"{name} order {order}: diag var min {diag.min():.3f} max {diag.max():.3f}; "
              f"cond(C) {ev.max()/max(ev.min(), 1e-300):.1e}; "
              f"info per input pixel: GLS {gls/n:.3f}, diagonal-only weights {np.sum(1/diag)/n:.3f}")
