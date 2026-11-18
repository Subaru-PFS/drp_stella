"""Claude: PIPE2D-1993: unmasked spikes in the fluxCal (ConstantFocalPlaneFunction) of the regression visits

Spikes are pixels more than 50% off a 9-pixel median (PIPE2D-2005).
"""
import os
import numpy as np
from scipy.ndimage import median_filter
from lsst.daf.butler import Butler

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)
b = Butler(REPO, collections="u/kiyoyabe/weekly_hilo/w.2026.04")
for visit in (122783, 122784, 122785, 122786, 122771):
    try:
        fc = b.get("fluxCal", instrument="PFS", visit=visit).constantFocalPlaneFunction
    except Exception as exc:
        print(visit, "no fluxCal", type(exc).__name__)
        continue
    v = np.where(fc.mask, np.nan, fc.value.astype(float))
    good = np.isfinite(v)
    med = median_filter(np.where(good, v, np.nanmedian(v)), 9)
    ratio = v/med
    bad = np.flatnonzero(good & (fc.wavelength > 400) & (np.abs(ratio - 1) > 0.5))
    print(visit, "unmasked fluxCal pixels >50% off a 9-pixel median (lambda>400nm):", len(bad),
          "; masked:", int(np.sum(~good)))
    for ii in bad[:12]:
        print("   %.3f nm ratio %.3g" % (fc.wavelength[ii], ratio[ii]))
