import numpy as np
import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import FiberStatus, TargetType


__all__ = ["plot2dSpectrumSlice"]


def plot2dSpectrumSlice(exposure, pfsConfig, detectorMap, title="", r0=1980, r1=2020, 
                       nrows=3, ncols=2, overlap=50):
    """Plot a slice through all the columns of a 2-D spectrum in 
    nrows x ncols panels
    
    exposure : `lsst.afw.image.Image` 2-D spectrum
    pfsConfig: FfsConfig for exposure
    detectorMap: DetectorMap for exposure
    r0, r1: `int`  Use mean of flux in rows r0..r1
    nrows: `int` Number of subplots per row (passed to plt.subplots)
    ncols: `int` Number of subplots per column  (passed to plt.subplots)
    overlap: `int` how much to overlap the pixel range in the panels
    """
    fig, axs = plt.subplots(nrows, ncols, sharey=True)
    axs = axs.flatten()
    n = len(axs)   # number of panels
    xlen = exposure.getWidth()//n + 1
    
    for i in range(n):
        plt.axes(axs[i])
        axs[i].label_outer()

        plt.plot(np.median(exposure.image.array[r0:r1 + 1, :], axis=0))

        for fid in detectorMap.fiberId:
            if fid not in pfsConfig.fiberId:
                continue

            xc = detectorMap.getXCenter(fid)[(r0 + r1)//2]
            ind = pfsConfig.selectFiber(fid)[0]
            color = 'red' if pfsConfig.targetType[ind] == TargetType.SUNSS_DIFFUSE else 'green'
            ls = ':' if pfsConfig.fiberStatus[ind] == FiberStatus.BROKENFIBER else '-'

            plt.axvline(xc, ls=ls, color=color, alpha=0.25, zorder=-1)

        x01 = np.array([i*xlen - overlap, (i + 1)*xlen + overlap])
        if i == 0:
            x01 += overlap
        elif i == n - 1:
            x01 -= overlap
        plt.xlim(*x01)

        plt.axhline(0, ls='-', color='black', alpha=0.5)

    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel('Column', labelpad=10) # Use argument `labelpad` to move label downwards.
    ax.set_ylabel('Mean flux', labelpad=20)
    plt.suptitle(title + f"  rows {r0}..{r1}", y=1.0)


    plt.tight_layout()
    
    return fig
