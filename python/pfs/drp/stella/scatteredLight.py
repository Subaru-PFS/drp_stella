import numpy as np
import scipy.signal as signal

import lsst.afw.image as afwImage
from pfs.drp.stella import FiberTrace, FiberTraceSet, SpectrumSet

__all__ = ["estimateScatteredLight"]


def estimateScatteredLight(pfsArm, detMap, log=None,
                           A=1, alpha=-1.5, a=1,
                           Bratio=0.2, beta=-3, b=5,
                           hsize=4096, gap=69, kernels=None, models=None):
    """
Make a model of the image, using 1-pixel wide traces and the real extracted spectra

A = 1      # total power in the first component of the scattering kernel
alpha = -1.5  # index (2-D)
a = 1      # softening (pixels)
Bratio = 0.2      # ratio of total power in the second component of the scattering kernel to that in the first
beta = -3  # index (2-D)
b = 5      # softening (pixels)
hsize=4096  # kernel is (2*hsize + 1)*(2*hsize + 1)
gap=69     # gap between the active areas of the CCDs
    kernels:  array to return the components and total kernel
    models:   array to return the model spectrum used
"""
    if detMap is None:
        return None

    if log:
        log.info("Estimating scattered light")

    dims = detMap.getBBox().getDimensions()

    boxcarWidth = 1

    traces = FiberTraceSet(len(pfsArm))
    for fid in pfsArm.fiberId:
        centers = detMap.getXCenter(fid)
        traces.add(FiberTrace.boxcar(fid, dims, boxcarWidth/2, centers))

    spectra = SpectrumSet.fromPfsArm(pfsArm)
    model = spectra.makeImage(dims, traces).array

    if models is not None:
        models.append(model)

    # Construct a kernel of the form
    #  1/(r^2 + a^2)^(alpha/2)

    dX, dY = np.meshgrid(np.arange(-hsize, hsize + 1), np.arange(-hsize, hsize + 1))

    weights = 0

    weight = ((dX**2 + dY**2) + a*a)**(alpha/2)
    weight *= A/(1 + Bratio)/np.sum(weight)
    if kernels is not None:
        kernels.append(weight)

    weights += weight

    if Bratio != 0:
        weight = ((dX**2 + dY**2) + b*b)**(beta/2)
        weight *= A*Bratio/(1 + Bratio)/np.sum(weight)
        if kernels is not None:
            kernels.append(weight)

        weights += weight

    if kernels is not None:
        kernels.append(weights)
    #
    # Allow for the fact that there's a physical gap of c. 1040microns between the two CCDs in the optical
    #
    if gap > 0:
        height, width = model.shape
        _model = np.zeros((height, width + gap))
        _model[:, 0:width//2] = model[:, 0:width//2]
        _model[:, -width//2:] = model[:, -width//2:]
        model = _model
    #
    # Convolve;  zero-padded but that's OK as we're using a model with a background of zero
    #
    smodel = signal.convolve(model, weights, mode='same')

    if gap > 0:
        _model = np.zeros((height, width))
        _model[:, 0:width//2] = smodel[:, 0:width//2]
        _model[:, -width//2:] = smodel[:, -width//2:]
        smodel = _model

    smodel = afwImage.ImageF(smodel.astype(np.float32))

    return smodel
