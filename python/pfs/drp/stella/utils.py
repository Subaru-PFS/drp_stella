import re

import numpy as np
from astropy.io import fits as pyfits

from pfs.drp.stella import ReferenceLine

__all__ = ["readWavelengthFile", "readLineListFile", "plotReferenceLines"]


def readWavelengthFile(wLenFile):
    """read wavelength file and return 1-D arrays of length nFibre*nwavelength

    These arrays are used by evaluating e.g. wavelengths[np.where(traceId == fid)]
    """
    hdulist = pyfits.open(wLenFile)
    tbdata = hdulist[1].data
    traceIds = tbdata[:]['fiberNum'].astype('int32')
    wavelengths = tbdata[:]['pixelWave'].astype('float32')
    xCenters = tbdata[:]['xc'].astype('float32')

    traceIdSet = np.unique(traceIds)
    assert len(wavelengths) == len(traceIds[traceIds == traceIdSet[0]])*len(traceIdSet)  # could check all

    return [xCenters, wavelengths, traceIds]


def readLineListFile(lineListFilename, lamps=["Ar", "Cd", "Hg", "Ne", "Xe"], minIntensity=0):
    """Read line list

    File consists of lines of
       lambda Intensity Species Flag
    where lambda is the wavelength in vacuum nm

    Flag:  bitwise OR of:
      0   Good
      1   Not visible
      2   Blend; don't use
      4   Unknown; check

    Return:
       list of drp::ReferenceLine
    """
    #
    # Pack into a list of ReferenceLines
    #
    referenceLines = []

    with open(lineListFilename) as fd:
        for line in fd:
            line = re.sub(r"\s*#.*$", "", line).rstrip()  # strip comments

            if not line:
                continue
            fields = line.split()
            try:
                lam, intensity, species, flag = fields
            except Exception as e:
                print("%s: %s" % (e, fields))
                raise

            flag = int(flag)
            if flag != 0:
                continue

            try:
                intensity = float(intensity)
            except ValueError:
                intensity = np.nan

            if lamps:
                keep = False
                for lamp in lamps:
                    if species.startswith(lamp):
                        keep = True
                        break

                if not keep:
                    continue

            if minIntensity > 0:
                if not np.isfinite(intensity) or intensity < minIntensity:
                    continue

            referenceLines.append(ReferenceLine(species, wavelength=float(lam), guessedIntensity=intensity))

    if len(referenceLines) == 0:
        raise RuntimeError("You have not selected any lines from %s" % lineListFilename)

    return referenceLines


def plotReferenceLines(referenceLines, what, ls=':', alpha=1, color=None, label=None, labelStatus=True,
                       labelLines=False, wavelength=None, spectrum=None):
    r"""Plot a set of reference lines using axvline

    \param referenceLines   List of ReferenceLine
    \param what   which field in ReferenceLine to plot
    \param ls Linestyle (default: ':')
    \param alpha Transparency (default: 1)
    \param color Colour (default: None => let matplotlib choose)
    \param label Label for lines (default: None => use "what" or "what status")
    \param labelStatus Include status in labels (default: True)
    \param labelLines Label lines with their ion (default: False)
    \param wavelength Wavelengths array for underlying plot (default: None)
    \param spectrum   Intensity array for underlying plot (default: None)

    If labelLines is True the lines will be labelled at the top of the plot; if you provide the spectrum
    the labels will appear near the peaks of the lines
    """
    if label == '':
        label = None

    import matplotlib.pyplot as plt

    def maybeSetLabel(status):
        if labelLines:
            if labelStatus:
                lab = "%s %s" % (what, status)  # n.b. label is in caller's scope
            else:
                lab = what

            lab = None if lab in labels else lab
            labels[lab] = True

            return lab
        else:
            return label

    labels = {}                         # labels that we've used if labelLines is True
    for rl in referenceLines:
        if not (rl.status & rl.Status.FIT):
            color = 'black'
            label = maybeSetLabel("Bad fit")
        elif (rl.status & rl.Status.RESERVED):
            color = 'blue'
            label = maybeSetLabel("Reserved")
        elif (rl.status & rl.Status.SATURATED):
            color = 'magenta'
            label = maybeSetLabel("Saturated")
        elif (rl.status & rl.Status.CR):
            color = 'cyan'
            label = maybeSetLabel("Cosmic ray")
        elif (rl.status & rl.Status.MISIDENTIFIED):
            color = 'brown'
            label = maybeSetLabel("Misidentified")
        elif (rl.status & rl.Status.CLIPPED):
            color = 'red'
            label = maybeSetLabel("Clipped")
        else:
            color = 'green'
            label = maybeSetLabel("Fit")

        x = getattr(rl, what)
        if not np.isfinite(x):
            continue
        plt.axvline(x, ls=ls, color=color, alpha=alpha, label=label)
        label = None

        if labelLines:
            if spectrum is None:
                y = 0.95*plt.ylim()[1]
            else:
                if wavelength is None:
                    ix = x
                else:
                    ix = np.searchsorted(wavelength, x)

                    if ix == 0 or ix == len(wavelength):
                        continue

                i0 = max(0, int(ix) - 2)
                i1 = min(len(spectrum), int(ix) + 2 + 1)
                y = 1.05*spectrum[i0:i1].max()

            plt.text(x, y, rl.description, ha='center')
