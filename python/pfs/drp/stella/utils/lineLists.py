import os.path
import re
import numpy as np
import matplotlib.pyplot as plt

from lsst.utils import getPackageDir
from pfs.drp.stella import ReferenceLine

__all__ = ["readLineListFile", "plotReferenceLines"]


def readLineListFile(lineListFilename, lamps=None, minIntensity=0, flagsToIgnore=0x0):
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

    if not os.path.isabs(lineListFilename):
        full_lineListFilename = os.path.join(getPackageDir("obs_pfs"), "pfs",
                                             "lineLists", lineListFilename)
        if os.path.exists(full_lineListFilename):
            lineListFilename = full_lineListFilename
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
            if (flag & ~flagsToIgnore) != 0:
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

            referenceLines.append(ReferenceLine(species, wavelength=float(lam),
                                                guessedIntensity=intensity,
                                                status=ReferenceLine.Status(flag)))

    if len(referenceLines) == 0:
        raise RuntimeError("You have not selected any lines from %s" % lineListFilename)

    return referenceLines


def plotReferenceLines(referenceLines, what="wavelength", ls='-', alpha=1, color=None, label=None,
                       labelStatus=True, labelLines=False, wavelength=None, spectrum=None,
                       referenceFile=False):
    r"""Plot a set of reference lines using axvline

    \param referenceLines   List of ReferenceLine
    \param what   which field in ReferenceLine to plot
    \param ls Linestyle (default: '-')
    \param alpha Transparency (default: 1)
    \param color Colour (default: None => let matplotlib choose)
    \param label Label for lines (default: None => use "what" or "what status")
    \param labelStatus Include status in labels (default: True)
    \param labelLines Label lines with their ion (default: False)
    \param wavelength Wavelengths array for underlying plot (default: None)
    \param spectrum   Intensity array for underlying plot (default: None)
    \param referenceFile  The lines are from a reference file, so status
                          should be interpreted in that light

    If labelLines is True the lines will be labelled at the top of the plot; if you provide the spectrum
    the labels will appear near the peaks of the lines
    """
    if label == '':
        label = None

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

    if len(plt.gca().get_lines()) > 0:  # they've plotted something already
        xlim = plt.xlim()
    else:
        xlim = None

    labels = {}                         # labels that we've used if labelLines is True
    for rl in referenceLines:
        if referenceFile:
            color, label = {0: ('green', "isolated, good"),
                            1: ('black', "not visible"),
                            2: ('red', "blended"),
                            4: ('blue', "unclassified"),
                            }.get(rl.status, ('cyan', "unknown"))
        else:
            if not (rl.status & rl.Status.FIT):
                color = 'black'
                label = "Bad fit"
            elif (rl.status & rl.Status.RESERVED):
                color = 'blue'
                label = "Reserved"
            elif (rl.status & rl.Status.SATURATED):
                color = 'magenta'
                label = "Saturated"
            elif (rl.status & rl.Status.CR):
                color = 'cyan'
                label = "Cosmic ray"
            elif (rl.status & rl.Status.MISIDENTIFIED):
                color = 'brown'
                label = "Misidentified"
            elif (rl.status & rl.Status.CLIPPED):
                color = 'red'
                label = "Clipped"
            else:
                color = 'green'
                label = "Fit"

        label = maybeSetLabel(label)

        x = getattr(rl, what)
        if not np.isfinite(x):
            continue
        plt.axvline(x, ls=ls, color=color, alpha=alpha, label=label)
        label = None

        if labelLines:
            if xlim is not None and not (xlim[0] < x < xlim[1]):
                continue

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

    plt.xlim(xlim)
