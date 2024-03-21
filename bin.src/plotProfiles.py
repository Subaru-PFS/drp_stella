#!/usr/bin/env python

import os
from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
plt.switch_backend("agg")
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import medfilt  # noqa: E402

from pfs.drp.stella import FiberProfileSet  # noqa: E402


def plotProfiles(directory: str, output: str):
    """Plot fiber profiles

    Parameters
    ----------
    directory : `str`
        Directory containing fiber profiles.
    output : `str`
        Output PDF filename.
    """
    filenames = glob(f"{directory}/pfsFiberProfiles-*.fits")
    with PdfPages(output) as pdf:
        for fn in sorted(filenames):
            print(f"Reading {fn}")
            profiles = FiberProfileSet.readFits(fn)
            fn = os.path.basename(fn)

            fig, axes = profiles.plotHistograms(show=False)
            fig.suptitle(f"{fn} profile stats")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            figAxes = profiles.plot(show=False)
            for ff in profiles:
                fig, axes = figAxes[ff]
                fig.suptitle(f"Fiber profiles for {fn}")
                axes.semilogy()
                axes.set_ylim(3e-4, 5e-1)

            figures = {id(fig): fig for fig, _ in figAxes.values()}
            for ff in profiles:
                fig, axes = figAxes[ff]
                if id(fig) in figures:
                    pdf.savefig(fig)
                    del figures[id(fig)]
                    plt.close(fig)

            haveProfiles = False
            fig, axes = plt.subplots()
            for ff in profiles:
                if profiles[ff].norm is None:
                    continue
                axes.plot(profiles[ff].norm, label=str(ff))
                haveProfiles = True
            if haveProfiles:
                axes.set_xlabel("Row (pixels)")
                axes.set_ylabel("Flux (electrons)")
                axes.set_title(f"{fn} normalization")
                axes.semilogy()
                top = np.max([np.max(medfilt(np.nan_to_num(profiles[ff].norm), 15)) for ff in profiles])
                axes.set_ylim(1, 2*top)
                pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("directory", help="Directory containing fiber profiles")
    parser.add_argument("output", help="Output PDF file")
    args = parser.parse_args()

    plotProfiles(args.directory, args.output)
