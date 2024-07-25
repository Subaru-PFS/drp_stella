#!/usr/bin/env python

import os
from argparse import ArgumentParser
from itertools import product

from lsst.afw.image import ExposureF, VisitInfo
from pfs.drp.stella.calibs import setCalibHeader


def makeFakeFlat(arm: str, spectrograph: int) -> ExposureF:
    """Make a fake flat for a given arm and spectrograph

    Parameters
    ----------
    arm : `str`
        Spectrograph arm (b, r, m, n).
    spectrograph : `int`
        Spectrograph number (1-4).

    Returns
    -------
    flat : `lsst.afw.image.ExposureF`
        Fake flat.
    """
    width = 4096
    if arm == "n":
        height = 4096
    elif arm in "brm":
        height = 4176
    else:
        raise ValueError(f"Unrecognized arm {arm}")

    flat = ExposureF(width, height)
    flat.image.array[:] = 1.0
    flat.mask.array[:] = 0
    flat.variance.array[:] = 0.0

    header = flat.getMetadata()
    dataId = dict(
        visit0=0,
        arm=arm,
        ccd=3*(spectrograph - 1) + dict(b=0, r=1, m=1, n=2)[arm],
        spectrograph=spectrograph,
        filter=arm,
        calibDate="2000-01-01",
        calibTime="2000-01-01T00:00:00.0",
    )
    setCalibHeader(header, "flat", [0], dataId)
    header.set("ARM", arm)
    header.set("SPECTROGRAPH", spectrograph)
    header.set("CALIB_INPUT_0", "FAKE")

    flat.getInfo().setVisitInfo(VisitInfo(exposureTime=1.0, darkTime=1.0))

    return flat


def main(path: str = ".", doNir: bool = False):
    """Main function to make fake flats

    Parameters
    ----------
    path : `str`, optional
        Path to write fake flats.
    doNir : `bool`, optional
        Construct NIR flats?
    """
    armList = "brm"
    if doNir:
        armList += "n"
    for arm, spectrograph in product(armList, range(1, 5)):
        flat = makeFakeFlat(arm, spectrograph)
        filename = os.path.join(path, f"pfsFakeFlat-{arm}{spectrograph}.fits")
        flat.writeFits(filename)
        print(f"Wrote fake flat for {arm}{spectrograph} as {filename}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", nargs="?", default=".", help="Path to write fake flats")
    parser.add_argument("--doNir", action="store_true", help="Construct NIR flats?")
    args = parser.parse_args()
    main(**vars(args))
