#!/usr/bin/env python

"""
Executable script to create a PfsDesign for a LAM exposure, given the colors
of fibers used.
"""

import numpy as np
from pfs.datamodel import PfsDesign, TargetType

# Mapping of colors to fiberIds
# Constructed from a snippet by Fabrice Madec, "dummy cable B fibers"
# https://sumire-pfs.slack.com/files/U3MLENNHH/FFS6P4UR5/dummy_cable_b_fibers.txt
FIBER_COLORS = {"red1": [2],
                "red2": [3],
                "red3": [308],
                "red4": [339],
                "red5": [340],
                "red6": [342],
                "red7": [649],
                "red8": [650],
                "orange": [12, 60, 110, 161, 210, 259, 341],
                "blue": [32, 111, 223, 289, 418, 518, 620],
                "green": [63, 192, 255, 401, 464, 525, 587],
                "yellow": [347, 400, 449, 545, 593, 641],
                }

# Mapping of colors to hash value
# This scheme makes the hash look like binary, with a 1 if the color was used and 0 if not
HASH_COLORS = {color: 16**ii for ii, color in enumerate(sorted(FIBER_COLORS.keys()))}


def makePfsDesign(pfsDesignId, fiberId):
    """Build a ``PfsConfig``

    Parameters
    ----------
    pfsDesignId : `int`
        Identifier for the top-end design. For our purposes, this is just a
        unique integer.
    fiberId : `numpy.ndarray` of `int`
        Array of identifiers for fibers that will be lit.

    Returns
    -------
    config : `pfs.datamodel.PfsConfig`
        Configuration of the top-end.
    """
    raBoresight = 0.0
    decBoresight = 0.0
    tract = np.zeros_like(fiberId, dtype=int)
    patch = ["0,0" for _ in fiberId]

    num = len(fiberId)
    catId = np.zeros_like(fiberId, dtype=int)
    objId = fiberId
    targetTypes = TargetType.SCIENCE*np.ones_like(fiberId, dtype=int)
    ra = np.zeros_like(fiberId, dtype=float)
    dec = np.zeros_like(fiberId, dtype=float)
    pfiNominal = np.zeros((num, 2), dtype=float)

    fiberMags = [[] for _ in fiberId]
    filterNames = [[] for _ in fiberId]

    return PfsDesign(pfsDesignId, raBoresight, decBoresight,
                     fiberId, tract, patch, ra, dec, catId, objId, targetTypes,
                     fiberMags, filterNames, pfiNominal)


def colorsToFibers(colors):
    """Convert a list of colors to an array of fiber IDs

    Parameters
    ----------
    colors : iterable of `str`
        List of colors.

    Returns
    -------
    fiberId : `numpy.ndarray`
        Array of fiber IDs.
    """
    return np.array(sorted(set(sum([FIBER_COLORS[col] for col in colors], []))))


def hashColors(colors):
    """Convert a list of colors to a hash for the pfsDesignId

    Parameters
    ----------
    colors : iterable of `str`
        List of colors.

    Returns
    -------
    hash : `int`
        Hash, for the pfsDesignId.
    """
    return sum(HASH_COLORS[col] for col in set(colors))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a PfsDesign for a LAM exposure, "
                                                 "given the colors of fibers used.")
    parser.add_argument("--directory", default=".", help="Directory in which to write file")
    parser.add_argument("colors", nargs="+", type=str, choices=FIBER_COLORS.keys(),
                        help="Color(s) specifying fibers that were lit")
    args = parser.parse_args()

    fiberId = colorsToFibers(args.colors)
    pfsDesignId = hashColors(args.colors)
    config = makePfsDesign(pfsDesignId, fiberId)
    config.write(dirName=args.directory)
    print("Wrote %s" % (config.filename,))
