#!/usr/bin/env python

from pfs.datamodel.identity import Identity
from pfs.datamodel.pfsConfig import PfsConfig
from pfs.drp.stella import PfsArm


def plotPfsArm(dirName, visit, arm, spectrograph, fiberId=None):
    """Plot a PfsArm from disk

    Parameters
    ----------
    dirName : `str`
        Name of directory containing the data.
    visit : `int`
        Visit identifier.
    arm : `str`
        Spectrograph arm (one of ``brnm``).
    spectrograph : `int`
        Spectrograph module number.
    fiberId : `int`, optional
        Fiber identifier to plot.

    Returns
    -------
    figure : `matplotlib.Figure`
        Figure containing the plot.
    axes : `matplotlib.Axes`
        Axes containing the plot.
    """
    identity = Identity(visit=visit, arm=arm, spectrograph=spectrograph)
    arm = PfsArm.read(identity, dirName=dirName)
    return arm.plot(fiberId=fiberId, show=False)


def selectFiber(dirName, pfsDesignId, visit, catId, tractId, patch, objId):
    """Select a fiber by its target

    Parameters
    ----------
    dirName : `str`
        Name of directory containing the data.
    pfsDesignId : `int`
        Identifier for top-end design.
    visit : `int`
        Visit for top-end design.
    catId : `int`
        Catalog identifier.
    tractId : `int`
        Tract identifier.
    patch : `str`
        Patch identifier (typically two integers separated by a comma).
    objId : `int`
        Object identifier.

    Returns
    -------
    fiberId : `int`
        Fiber identifier.
    """
    config = PfsConfig.read(pfsDesignId, visit, dirName=dirName)
    index = config.selectTarget(catId, tractId, patch, objId)
    return config.fiberId[index]


def integer(xx):
    """Create an integer using auto base detection"""
    return int(xx, 0)


def main():
    from argparse import ArgumentParser
    epilog = """To plot a specific fiber, you can either specify the fiber directly using
'--fiberId' or select a fiber by its target by specifying all of
'--pfsDesignId', '--visit', '--catId', '--tract', '--patch', and '--objId'.
"""
    parser = ArgumentParser(description="Plot PfsArm", epilog=epilog)
    parser.add_argument("--visit", type=int, required=True, help="Visit number")
    parser.add_argument("--arm", choices=["b", "r", "n", "m"], required=True, help="Spectrograph arm")
    parser.add_argument("--spectrograph", type=int, required=True, help="Spectrograph module number")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing data")

    # To select a single fiber, specify either:
    parser.add_argument("--fiberId", type=int, help="Desired fiber")
    # Or ALL of the following:
    parser.add_argument("--pfsDesignId", type=integer, help="Identifier for top-end design")
    parser.add_argument("--catId", type=int, help="Desired catalog identifier")
    parser.add_argument("--tract", type=int, help="Desired tract")
    parser.add_argument("--patch", type=str, help="Desired patch")
    parser.add_argument("--objId", type=integer, help="Desired object identifier")

    args = parser.parse_args()
    fiberId = None
    if args.fiberId is not None:
        if (args.pfsDesignId is not None or
                args.catId is not None or
                args.tract is not None or
                args.patch is not None or
                args.objId is not None):
            import warnings
            warnings.warn("Ignoring provided pfsDesignId/visit0/catId/tract/patch/objId "
                          "because fiberId was provided")
        fiberId = [args.fiberId]
    elif (args.pfsDesignId is not None and
          args.catId is not None and
          args.tract is not None and
          args.patch is not None and
          args.objId is not None):
        fiberId = [selectFiber(args.dir, args.pfsDesignId, args.visit,
                               args.catId, args.tract, args.patch, args.objId)]

    fig, axes = plotPfsArm(args.dir, args.visit, args.arm, args.spectrograph, fiberId)
    # pyplot takes care of keeping the plot window open
    import matplotlib.pyplot as plt
    plt.show(block=True)
    return fig, axes


if __name__ == "__main__":
    main()
