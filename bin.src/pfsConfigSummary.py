#!/usr/bin/env python

from argparse import ArgumentParser
from collections import Counter
from typing import List, Optional

from lsst.daf.persistence import Butler, NoResults
from pfs.datamodel import PfsConfig, FiberStatus, TargetType
from pfs.drp.stella.utils.visits import parseIntegerList


def pfsConfigSummary(pfsConfig: PfsConfig, spectrograph: Optional[List[int]] = None) -> str:
    """Summarize a PfsConfig

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig`
        PfsConfig to summarize.

    Returns
    -------
    summary : `str`
        Summary of PfsConfig.
    """
    summary = f"visit={pfsConfig.visit} pfsDesignId=0x{pfsConfig.pfsDesignId:016x}"
    if spectrograph:
        pfsConfig = pfsConfig.select(spectrograph=spectrograph)
        summary += " spectrograph=" + ",".join(str(ss) for ss in sorted(spectrograph))
    summary += ":"

    fiberStatus = Counter(pfsConfig.fiberStatus)
    summary += " fiberStatus={"
    summary += ", ".join(f"{FiberStatus(fs).name}={fiberStatus[fs]}" for fs in sorted(fiberStatus))
    summary += "}"

    targetType = Counter(pfsConfig.targetType)
    summary += " targetType={"
    summary += ", ".join(f"{TargetType(tt).name}={targetType[tt]}" for tt in sorted(targetType))
    summary += "}"

    return summary


def main():
    parser = ArgumentParser()
    parser.add_argument("butler", help="Butler root")
    parser.add_argument("visits", type=str, help="Visit(s) to summarize (LSST Gen2 id syntax)")
    parser.add_argument("--spectrograph", type=int, nargs="+", help="Spectrograph(s) to summarize")
    args = parser.parse_args()

    butler = Butler(args.butler)
    visits = parseIntegerList(args.visits)
    for vv in visits:
        try:
            pfsConfig = butler.get("pfsConfig", visit=vv)
        except NoResults:
            print(f"visit={vv} : No pfsConfig found")
        else:
            print(pfsConfigSummary(pfsConfig, args.spectrograph))


if __name__ == "__main__":
    main()
