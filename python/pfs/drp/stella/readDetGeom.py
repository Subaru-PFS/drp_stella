#!/usr/bin/env python

import argparse
import lsst.afw.table
import lsst.afw.cameraGeom as camGeom

def main(fitsFileName):

    ampInfoCatalog = lsst.afw.table.AmpInfoCatalog(lsst.afw.table.AmpInfoTable.makeMinimalSchema())

    # Create instance of ampInfoCatalog by reading fitsFileName
    amps = ampInfoCatalog.readFits(fitsFileName)

    for iAmp in range(len(amps)):#range(nAmps):
        ampSchema = amps[iAmp].getSchema()
        for name in ampSchema.getNames():
            print 'amps[',iAmp,']: ',name,': value = ',amps[iAmp].get(name)
        print ' '

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fitsFileName", help="Name of fits file containing the detector geometry")
    args = parser.parse_args()

    main(args.fitsFileName)
