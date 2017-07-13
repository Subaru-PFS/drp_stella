/**
  * LSST Data Management System
  *
  * Copyright 2008-2017  AURA/LSST.
  *
  * This product includes software developed by the
  * LSST Project (http://www.lsst.org/).
  *
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the LSST License Statement and
  * the GNU General Public License along with this program.  If not,
  * see <https://www.lsstcorp.org/LegalNotices/>.
  **/
///
///This module describes class for NIST emission lines.
///
///@author Andreas Ritter, Princeton University
///
#ifndef __PFS_DRP_STELLA_LINES_H__
#define __PFS_DRP_STELLA_LINES_H__

#include <vector>
#include "lsst/base.h"
#include "math/CurveFitting.h"
#include "ndarray.h"

using namespace std;
namespace pfs { namespace drp { namespace stella {

    struct NistLine {
    /**
      * A struct to encapsulate spectral lines in the NIST Atomic Spectra Database
      * (https://www.nist.gov/pml/atomic-spectra-database)
      * NIST flags:
      *    *    Intensity is shared by several lines (typically, for multiply classified lines).
      *    :    Observed value given is actually the rounded Ritz value, e.g., Ar IV, lambda = 443.40 nm.
      *    -    Somewhat lower intensity than the value given.
      *    a    Observed in absorption.
      *    b    Band head.
      *    bl   Blended with another line that may affect the wavelength and intensity.
      *    B    Line or feature having large width due to autoionization broadening.
      *    c    Complex line.
      *    d    Diffuse line.
      *    D    Double line.
      *    E    Broad due to overexposure in the quoted reference
      *    f    Forbidden line.
      *    g    Transition involving a level of the ground term.
      *    G    Line position roughly estimated.
      *    H    Very hazy line.
      *    h    Hazy line (same as "diffuse").
      *    hfs  Line has hyperfine structure.
      *    i    Identification uncertain.
      *    j    Wavelength smoothed along isoelectronic sequence.
      *    l    Shaded to longer wavelengths; NB: This may look like a "one" at the end
      *         of the number!
      *    m    Masked by another line (no wavelength measurement).
      *    p    Perturbed by a close line. Both wavelength and intensity may be affected.
      *    q    Asymmetric line.
      *    r    Easily reversed line.
      *    s    Shaded to shorter wavelengths.
      *    t    Tentatively classified line.
      *    u    Unresolved from a close line.
      *    w    Wide line.
      *    x    Extrapolated wavelength
      **/

        std::string element;
        std::string flags;
        int id;
        std::string ion;
        float laboratoryWavelength;
        float predictedStrength;
        std::string sources;

        /**
         * @brief Standard Constructor
         * @return *this
         */
        NistLine():
            element(""),
            flags(""),
            id(0),
            ion(""),
            laboratoryWavelength(0.0),
            predictedStrength(0.0),
            sources("")
            {};

        /**
         * @brief Copy constructor
         * @param nistLine : NistLine to copy using copy constructors of the elements
         * @return *this
         */
        NistLine(NistLine const& nistLine):
            element(nistLine.element),
            flags(nistLine.flags),
            id(nistLine.id),
            ion(nistLine.ion),
            laboratoryWavelength(nistLine.laboratoryWavelength),
            predictedStrength(nistLine.predictedStrength),
            sources(nistLine.sources)
            {}

        /**
         * @brief Destructor
         */
        ~NistLine(){}

        /**
         * @brief Return shared pointer to this
         * @return Shared pointer to this
         */
        PTR(NistLine) getPointer();
    };

    struct NistLineMeas {
    /**
     * A class to encapsulate measurements of NistLines
     *
     * Possible flags:
     * p: measured position more than self.config.maxDistance pixels away from predicted position
     * s: measured strength lower than self.config.minStrength
     * m: gaussFit returned a negative value for the width (sigma)
     * a: gaussFit returned a value for the width (sigma) which is too different from the
     *    average value
     * b: rejected line because it is blended (didn't pass the distance / strength test)
     * i: line rejected by the calibration procedure 'identify' because the Gaussian fit failed
     * f: line rejected by the PolyFit inside 'identify'
     * n: line rejected because NIST flagged it as somehow problematic
     * g: line deemed good and sent to 'identify'
     * h: line held back from the fitting procedure to check the wavelength solution
     * l: measured wavelength too far off
     **/
        math::GaussCoeffs eGaussCoeffsLambda;
        math::GaussCoeffs eGaussCoeffsPixel;
        math::GaussCoeffs gaussCoeffsLambda;
        math::GaussCoeffs gaussCoeffsPixel;
        std::string flags;
        NistLine nistLine;
        float pixelPosPredicted;
        float wavelengthFromPixelPosAndPoly;

        /**
         * @brief Standard Constructor
         * @return *this
         */
        NistLineMeas():
            eGaussCoeffsLambda(),
            eGaussCoeffsPixel(),
            gaussCoeffsLambda(),
            gaussCoeffsPixel(),
            flags(""),
            nistLine(),
            pixelPosPredicted(0.0),
            wavelengthFromPixelPosAndPoly(0.0)
        {}

        /**
         * @brief Copy constructor
         * @param nistLineMeas : NistLineMeas to copy using the copy constructors of the elements
         * @return *this
         */
        NistLineMeas(NistLineMeas const& nistLineMeas):
            eGaussCoeffsLambda(nistLineMeas.eGaussCoeffsLambda),
            eGaussCoeffsPixel(nistLineMeas.eGaussCoeffsPixel),
            gaussCoeffsLambda(nistLineMeas.gaussCoeffsLambda),
            gaussCoeffsPixel(nistLineMeas.gaussCoeffsPixel),
            flags(nistLineMeas.flags),
            nistLine(nistLineMeas.nistLine),
            pixelPosPredicted(nistLineMeas.pixelPosPredicted),
            wavelengthFromPixelPosAndPoly(nistLineMeas.wavelengthFromPixelPosAndPoly)
        {}

        /**
         * @brief Destructor
         */
        ~NistLineMeas(){}

        /**
         * @brief Set this->nistLine to nistLine
         * @param nistLine : use this nistLine
         */
        void setNistLine(NistLine const& nistLine);

        /**
         * @brief Return shared pointer to this
         * @return Shared pointer to this
         */
        PTR(NistLineMeas) getPointer();
    };

    /**
     * @brief Return vector<NistLineMeas> containing all lines where lines[i].flags has all the flags in
     *        needFlags and none of the flags in ignoreFlags
     * @param lines : Vector containing Shared Pointers to NistLineMeas
     * @param needFlags : One string containing all the flags you want
     * @param ignoreFlags : One string containing all the flags you don't want
     * @return vector of NistLineMeas which contain the flags in needFlags and don't contain any of the
     *         flags in ignoreFlags
     */
    std::vector<PTR(NistLineMeas)> getLinesWithFlags(std::vector<PTR(NistLineMeas)> & lines,
                                                     std::string const& needFlags,
                                                     std::string const& ignoreFlags = "");

    /**
     * @brief Return all NistLineMeas in lines with ID equal to id
     * @param lines : vector of Shared Pointers to NistLineMeas to check for ID id
     * @param id : ID to check for
     * @return vector containing all NistLineMeas in lines with ID equal to id
     */
    std::vector<PTR(NistLineMeas)> getLinesWithID(std::vector<PTR(NistLineMeas)> & lines,
                                                  int id);

    /**
     * @brief Return vector of all indices for lines where ID equals id
     * @param lines : vector of NistLineMeas to check for ID id
     * @param id : ID to check for
     * @return vector<int> containing all indices where lines[index].nistLine.id is equal to id
     */
    std::vector<int> getIndexOfLineWithID(std::vector<PTR(NistLineMeas)> const& lines,
                                          int id);

    /**
     * @brief For Spectra::identify to run we now need a vector of NistLineMeas
     *        This function takes a ndarray(wavelength, pixel) as input and
     *        returns a vector of NistLineMeas which can be used for identify
     * @param linesIn : ndarray(shape=(nLines, 2 or 3)) with [*,0]: wLen, [*,1]: Pix, ([*,2]: strength)
     * @return std::vector<PTR(NistLineMeas)> which is mostly empty, only
     *         line.NistLine.laboratoryWavelength, line.pixelPosPredicted, and possibly
     *         line.NistLine.predictedStrength are set
     */
    std::vector<PTR(NistLineMeas)> createLineListFromWLenPix(ndarray::Array<float, 2, 1> const& linesIn);

    /**
     * @brief Return a vector<PTR(NistLine)> of size 0 for Python to be able
     *        to call C++ functions which take such a vector as input
     * @return vector<PTR(NistLine)>(0)
     */
    std::vector<PTR(NistLine)> getNistLineVec();

    /**
     * @brief Return a vector<PTR(NistLineMeas)> of size 0 for Python to be able
     *        to call C++ functions which take such a vector as input
     * @return vector<PTR(NistLineMeas)>(0)
     */
    std::vector<PTR(NistLineMeas)> getNistLineMeasVec();

    /**
     * @brief append one line list to another
     * @param lineListInOut : line list to append the 2nd one to
     * @param lineListIn : line list to append to the 1st one
     */
    void append(std::vector<PTR(NistLineMeas)> & lineListInOut,
                std::vector<PTR(NistLineMeas)> const& lineListIn);
    void append(std::vector<PTR(NistLine)> & lineListInOut,
                std::vector<PTR(NistLine)> const& lineListIn);
/*
    template<typename T>
    void append(std::vector<PTR(T)> & lineListInOut,
                std::vector<PTR(T)> const& lineListIn);
 */
}}}
#endif
