#include "pfs/drp/stella/Lines.h"
namespace pfs { namespace drp { namespace stella {
    PTR(NistLine) NistLine::getPointer(){
        PTR(NistLine) ptr(new NistLine(*this));
        return ptr;
    }

    PTR(NistLineMeas) NistLineMeas::getPointer(){
        PTR(NistLineMeas) ptr(new NistLineMeas(*this));
        return ptr;
    }

    void NistLineMeas::setNistLine(NistLine const& pNistLine){
        nistLine.element = pNistLine.element;
        nistLine.flags = pNistLine.flags;
        nistLine.id = pNistLine.id;
        nistLine.ion = pNistLine.ion;
        nistLine.laboratoryWavelength = pNistLine.laboratoryWavelength;
        nistLine.predictedStrength = pNistLine.predictedStrength;
        nistLine.sources = pNistLine.sources;
    }

    std::vector<PTR(NistLineMeas)> getLinesWithFlags(std::vector<PTR(NistLineMeas)> & lines,
                                                     std::string const& needFlags,
                                                     std::string const& ignoreFlags){
        std::vector<PTR(NistLineMeas)> outputLines(0);
        outputLines.reserve(lines.size());

        /// Check flags for each line in lines
        for (auto it=lines.begin(); it!=lines.end(); ++it){
            bool takeLine = false;

            /// First check for all flags in needFlags
            int i=0;
            while (i < needFlags.length()){
                if ((*it)->flags.find(needFlags[i]) != std::string::npos)///needed flag found
                    takeLine = true;
                else{
                    takeLine = false;
                    break;
                }
                ++i;
            }

            /// If good so far, check for any flag in ignoreFlags
            if (takeLine){
                i=0;
                while (i < ignoreFlags.length()){
                    if ((*it)->flags.find(ignoreFlags[i]) != std::string::npos){///flag to ignore found
                        takeLine = false;
                        break;
                    }
                    ++i;
                }
            }
            if (takeLine)
                outputLines.push_back(*it);
        }
        return outputLines;
    }

    /**
     * @brief Return all NistLineMeas in lines with ID equal to id
     * @param lines : vector of Shared Pointers to NistLineMeas to check for ID id
     * @param id : ID to check for
     * @return vector containing all NistLineMeas in lines with ID equal to id
     */
    std::vector<PTR(NistLineMeas)> getLinesWithID(std::vector<PTR(NistLineMeas)> & lines,
                                                  int id){
        /// Create output vector, initialize to zero length, reserve enough space to hold all lines in lines
        std::vector<PTR(NistLineMeas)> outputLines(0);
        outputLines.reserve(lines.size());

        /// Compare each NistLineMeas's ID to id, and add the line to the output vector if IDs are equal
        for (auto it = lines.begin(); it != lines.end(); ++it){
            if ((*it)->nistLine.id == id)
                outputLines.push_back(*it);
        }
        return outputLines;
    }

    /**
     * @brief Return vector of all indices for lines where ID equals id
     * @param lines : vector of NistLineMeas to check for ID id
     * @param id : ID to check for
     * @return vector<int> containing all indices where lines[index].nistLine.id is equal to id
     */
    std::vector<int> getIndexOfLineWithID(std::vector<PTR(NistLineMeas)> const& lines,
                                          int id){
        /// Create output vector, initialize to zero length, reserve enough space to hold all lines in lines
        std::vector<int> outputIndices(0);
        outputIndices.reserve(lines.size());
        int iLine=0;
        for (auto it = lines.begin(); it != lines.end(); ++it, ++iLine){
            if ((*it)->nistLine.id == id)
                outputIndices.push_back(iLine);
        }
        return outputIndices;
    }

    /**
     * @brief For Spectra::identify to run we now need a vector of NistLineMeas
     *        This function takes a ndarray(wavelength, pixel) as input and
     *        returns a vector of NistLineMeas which can be used for identify
     * @param linesIn : ndarray(shape=(nLines, 2)) with [*,0]: wLen, [*,1]: Pix
     * @return std::vector<PTR(NistLineMeas)> which is mostly empty, only
     *         line.NistLine.predictedWavelength and line.pixelPosPredicted are set
     */
    std::vector<PTR(NistLineMeas)> createLineListFromWLenPix(ndarray::Array<float, 2, 1> const& linesIn){
        /// Create output vector, initialize to zero length, reserve enough space
        /// to hold all lines in lines
        std::vector<PTR(NistLineMeas)> outputLines(0);
        outputLines.reserve(linesIn.getShape()[0]);

        int i = 0;
        for (auto it = linesIn.begin(); it != linesIn.end(); ++it, ++i){
            NistLineMeas lineOut;
            lineOut.nistLine.laboratoryWavelength = *(it->begin());
            lineOut.pixelPosPredicted = *(it->begin()+1);
            lineOut.flags += "g";
            if (linesIn.getShape()[1] > 2){
                lineOut.nistLine.predictedStrength = *(it->begin()+2);
            }
            outputLines.push_back(lineOut.getPointer());
        }
        return outputLines;
    }

    std::vector<PTR(NistLine)> getNistLineVec(){
        std::vector<PTR(NistLine)> outputLines(0);
        return outputLines;
    }

    std::vector<PTR(NistLineMeas)> getNistLineMeasVec(){
        std::vector<PTR(NistLineMeas)> outputLines(0);
        return outputLines;
    }

    void append(std::vector<PTR(NistLineMeas)> & lineListInOut,
                std::vector<PTR(NistLineMeas)> const& lineListIn){
        lineListInOut.insert(lineListInOut.end(),
                             lineListIn.begin(),
                             lineListIn.end());
    }
    void append(std::vector<PTR(NistLine)> & lineListInOut,
                std::vector<PTR(NistLine)> const& lineListIn){
        lineListInOut.insert(lineListInOut.end(),
                             lineListIn.begin(),
                             lineListIn.end());
    }
}}}
