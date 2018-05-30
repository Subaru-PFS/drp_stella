#ifndef PFS_DRP_STELLA_UTILS_CHECKSIZE_H
#define PFS_DRP_STELLA_UTILS_CHECKSIZE_H

#include <sstream>
#include "lsst/pex/exceptions.h"

namespace pfs {
namespace drp {
namespace stella {
namespace utils {

template <typename T>
void checkSize(T gotSize, T expectSize, std::string const& description) {
    if (gotSize != expectSize) {
        std::ostringstream str;
        str << "Size mismatch in " << description << ": got " << gotSize << " but expected " << expectSize;
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, str.str());
    }
}

}}}} // namespace pfs::drp::stella::utils

#endif // include guard