#include "pfs/drp/stella/PSF.h"

namespace pfsDRPStella = pfs::drp::stella;

template<typename ImageT, typename MaskT, typename VarianceT>
pfsDRPStella::PSF<ImageT, MaskT, VarianceT>::PSF(
  unsigned int width,                 ///< number of columns
  unsigned int height                ///< number of rows
) :
_trace(width, height),

