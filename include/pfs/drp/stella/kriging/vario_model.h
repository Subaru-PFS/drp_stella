#ifndef VARIOGRAM_MODEL_HEADER_
#define VARIOGRAM_MODEL_HEADER_

#include <stddef.h>

struct LSFParam
{
        size_t n;
        double *dist;
        double *vario;
        double power;
};

#include "model_sph.h"
#include "model_stable.h"

#endif  //VARIOGRAM_MODEL_HEADER_
