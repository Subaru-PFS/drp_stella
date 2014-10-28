#ifndef MATRIX_EX_HEADER_FILE_
#define MATRIX_EX_HEADER_FILE_

#include <gsl/gsl_matrix.h>

namespace MatrixEx
{
        // inverse matrix
        void matrix_inverse(gsl_matrix *dest, const gsl_matrix *src);
        // pseudo inverse matrix
        void matrix_pseudo_inverse(gsl_matrix *dest, const gsl_matrix *src);
        void pseudo_inverse_right(gsl_matrix *dest, const gsl_matrix *src);
        void pseudo_inverse_left(gsl_matrix *dest, const gsl_matrix *src);
};

#endif //MATRIX_EX_HEADER_FILE_
