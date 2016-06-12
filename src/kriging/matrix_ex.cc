#include "pfs/drp/stella/kriging/matrix_ex.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

void MatrixEx::matrix_pseudo_inverse( gsl_matrix *dest, const gsl_matrix *src )
{
        if (src->size1 < src->size2)
                pseudo_inverse_right(dest, src);
        else if (src->size1 > src->size2)
                pseudo_inverse_left(dest, src);
        else
                matrix_inverse(dest, src);
}

void MatrixEx::pseudo_inverse_right(gsl_matrix *dest, const gsl_matrix *src)
{
        gsl_matrix *trans, *prod, *inv;

        trans = gsl_matrix_alloc(src->size2, src->size1);
        prod = gsl_matrix_alloc(src->size1, src->size1);
        inv = gsl_matrix_alloc(src->size1, src->size1);

        gsl_matrix_transpose_memcpy(trans, src);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, src, trans, 0.0, prod);
        matrix_inverse(inv, prod);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, trans, inv, 0.0, dest);

        gsl_matrix_free(trans);
        gsl_matrix_free(prod);
        gsl_matrix_free(inv);
}

void MatrixEx::pseudo_inverse_left(gsl_matrix *dest, const gsl_matrix *src)
{
        gsl_matrix *trans, *prod, *inv;

        trans = gsl_matrix_alloc(src->size2, src->size1);
        prod = gsl_matrix_alloc(src->size2, src->size2);
        inv = gsl_matrix_alloc(src->size2, src->size2);

        gsl_matrix_transpose_memcpy(trans, src);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, trans, src, 0.0, prod);
        matrix_inverse(inv, prod);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inv, trans, 0.0, dest);

        gsl_matrix_free(trans);
        gsl_matrix_free(prod);
        gsl_matrix_free(inv);
}

void MatrixEx::matrix_inverse(gsl_matrix *dest, const gsl_matrix *src)
{
        gsl_matrix *tmp;
        int signum;
        gsl_permutation *p = gsl_permutation_alloc(src->size1);
        tmp = gsl_matrix_alloc(src->size1, src->size2);
        gsl_matrix_memcpy(tmp, src);

        gsl_linalg_LU_decomp(tmp, p, &signum);
        gsl_linalg_LU_invert(tmp, p, dest);

        gsl_permutation_free(p);
        gsl_matrix_free(tmp);
}
