#include "blas_extended.h"
#include "blas_extended_private.h"
#include "spmv/XBLAS_spmv.hpp"

void BLAS_dspmv_s_s(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    int n,
                    double alpha,
                    const float *ap,
                    const float *x,
                    int incx,
                    double beta,
                    double *y,
                    int incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * ap * x + beta * y, where ap is a symmetric
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of ap; row or column major
 *
 * uplo   (input) enum blas_uplo_type
 *        Whether ap is upper or lower
 *
 * n      (input) int
 *        Dimension of ap and the length of vector x
 *
 * alpha  (input) double
 *
 * ap     (input) const float*
 *
 * x      (input) const float*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) double
 *
 * y      (input/output) double*
 *
 * incy   (input) int
 *        The stride for vector y.
 *
 */
{
//static const char *routine_name = "BLAS_dspmv_s_s";
  XBLAS::spmv(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
} /* end BLAS_dspmv_s_s */
