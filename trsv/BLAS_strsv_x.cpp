#include <math.h>
#include "blas_extended.h"
#include "blas_extended_private.h"
#include "trsv.hpp"

void BLAS_strsv_x(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  enum blas_trans_type trans,
                  enum blas_diag_type diag,
                  int n,
                  float alpha,
                  const float *T,
                  int ldt,
                  float *x,
                  int incx,
                  enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine solve :
 *
 *     x <- alpha * inverse(T) * x
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        column major, row major
 *
 * uplo   (input) enum blas_uplo_type
 *        upper, lower
 *
 * trans  (input) enum blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) enum blas_diag_type
 *        unit, non unit
 *
 * n      (input) int
 *        the dimension of T
 *
 * alpha  (input) float
 *
 * T      (input) const float*
 *        Triangular matrix
 *
 * x      (input/output) float*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 * prec   (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 */
{
//static const char *routine_name = "BLAS_strsv_x";
  XBLAS::trsv_x(order, uplo, trans, diag, n, alpha, T, ldt, x, incx, prec);
} /* end BLAS_strsv_x */
