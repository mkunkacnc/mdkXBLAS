#include "blas_extended.h"
#include "blas_extended_private.h"
#include "trmv/XBLAS_trmv.hpp"

void BLAS_dtrmv_s(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  enum blas_trans_type trans,
                  enum blas_diag_type diag,
                  int n,
                  double alpha,
                  const float *T,
                  int ldt,
                  double *x,
                  int incx)
/*
 * Purpose
 * =======
 *
 * Computes x <-- alpha * T * x, where T is a triangular matrix.
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
 * alpha  (input) double
 *
 * T      (input) const float*
 *        Triangular matrix
 *
 * ldt    (input) int
 *        Leading dimension of T
 *
 * x      (input/output) double*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 */
{
//static const char routine_name[] = "BLAS_dtrmv_s";
  XBLAS::trmv(order, uplo, trans, diag, n, alpha, T, ldt, x, incx);
} /* end BLAS_dtrmv_s */
