#include "blas_extended.h"
#include "blas_extended_private.h"
#include "tpmv/XBLAS_tpmv.hpp"

void BLAS_dtpmv_s(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  enum blas_trans_type trans,
                  enum blas_diag_type diag,
                  int n,
                  double alpha,
                  const float *tp,
                  double *x,
                  int incx)
/*
 * Purpose
 * =======
 *
 * Computes x = alpha * tp * x, x = alpha * tp_transpose * x,
 * or x = alpha * tp_conjugate_transpose where tp is a triangular
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of tp; row or column major
 *
 * uplo   (input) enum blas_uplo_type
 *        Whether tp is upper or lower
 *
 * trans  (input) enum blas_trans_type
 *
 * diag   (input) enum blas_diag_type
 *        Whether the diagonal entries of tp are 1
 *
 * n      (input) int
 *        Dimension of tp and the length of vector x
 *
 * alpha  (input) double
 *
 * tp     (input) const float*
 *
 * x      (input/output) double*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 */
{
//static const char routine_name[] = "BLAS_dtpmv_s";
  XBLAS::tpmv(order, uplo, trans, diag, n, alpha, tp, x, incx);
} /* end BLAS_dtpmv_s */
