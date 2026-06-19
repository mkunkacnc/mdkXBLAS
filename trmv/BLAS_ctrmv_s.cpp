#include "blas_extended.h"
#include "blas_extended_private.h"
#include "trmv.hpp"

void BLAS_ctrmv_s(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  enum blas_trans_type trans,
                  enum blas_diag_type diag,
                  int n,
                  const void *alpha,
                  const float *T,
                  int ldt,
                  void *x,
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
 * alpha  (input) const void*
 *
 * T      (input) const float*
 *        Triangular matrix
 *
 * ldt    (input) int
 *        Leading dimension of T
 *
 * x      (input/output) void*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 */
{
//static const char *routine_name = "BLAS_ctrmv_s";
  XBLAS::trmv(order,
              uplo,
              trans,
              diag,
              n,
              *static_cast<const std::complex<float> *>(alpha),
              T,
              ldt,
              static_cast<std::complex<float> *>(x),
              incx);
} /* end BLAS_ctrmv_s */
