#include <math.h>
#include "blas_extended.h"
#include "blas_extended_private.h"
#include "tbsv.hpp"

void BLAS_ztbsv_c(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  enum blas_trans_type trans,
                  enum blas_diag_type diag,
                  int n,
                  int k,
                  const void *alpha,
                  const void *t,
                  int ldt,
                  void *x,
                  int incx)
/*
 * Purpose
 * =======
 *
 * This routine solves :
 *
 *     x <- alpha * inverse(t) * x
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        column major, row major (blas_rowmajor, blas_colmajor)
 *
 * uplo   (input) enum blas_uplo_type
 *        upper, lower (blas_upper, blas_lower)
 *
 * trans  (input) enum blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) enum blas_diag_type
 *        unit, non unit (blas_unit_diag, blas_non_unit_diag)
 *
 * n      (input) int
 *        the dimension of t
 *
 * k      (input) int
 *        the number of subdiagonals/superdiagonals of t
 *
 * alpha  (input) const void*
 *
 * t      (input) const void*
 *        Triangular Banded matrix
 *
 * x      (input/output) void*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 */
{
//static const char *routine_name = "BLAS_ztbsv_c";
  XBLAS::tbsv(order,
              uplo,
              trans,
              diag,
              n,
              k,
              *static_cast<const std::complex<double> *>(alpha),
              static_cast<const std::complex<float> *>(t),
              ldt,
              static_cast<std::complex<double> *>(x),
              incx);
} /* end BLAS_ztbsv_c */
