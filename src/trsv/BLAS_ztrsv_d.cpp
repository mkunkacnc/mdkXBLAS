#include <math.h>
#include "blas_extended.h"
#include "blas_extended_private.h"
#include "trsv.hpp"

void BLAS_ztrsv_d(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  enum blas_trans_type trans,
                  enum blas_diag_type diag,
                  int n,
                  const void *alpha,
                  const double *T,
                  int ldt,
                  void *x,
                  int incx)
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
 * alpha  (input) const void*
 *
 * T      (input) const double*
 *        Triangular matrix
 *
 * x      (input/output) void*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 */
{
//static const char *routine_name = "BLAS_ztrsv_d";
  XBLAS::trsv(order,
              uplo,
              trans,
              diag,
              n,
              *static_cast<const std::complex<double> *>(alpha),
              T,
              ldt,
              static_cast<std::complex<double> *>(x),
              incx);
} /* end BLAS_ztrsv_d */
