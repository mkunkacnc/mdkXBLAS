#include <math.h>
#include "blas_extended.h"
#include "blas_extended_private.h"
#include "trsv/XBLAS_trsv.hpp"

void BLAS_ctrsv_x(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  enum blas_trans_type trans,
                  enum blas_diag_type diag,
                  int n,
                  const void *alpha,
                  const void *T,
                  int ldt,
                  void *x,
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
 * alpha  (input) const void*
 *
 * T      (input) const void*
 *        Triangular matrix
 *
 * x      (input/output) void*
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
//static const char *routine_name = "BLAS_ctrsv_x";
  XBLAS::trsv_x(order,
                uplo,
                trans,
                diag,
                n,
                *static_cast<const std::complex<float> *>(alpha),
                static_cast<const std::complex<float> *>(T),
                ldt,
                static_cast<std::complex<float> *>(x),
                incx,
                prec);
} /* end BLAS_ctrsv_x */
