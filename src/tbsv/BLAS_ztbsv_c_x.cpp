#include "blas_extended.h"
#include "tbsv.hpp"

void BLAS_ztbsv_c_x(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    enum blas_trans_type trans,
                    enum blas_diag_type diag,
                    int n,
                    int k,
                    const void *alpha,
                    const void *t,
                    int ldt,
                    void *x,
                    int incx,
                    enum blas_prec_type prec)
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
//static const char *routine_name = "BLAS_ztbsv_c_x";
  XBLAS::tbsv_x(order,
                uplo,
                trans,
                diag,
                n,
                k,
                *static_cast<const std::complex<double> *>(alpha),
                static_cast<const std::complex<float> *>(t),
                ldt,
                static_cast<std::complex<double> *>(x),
                incx,
                prec);
} /* end BLAS_ztbsv_c_x */
