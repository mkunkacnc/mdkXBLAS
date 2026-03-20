#include "blas_extended.h"
#include "blas_extended_private.h"
#include "tpmv/XBLAS_tpmv.hpp"

void BLAS_ctpmv_s_x(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    enum blas_trans_type trans,
                    enum blas_diag_type diag,
                    int n,
                    const void *alpha,
                    const float *tp,
                    void *x,
                    int incx,
                    enum blas_prec_type prec)
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
 * alpha  (input) const void*
 *
 * tp     (input) const float*
 *
 * x      (input/output) void*
 *
 * incx   (input) int
 *        The stride for vector x.
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
//static const char *routine_name = "BLAS_ctpmv_s_x";
  XBLAS::tpmv_x(order,
                uplo,
                trans,
                diag,
                n,
                *static_cast<const std::complex<float> *>(alpha),
                tp,
                static_cast<std::complex<float> *>(x),
                incx,
                prec);
} /* end BLAS_ctpmv_s_x */
