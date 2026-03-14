#include "blas_extended.h"
#include "blas_extended_private.h"
#include "hpmv/XBLAS_hpmv.hpp"

void BLAS_zhpmv_c_z(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    int n,
                    const void *alpha,
                    const void *ap,
                    const void *x,
                    int incx,
                    const void *beta,
                    void *y,
                    int incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * ap * x + beta * y, where ap is a hermitian
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
 * alpha  (input) const void*
 *
 * ap     (input) const void*
 *
 * x      (input) const void*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) const void*
 *
 * y      (input/output) void*
 *
 * incy   (input) int
 *        The stride for vector y.
 *
 */
{
//static const char routine_name[] = "BLAS_zhpmv_c_z";
  XBLAS::hpmv(order,
              uplo,
              n,
              *static_cast<const std::complex<double> *>(alpha),
              static_cast<const std::complex<float> *>(ap),
              static_cast<const std::complex<double> *>(x),
              incx,
              *static_cast<const std::complex<double> *>(beta),
              static_cast<std::complex<double> *>(y),
              incy);
} /* end BLAS_zhpmv_c_z */
