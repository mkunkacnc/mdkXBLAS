#ifndef XBLAS_SPMV_HPP
#define XBLAS_SPMV_HPP

#include "hpmv.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename TmpType = T,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void spmv(blas_order_type order,
                    blas_uplo_type uplo,
                    N n,
                    T alpha,
                    const A *ap,
                    const X *x,
                    N incx,
                    T beta,
                    T *y,
                    N incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * ap * x + beta * y, where ap is a symmetric
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of ap; row or column major
 *
 * uplo   (input) blas_uplo_type
 *        Whether ap is upper or lower
 *
 * n      (input) N
 *        Dimension of ap and the length of vector x
 *
 * alpha  (input) T
 *
 * ap     (input) const A*
 *
 * x      (input) const X*
 *
 * incx   (input) N
 *        The stride for vector x.
 *
 * beta   (input) T
 *
 * y      (input/output) T*
 *
 * incy   (input) N
 *        The stride for vector y.
 *
 */
{
  static const char *routine_name = "XBLAS::spmv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* Check for error conditions. */
  if (order != blas_colmajor && order != blas_rowmajor) {
    BLAS_error(routine_name, -1, order, nullptr);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  }
  if (n < 0) {
    BLAS_error(routine_name, -3, n, nullptr);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -7, incx, nullptr);
  }
  if (incy == 0) {
    BLAS_error(routine_name, -10, incy, nullptr);
  }

  if (n == 0) {
    return;
  }
  if (alpha == T(0) && beta == T(1)) {
    return;
  }

  IdxType x_start;
  if (incx < 0)
    x_start = (-n + 1) * incx;
  else
    x_start = 0;

  IdxType y_start;
  if (incy < 0)
    y_start = (-n + 1) * incy;
  else
    y_start = 0;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if (alpha == T(0)) {
    IdxType y_index = y_start;
    for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
      TmpType tmp2 = impl::mul<TmpType>(beta, y[y_index]);
      y[y_index] = impl::to<T>(tmp2);
      y_index += incy;
    }
  } else {
    if (uplo == blas_lower)
      order = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;

    if (order == blas_rowmajor) {
      if (alpha == T(1)) {
        if (beta == T(0)) {
          impl::hpmv_impl<0, 0, 1,  0, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
        } else {
          impl::hpmv_impl<0, 0, 1, -1, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
        }
      } else {
        if (beta == T(0)) {
          impl::hpmv_impl<0, 0, -1,  0, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
        } else {
          impl::hpmv_impl<0, 0, -1, -1, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
        }
      }
    } else {
      if (alpha == T(1)) {
        if (beta == T(0)) {
          impl::hpmv_impl<0, 0, 1,  0, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
        } else {
          impl::hpmv_impl<0, 0, 1, -1, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
        }
      } else {
        if (beta == T(0)) {
          impl::hpmv_impl<0, 0, -1,  0, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
        } else {
          impl::hpmv_impl<0, 0, -1, -1, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
        }
      }
    } /* if order == ... */
  }   /* alpha != 0 */

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::spmv */

//-----------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void spmv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      N n,
                      T alpha,
                      const A *ap,
                      const X *x,
                      N incx,
                      T beta,
                      T *y,
                      N incy,
                      blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * ap * x + beta * y, where ap is a symmetric
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of ap; row or column major
 *
 * uplo   (input) blas_uplo_type
 *        Whether ap is upper or lower
 *
 * n      (input) N
 *        Dimension of ap and the length of vector x
 *
 * alpha  (input) T
 *
 * ap     (input) const A*
 *
 * x      (input) const X*
 *
 * incx   (input) N
 *        The stride for vector x.
 *
 * beta   (input) T
 *
 * y      (input/output) T*
 *
 * incy   (input) N
 *        The stride for vector y.
 * prec   (input) blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 *
 */
{
  static const char *routine_name = "XBLAS::spmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::spmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::spmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::spmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::spmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -11, prec, nullptr);
    break;
  }
} /* end XBLAS::spmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_SPMV_HPP
