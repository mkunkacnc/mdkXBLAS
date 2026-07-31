#ifndef XBLAS_WAXPBY_HPP
#define XBLAS_WAXPBY_HPP

#include "impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename X,
         typename Y,
         typename N,
         typename TmpType = T,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<X, T> &&
          impl::size_le_v<Y, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void waxpby(N n,
                      T alpha,
                      const X *x,
                      N incx,
                      T beta,
                      const Y *y,
                      N incy,
                      T *w,
                      N incw)
/*
 * Purpose
 * =======
 *
 * This routine computes:
 *
 *     w <- alpha * x + beta * y
 *
 * Arguments
 * =========
 *
 * n      (input) N
 *        The length of vectors x, y, and w.
 *
 * alpha  (input) T
 *
 * x      (input) const X*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
 *
 * beta   (input) T
 *
 * y      (input) const Y*
 *        Array of length n.
 *
 * incy   (input) N
 *        The stride used to access components y[i].
 *
 * w      (output) T*
 *        Array of length n.
 *
 * incw   (input) N
 *        The stride used to write components w[i].
 *
 */
{
  static const char *routine_name = "XBLAS::waxpby";

  FPU_FIX_DECL;

  /* Test the input parameters. */
  if (n < 0)
    BLAS_error(routine_name, -1, n, nullptr);
  if (incx == 0)
    BLAS_error(routine_name, -4, incx, nullptr);
  else if (incy == 0)
    BLAS_error(routine_name, -7, incy, nullptr);
  else if (incw == 0)
    BLAS_error(routine_name, -9, incw, nullptr);

  /* Immediate return */
  if (n == 0) {
    return;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  IdxType ix = 0;
  if (incx < 0)
    ix = (-n + 1) * incx;

  IdxType iy = 0;
  if (incy < 0)
    iy = (-n + 1) * incy;

  IdxType iw = 0;
  if (incw < 0)
    iw = (-n + 1) * incw;

  for (IdxType i = 0; i < n; ++i) {
    TmpType tmpx = impl::mul<TmpType>(alpha, x[ix]);  /* tmpx = alpha * x[ix] */
    TmpType tmpy = impl::mul<TmpType>(beta, y[iy]);   /* tmpy = beta * y[iy] */
    w[iw] = impl::add<T>(tmpx, tmpy);
    ix += incx;
    iy += incy;
    iw += incw;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::waxpby */

//-----------------

template<typename T,
         typename X,
         typename Y,
         typename N,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<X, T> &&
          impl::size_le_v<Y, T> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void waxpby_x(N n,
                        T alpha,
                        const X *x,
                        N incx,
                        T beta,
                        const Y *y,
                        N incy,
                        T *w,
                        N incw,
                        blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes:
 *
 *     w <- alpha * x + beta * y
 *
 * Arguments
 * =========
 *
 * n      (input) N
 *        The length of vectors x, y, and w.
 *
 * alpha  (input) T
 *
 * x      (input) const X*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
 *
 * beta   (input) T
 *
 * y      (input) const Y*
 *        Array of length n.
 *
 * incy   (input) N
 *        The stride used to access components y[i].
 *
 * w      (output) T*
 *        Array of length n.
 *
 * incw   (input) N
 *        The stride used to write components w[i].
 *
 * prec   (input) blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 */
{
  static const char *routine_name = "XBLAS::waxpby_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::waxpby<T, X, Y, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(n, alpha, x, incx, beta, y, incy, w, incw);
    break;
  case blas_prec_double:
    XBLAS::waxpby<T, X, Y, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(n, alpha, x, incx, beta, y, incy, w, incw);
    break;
  case blas_prec_indigenous:
    XBLAS::waxpby<T, X, Y, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(n, alpha, x, incx, beta, y, incy, w, incw);
    break;
  case blas_prec_extra:
    XBLAS::waxpby<T, X, Y, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(n, alpha, x, incx, beta, y, incy, w, incw);
    break;
  default:
    BLAS_error(routine_name, -10, prec, nullptr);
    break;
  }
} /* end XBLAS::waxpby_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_WAXPBY_HPP
