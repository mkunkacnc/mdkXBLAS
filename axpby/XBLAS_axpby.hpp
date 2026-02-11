#ifndef XBLAS_AXPBY_HPP
#define XBLAS_AXPBY_HPP

#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename X,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void axpby(IdxType n,
                     T alpha,
                     const X *x,
                     IdxType incx,
                     T beta,
                     T *y,
                     IdxType incy)
/*
 * Purpose
 * =======
 *
 * This routine computes:
 *
 *      y <- alpha * x + beta * y.
 *
 * Arguments
 * =========
 *
 * n         (input) IdxType
 *           The length of vectors x and y.
 *
 * alpha     (input) T
 *
 * x         (input) const X*
 *           Array of length n.
 *
 * incx      (input) IdxType
 *           The stride used to access components x[i].
 *
 * beta      (input) T
 *
 * y         (input/output) T*
 *           Array of length n.
 *
 * incy      (input) IdxType
 *           The stride used to access components y[i].
 *
 */
{
  static const char routine_name[] = "XBLAS::axpby";

  FPU_FIX_DECL;

  /* Test the input parameters. */
  if (incx == 0)
    BLAS_error(routine_name, -4, incx, NULL);
  else if (incy == 0)
    BLAS_error(routine_name, -7, incy, NULL);

  /* Immediate return */
  if (n <= 0 || (alpha == T(0) && beta == T(1)))
    return;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  IdxType ix = 0;
  if (incx < 0)
    ix = (-n + 1) * incx;
  IdxType iy = 0;
  if (incy < 0)
    iy = (-n + 1) * incy;

  for (IdxType i = 0; i < n; ++i) {
    TmpType tmpx = impl::mul<TmpType>(alpha, x[ix]); /* tmpx = alpha * x[ix] */
    TmpType tmpy = impl::mul<TmpType>(beta, y[iy]);  /* tmpy = beta * y[iy] */
    tmpy += tmpx;
    y[iy] = impl::to<T>(tmpy);
    ix += incx;
    iy += incy;
  } /* endfor */

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::axpby */

//-----------------

template<typename T,
         typename X,
         typename IdxType = int>
requires std::signed_integral<IdxType>
constexpr void axpby_x(IdxType n,
                       T alpha,
                       const X *x,
                       IdxType incx,
                       T beta,
                       T *y,
                       IdxType incy,
                       blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes:
 *
 *      y <- alpha * x + beta * y.
 *
 * Arguments
 * =========
 *
 * n         (input) IdxType
 *           The length of vectors x and y.
 *
 * alpha     (input) T
 *
 * x         (input) const X*
 *           Array of length n.
 *
 * incx      (input) IdxType
 *           The stride used to access components x[i].
 *
 * beta      (input) T
 *
 * y         (input/output) T*
 *           Array of length n.
 *
 * incy      (input) IdxType
 *           The stride used to access components y[i].
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
//static const char routine_name[] = "XBLAS::axpby_x";
  switch (prec) {
  case blas_prec_single:
    axpby<T, X, impl::internal_precision_t<T, blas_prec_single>, IdxType>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    axpby<T, X, impl::internal_precision_t<T, blas_prec_double>, IdxType>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    axpby<T, X, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<T, X, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(n, alpha, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::axpby_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_AXPBY_HPP
