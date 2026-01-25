#ifndef XBLAS_AXPBY_HPP
#define XBLAS_AXPBY_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename X,
         typename TmpType = T,
         typename IdxType = int>
requires (sizeof(X) <= sizeof(T) &&
          sizeof(TmpType) >= sizeof(T) &&
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
 * n         (input) int
 *           The length of vectors x and y.
 *
 * alpha     (input) T
 *
 * x         (input) const X*
 *           Array of length n.
 *
 * incx      (input) int
 *           The stride used to access components x[i].
 *
 * beta      (input) T
 *
 * y         (input/output) T*
 *           Array of length n.
 *
 * incy      (input) int
 *           The stride used to access components y[i].
 *
 */
{
  static const char routine_name[] = "XBLAS::axpby";

  const X *x_i = x;
  T *y_i = y;
  T alpha_i = alpha;
  T beta_i = beta;
  FPU_FIX_DECL;

  /* Test the input parameters. */
  if (incx == 0)
    BLAS_error(routine_name, -4, incx, NULL);
  else if (incy == 0)
    BLAS_error(routine_name, -7, incy, NULL);

  /* Immediate return */
  if (n <= 0 || (alpha_i == T(0) && beta_i == T(1)))
    return;

  if constexpr (std::is_same_v<TmpType, double_double> ||
                std::is_same_v<TmpType, std::complex<double_double>>) {
    FPU_FIX_START;
  }

  IdxType ix = 0;
  if (incx < 0)
    ix = (-n + 1) * incx;
  IdxType iy = 0;
  if (incy < 0)
    iy = (-n + 1) * incy;

  for (IdxType i = 0; i < n; ++i) {
    X x_ii = x_i[ix];
    T y_ii = y_i[iy];
    TmpType tmpx = impl::mul<TmpType>(alpha_i, x_ii); /* tmpx = alpha * x[ix] */
    TmpType tmpy = impl::mul<TmpType>(beta_i, y_ii);  /* tmpy = beta * y[iy] */
    tmpy += tmpx;
    y_i[iy] = impl::to<T>(tmpy);
    ix += incx;
    iy += incy;
  } /* endfor */

  if constexpr (std::is_same_v<TmpType, double_double> ||
                std::is_same_v<TmpType, std::complex<double_double>>) {
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
                       enum blas_prec_type prec)
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

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_AXPBY_HPP
