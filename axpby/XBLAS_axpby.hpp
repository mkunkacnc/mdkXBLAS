#ifndef XBLAS_AXPBY_HPP
#define XBLAS_AXPBY_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename X,
         typename TmpType = T>
requires (sizeof(X) <= sizeof(T) && sizeof(TmpType) >= sizeof(T))
void axpby(int n,
           T alpha,
           const X *x,
           int incx,
           T beta,
           T *y,
           int incy)
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

  int i, ix = 0, iy = 0;
  const X *x_i = x;
  T *y_i = y;
  T alpha_i = alpha;
  T beta_i = beta;
  X x_ii;
  T y_ii;
  TmpType tmpx;
  TmpType tmpy;
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

  if (incx < 0)
    ix = (-n + 1) * incx;
  if (incy < 0)
    iy = (-n + 1) * incy;

  for (i = 0; i < n; ++i) {
    x_ii = x_i[ix];
    y_ii = y_i[iy];
    tmpx = impl::mul<TmpType>(alpha_i, x_ii); /* tmpx = alpha * x[ix] */
    tmpy = impl::mul<TmpType>(beta_i, y_ii);  /* tmpy = beta * y[iy] */
    tmpy = tmpy + tmpx;
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
         typename X>
void axpby_x(int n,
             T alpha,
             const X *x,
             int incx,
             T beta,
             T *y,
             int incy,
             enum blas_prec_type prec)
{
//static const char routine_name[] = "XBLAS::axpby_x";
  switch (prec) {
  case blas_prec_single:
    axpby<T, X, impl::internal_precision_t<T, blas_prec_single>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    axpby<T, X, impl::internal_precision_t<T, blas_prec_double>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    axpby<T, X, impl::internal_precision_t<T, blas_prec_indigenous>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<T, X, impl::internal_precision_t<T, blas_prec_extra>>(n, alpha, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::axpby_x */

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_AXPBY_HPP
