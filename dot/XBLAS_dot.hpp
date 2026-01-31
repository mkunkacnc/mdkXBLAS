#ifndef XBLAS_DOT_HPP
#define XBLAS_DOT_HPP

#include "blas_extended_private.h"
#include "blas_enum.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename X,
         typename Y,
         typename TmpType = T,
         typename IdxType = int>
requires (sizeof(X) <= sizeof(T) &&
          sizeof(Y) <= sizeof(T) &&
          sizeof(TmpType) >= sizeof(T) &&
          std::signed_integral<IdxType>)
constexpr void dot(blas_conj_type conj,
                   IdxType n,
                   T alpha,
                   const X *x,
                   IdxType incx,
                   T beta,
                   const Y *y,
                   IdxType incy,
                   T *r)
/*
 * Purpose
 * =======
 *
 * This routine computes the inner product:
 *
 *     r <- beta * r + alpha * SUM_{i=0, n-1} x[i] * y[i].
 *
 * Arguments
 * =========
 *
 * conj   (input) enum blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *        components x[i] are used unconjugated or conjugated.
 *
 * n      (input) IdxType
 *        The length of vectors x and y.
 *
 * alpha  (input) T
 *
 * x      (input) const X*
 *        Array of length n.
 *
 * incx   (input) IdxType
 *        The stride used to access components x[i].
 *
 * beta   (input) T
 *
 * y      (input) const Y*
 *        Array of length n.
 *
 * incy   (input) IdxType
 *        The stride used to access components y[i].
 *
 * r      (input/output) T*
 *
 */
{
  static const char routine_name[] = "XBLAS::dot";

  FPU_FIX_DECL;

  IdxType i, ix = 0, iy = 0;
  T *r_i = r;
  const X *x_i = x;
  const Y *y_i = y;
  T alpha_i = alpha;
  T beta_i = beta;
  X x_ii;
  Y y_ii;
  T r_v;
  TmpType prod;
  TmpType sum;
  TmpType tmp1;
  TmpType tmp2;

  /* Test the input parameters. */
  if (n < 0)
    BLAS_error(routine_name, -2, n, NULL);
  else if (incx == 0)
    BLAS_error(routine_name, -5, incx, NULL);
  else if (incy == 0)
    BLAS_error(routine_name, -8, incy, NULL);

  /* Immediate return. */
  if ((beta_i == T(1)) && (n == 0 || (alpha_i == T(0))))
    return;

  if constexpr (std::is_same_v<TmpType, double_double> ||
                std::is_same_v<TmpType, std::complex<double_double>>) {
    FPU_FIX_START;
  }

  r_v = r_i[0];
  sum = TmpType(0);

  if (incx < 0)
    ix = (-n + 1) * incx;
  if (incy < 0)
    iy = (-n + 1) * incy;

  if (conj == blas_conj) {
    for (i = 0; i < n; ++i) {
      x_ii = impl::Conj::func(x_i[ix]);
      y_ii = y_i[iy];

      prod = impl::mul<TmpType>(x_ii, y_ii); /* prod = x[i]*y[i] */
      sum = sum + prod;                      /* sum = sum+prod */
      ix += incx;
      iy += incy;
    } /* endfor */
  } else {
    for (i = 0; i < n; ++i) {
      x_ii = x_i[ix];
      y_ii = y_i[iy];

      prod = impl::mul<TmpType>(x_ii, y_ii); /* prod = x[i]*y[i] */
      sum = sum + prod;                      /* sum = sum+prod */
      ix += incx;
      iy += incy;
    } /* endfor */
  }

  tmp1 = sum * alpha_i;                   /* tmp1 = sum*alpha */
  tmp2 = impl::mul<TmpType>(r_v, beta_i); /* tmp2 = r*beta */
  tmp1 = tmp1 + tmp2;                     /* tmp1 = tmp1+tmp2 */
  *r = impl::to<T>(tmp1);                 /* r = tmp1 */


  if constexpr (std::is_same_v<TmpType, double_double> ||
                std::is_same_v<TmpType, std::complex<double_double>>) {
    FPU_FIX_STOP;
  }
}

//-----------------

inline
void mydot(enum blas_conj_type conj,
           int n,
           std::complex<float> alpha,
           const std::complex<float> *x,
           int incx,
           std::complex<float> beta,
           const float *y,
           int incy,
           std::complex<float> *r)
/*
 * Purpose
 * =======
 *
 * This routine computes the inner product:
 *
 *     r <- beta * r + alpha * SUM_{i=0, n-1} x[i] * y[i].
 *
 * Arguments
 * =========
 *
 * conj   (input) enum blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *        components x[i] are used unconjugated or conjugated.
 *
 * n      (input) int
 *        The length of vectors x and y.
 *
 * alpha  (input) const void*
 *
 * x      (input) const void*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 * beta   (input) const void*
 *
 * y      (input) const float*
 *        Array of length n.
 *
 * incy   (input) int
 *        The stride used to access components y[i].
 *
 * r      (input/output) void*
 *
 */
{
  static const char routine_name[] = "BLAS_cdot_c_s";

  using T = std::complex<float>;
  using X = std::complex<float>;
  using Y = float;
  using TmpType = std::complex<float>;

  int i, ix = 0, iy = 0;
  T *r_i = r;
  const X *x_i = x;
  const Y *y_i = y;
  T alpha_i = alpha;
  T beta_i = beta;
  X x_ii;
  Y y_ii;
  T r_v;
  TmpType prod;
  TmpType sum;
  TmpType tmp1;
  TmpType tmp2;


  /* Test the input parameters. */
  if (n < 0)
    BLAS_error(routine_name, -2, n, NULL);
  else if (incx == 0)
    BLAS_error(routine_name, -5, incx, NULL);
  else if (incy == 0)
    BLAS_error(routine_name, -8, incy, NULL);

  /* Immediate return. */
  if (beta_i == T(1) && (n == 0 || alpha_i == T(0)))
    return;

  r_v = r_i[0];
  sum = TmpType(0);

  if (incx < 0)
    ix = (-n + 1) * incx;
  if (incy < 0)
    iy = (-n + 1) * incy;

  if (conj == blas_conj) {
    for (i = 0; i < n; ++i) {
      x_ii = x_i[ix];
      y_ii = y_i[iy];
      x_ii = std::conj(x_ii);
      prod = x_ii * y_ii;
      sum = sum + prod;
      ix += incx;
      iy += incy;
    }                                /* endfor */
  } else {
    /* do not conjugate */
    for (i = 0; i < n; ++i) {
      x_ii = x_i[ix];
      y_ii = y_i[iy];
      prod = x_ii * y_ii;
      sum = sum + prod;
      ix += incx;
      iy += incy;
    }                                /* endfor */
  }

  tmp1 = sum * alpha;
  tmp2 = r_v * beta;
  tmp1 = tmp1 + tmp2;
  *r = tmp1;
}


//-----------------

template<typename T,
         typename X,
         typename Y,
         typename IdxType = int>
requires std::signed_integral<IdxType>
constexpr void dot_x(blas_conj_type conj,
                     IdxType n,
                     T alpha,
                     const X *x,
                     IdxType incx,
                     T beta,
                     const Y *y,
                     IdxType incy,
                     T *r,
                     blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes the inner product:
 *
 *     r <- beta * r + alpha * SUM_{i=0, n-1} x[i] * y[i].
 *
 * Arguments
 * =========
 *
 * conj   (input) enum blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *        components x[i] are used unconjugated or conjugated.
 *
 * n      (input) IdxType
 *        The length of vectors x and y.
 *
 * alpha  (input) T
 *
 * x      (input) const X*
 *        Array of length n.
 *
 * incx   (input) IdxType
 *        The stride used to access components x[i].
 *
 * beta   (input) T
 *
 * y      (input) const Y*
 *        Array of length n.
 *
 * incy   (input) IdxType
 *        The stride used to access components y[i].
 *
 * r      (input/output) T*
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
//static const char routine_name[] = "XBLAS::dot_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::dot<T, X, Y, impl::internal_precision_t<T, blas_prec_single>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_double:
    XBLAS::dot<T, X, Y, impl::internal_precision_t<T, blas_prec_double>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_indigenous:
    XBLAS::dot<T, X, Y, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_extra:
    XBLAS::dot<T, X, Y, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  }
}

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_DOT_HPP
