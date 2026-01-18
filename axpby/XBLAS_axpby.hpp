#ifndef XBLAS_AXPBY_HPP
#define XBLAS_AXPBY_HPP

#include "blas_extended_private.h"
#include "common/BLAS_doubledouble.hpp"

#include <type_traits>
#include <complex>

//---------------
namespace XBLAS {
//---------------

template<typename T>
concept has_value_type = requires(T t)  {
  typename T::value_type;
};

//-----------------

template<typename C,
         typename A,
         typename B>
requires (!std::is_same_v<C, A> || !std::is_same_v<C, B>)
inline C mul(A a, B b)
{
  return static_cast<C>(a) * b;
}

template<typename C>
inline C mul(C a, C b)
{
  return a * b;
}

template<typename C,
         typename A,
         typename B>
requires (std::floating_point<B> && has_value_type<C> && !std::is_same_v<typename C::value_type, B>)
inline C mul(std::complex<A> a, B b)
{
  return mul<C>(a, static_cast<C::value_type>(b));
}

template<typename C,
         typename A,
         typename B>
requires (has_value_type<C> && !std::is_same_v<typename C::value_type, B>)
inline C mul(std::complex<A> a, std::complex<B> b)
{
  return mul<C>(a, static_cast<C>(b));
}

template<>
inline DoubleDouble mul(double a, double b)
{
  return DoubleDouble::mul(a, b);
}

template<>
inline DoubleDouble mul(double a, float b)
{
  return DoubleDouble::mul(a, static_cast<double>(b));
}

template<>
inline DoubleDouble mul(float a, float b)
{
  return DoubleDouble::mul(a, b);
}

template<>
inline std::complex<DoubleDouble> mul(std::complex<float> beta_i, std::complex<float> y_ii)
{
  DoubleDouble e1, e2;
  //double head_e1, tail_e1;
  //double head_e2, tail_e2;
  double d1;
  double d2;
  /* Real part */
  d1 = (double) real(beta_i) * real(y_ii);
  d2 = (double) -imag(beta_i) * imag(y_ii);
  e1 = DoubleDouble::add(d1, d2);
  //compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
  /* imaginary part */
  d1 = (double) real(beta_i) * imag(y_ii);
  d2 = (double) imag(beta_i) * real(y_ii);
  e2 = DoubleDouble::add(d1, d2);
  //compute_doubledouble_eq_double_add_double(&head_e2, &tail_e2, d1, d2);
  //return std::complex<DoubleDouble>(DoubleDouble(head_e1, tail_e1), DoubleDouble(head_e2, tail_e2));
  return std::complex<DoubleDouble>(e1, e2);
}

template<>
inline std::complex<DoubleDouble> mul(std::complex<float> a, float b)
{
  return std::complex<DoubleDouble>(mul<DoubleDouble>(real(a), b), mul<DoubleDouble>(imag(a), b));
}

//-----------------

template<typename To,
         typename From>
inline To to(From from)
{
  return static_cast<To>(from);
}

template<typename To>
inline To to(DoubleDouble from)
{
  return static_cast<To>(from.head);
}

template<typename To>
requires has_value_type<To>
inline To to(std::complex<DoubleDouble> from)
{
  return To(to<typename To::value_type>(real(from)), to<typename To::value_type>(imag(from)));
}

//-----------------

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
 * y         (input) T*
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

  if constexpr (std::is_same_v<TmpType, DoubleDouble>) {
    FPU_FIX_START;
  }

  if (incx < 0)
    ix = (-n + 1) * incx;
  if (incy < 0)
    iy = (-n + 1) * incy;

  for (i = 0; i < n; ++i) {
    x_ii = x_i[ix];
    y_ii = y_i[iy];
    tmpx = mul<TmpType>(alpha_i, x_ii); /* tmpx = alpha * x[ix] */
    tmpy = mul<TmpType>(beta_i, y_ii);  /* tmpy = beta * y[iy] */
    tmpy = tmpy + tmpx;
    y_i[iy] = to<T>(tmpy);
    ix += incx;
    iy += incy;
  } /* endfor */

  if constexpr (std::is_same_v<TmpType, DoubleDouble>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::axpby */

//template<>
inline void my_axpby(int n,
              std::complex<float> alpha,
              const float *x,
              int incx,
              std::complex<float> beta,
              std::complex<float> *y,
              int incy)
{
  static const char routine_name[] = "XBLAS::my_axpby";

  using T = std::complex<float>;
  using X = float;
  using TmpType = std::complex<DoubleDouble>;

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

  FPU_FIX_START;

  if (incx < 0)
    ix = (-n + 1) * incx;
  if (incy < 0)
    iy = (-n + 1) * incy;

  for (i = 0; i < n; ++i) {
    x_ii = x_i[ix];
    y_ii = y_i[iy];
    tmpx = mul<TmpType>(alpha_i, x_ii); /* tmpx = alpha * x[ix] */
    tmpy = mul<TmpType>(beta_i, y_ii);  /* tmpy = beta * y[iy] */
#if 0
    {
      double head_e1, tail_e1;
      double head_e2, tail_e2;
      double d1;
      double d2;
      /* Real part */
      d1 = (double) real(beta_i) * real(y_ii);
      d2 = (double) -imag(beta_i) * imag(y_ii);
      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
      /* imaginary part */
      d1 = (double) real(beta_i) * imag(y_ii);
      d2 = (double) imag(beta_i) * real(y_ii);
      compute_doubledouble_eq_double_add_double(&head_e2, &tail_e2, d1, d2);
      tmpy = TmpType(DoubleDouble(head_e1, tail_e1), DoubleDouble(head_e2, tail_e2));
    }                        /* tmpy = beta * y[iy] */
#endif
    tmpy = tmpy + tmpx;
    y_i[iy] = to<T>(tmpy);
    ix += incx;
    iy += incy;
  } /* endfor */

  FPU_FIX_STOP;
}

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
  static_assert("Missing specialization");
} /* end BLAS_axpby_x_cpp */

template<typename X>
void axpby_x(int n,
             double alpha,
             const X *x,
             int incx,
             double beta,
             double *y,
             int incy,
             enum blas_prec_type prec)
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
 * alpha     (input) double
 *
 * x         (input) const X*
 *           Array of length n.
 *
 * incx      (input) int
 *           The stride used to access components x[i].
 *
 * beta      (input) double
 *
 * y         (input) double*
 *           Array of length n.
 *
 * incy      (input) int
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
  case blas_prec_double:
  case blas_prec_indigenous:
    axpby(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<double, X, DoubleDouble>(n, alpha, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::axpby_x */

template<> inline
void axpby_x(int n,
             float alpha,
             const float *x,
             int incx,
             float beta,
             float *y,
             int incy,
             enum blas_prec_type prec)
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
 * alpha     (input) float
 *
 * x         (input) const float*
 *           Array of length n.
 *
 * incx      (input) int
 *           The stride used to access components x[i].
 *
 * beta      (input) float
 *
 * y         (input) float*
 *           Array of length n.
 *
 * incy      (input) int
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
    axpby(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
  case blas_prec_indigenous:
    axpby<float, float, double>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<float, float, DoubleDouble>(n, alpha, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::axpby_x */

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_AXPBY_HPP
