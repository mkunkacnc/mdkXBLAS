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
requires (std::floating_point<B> &&
          has_value_type<C> &&
          !std::is_same_v<typename C::value_type, B> &&
          !std::is_same_v<typename C::value_type, DoubleDouble>)
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
inline std::complex<DoubleDouble> mul(std::complex<float> a, std::complex<float> b)
{
  /* Real part */
  double d1 = static_cast<double>(real(a)) * real(b);
  double d2 = static_cast<double>(-imag(a)) * imag(b);
  DoubleDouble cr = DoubleDouble::add(d1, d2); /* ar*br - ai*bi */
  /* imaginary part */
  d1 = static_cast<double>(real(a)) * imag(b);
  d2 = static_cast<double>(imag(a)) * real(b);
  DoubleDouble ci = DoubleDouble::add(d1, d2); /* ar*bi + ai*br */
  return std::complex<DoubleDouble>(cr, ci);
}

template<>
inline std::complex<DoubleDouble> mul(std::complex<double> a, std::complex<double> b)
{
  /* Compute complex-extra = complex-double * complex-double. */
  /* Real part */
  DoubleDouble t1 = DoubleDouble::mul( real(a), real(b));
  DoubleDouble t2 = DoubleDouble::mul(-imag(a), imag(b));
  DoubleDouble cr = t1 + t2; /* ar*br - ai*bi */
  /* Imaginary part */
  t1 = DoubleDouble::mul(imag(a), real(b));
  t2 = DoubleDouble::mul(real(a), imag(b));
  DoubleDouble ci = t1 + t2; /* ar*bi + ai*br */
  return std::complex<DoubleDouble>(cr, ci);
}

template<>
inline std::complex<DoubleDouble> mul(std::complex<float> a, float b)
{
  return std::complex<DoubleDouble>(DoubleDouble::mul(real(a), b), DoubleDouble::mul(imag(a), b));
}

template<>
inline std::complex<DoubleDouble> mul(std::complex<double> a, double b)
{
  return std::complex<DoubleDouble>(DoubleDouble::mul(real(a), b), DoubleDouble::mul(imag(a), b));
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

  if constexpr (std::is_same_v<TmpType, DoubleDouble> ||
                std::is_same_v<TmpType, std::complex<DoubleDouble>>) {
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

  if constexpr (std::is_same_v<TmpType, DoubleDouble> ||
                std::is_same_v<TmpType, std::complex<DoubleDouble>>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::axpby */

//-----------------

//template<>
inline void my_axpby(int n,
              std::complex<double> alpha,
              const std::complex<float> *x,
              int incx,
              std::complex<double> beta,
              std::complex<double> *y,
              int incy)
{
  static const char routine_name[] = "XBLAS::my_axpby";

  using T = std::complex<double>;
  using X = std::complex<float>;
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
    {
      T cd = x_ii;
      tmpx = mul<TmpType>(alpha_i, cd);
      #if 0
      {
        /* Compute complex-extra = complex-double * complex-double. */
        double head_t1, tail_t1;
        double head_t2, tail_t2;
        /* Real part */
        compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, real(alpha_i), real(cd));
        compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, imag(alpha_i), imag(cd));
        head_t2 = -head_t2;
        tail_t2 = -tail_t2;
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
        tmpx[0] = DoubleDouble(head_t1, tail_t1);
        /* Imaginary part */
        compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, imag(alpha_i), real(cd));
        compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, real(alpha_i), imag(cd));
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
        tmpx[1] = DoubleDouble(head_t1, tail_t1);
      }
      #endif
    }                        /* tmpx  = alpha * x[ix] */
    {
      tmpy = mul<TmpType>(beta_i, y_ii);
      #if 0
      /* Compute complex-extra = complex-double * complex-double. */
      double head_t1, tail_t1;
      double head_t2, tail_t2;
      /* Real part */
      compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, real(beta_i), real(y_ii));
      compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, imag(beta_i), imag(y_ii));
      head_t2 = -head_t2;
      tail_t2 = -tail_t2;
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
      tmpy[0] = DoubleDouble(head_t1, tail_t1);
      /* Imaginary part */
      compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, imag(beta_i), real(y_ii));
      compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, real(beta_i), imag(y_ii));
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
      tmpy[1] = DoubleDouble(head_t1, tail_t1);
      #endif
    }                        /* tmpy = beta * y[iy] */
    {
      double head_t, tail_t;
      double head_a, tail_a;
      double head_b, tail_b;
      /* Real part */
      head_a = real(tmpy).head;
      tail_a = real(tmpy).tail;
      head_b = real(tmpx).head;
      tail_b = real(tmpx).tail;
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
      DoubleDouble tmpyr;
      tmpyr = DoubleDouble(head_t, tail_t);
      /* Imaginary part */
      head_a = imag(tmpy).head;
      tail_a = imag(tmpy).tail;
      head_b = imag(tmpx).head;
      tail_b = imag(tmpx).tail;
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
      DoubleDouble tmpyi;
      tmpyi = DoubleDouble(head_t, tail_t);
      tmpy = TmpType(tmpyr, tmpyi);
    }
    y_i[iy] = T(real(tmpy).head, imag(tmpy).head);
    ix += incx;
    iy += incy;
  }                                /* endfor */

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

template<typename X>
void axpby_x(int n,
             std::complex<double> alpha,
             const X *x,
             int incx,
             std::complex<double> beta,
             std::complex<double> *y,
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
 * alpha     (input) std::complex<double>
 *
 * x         (input) const X*
 *           Array of length n.
 *
 * incx      (input) int
 *           The stride used to access components x[i].
 *
 * beta      (input) std::complex<double>
 *
 * y         (input) doublestd::complex<double>*
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
    axpby<std::complex<double>, X, std::complex<DoubleDouble>>(n, alpha, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::axpby_x */

template<typename X> inline
void axpby_x(int n,
             std::complex<float> alpha,
             const X *x,
             int incx,
             std::complex<float> beta,
             std::complex<float> *y,
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
    axpby<std::complex<float>, X, std::complex<double>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<std::complex<float>, X, std::complex<DoubleDouble>>(n, alpha, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::axpby_x */

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_AXPBY_HPP
