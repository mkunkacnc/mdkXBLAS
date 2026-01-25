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
    tmpx = mul<TmpType>(alpha_i, x_ii); /* tmpx = alpha * x[ix] */
    tmpy = mul<TmpType>(beta_i, y_ii);  /* tmpy = beta * y[iy] */
    tmpy = tmpy + tmpx;
    y_i[iy] = to<T>(tmpy);
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
  static_assert("Missing specialization");
} /* end BLAS_axpby_x_cpp */

template<typename T, int prec>
struct internal_precision { using type = T; };

template<>
struct internal_precision<float, blas_prec_single> { using type = float; };
template<>
struct internal_precision<float, blas_prec_double> { using type = double; };
template<>
struct internal_precision<float, blas_prec_indigenous> { using type = double; };
template<>
struct internal_precision<float, blas_prec_extra> { using type = double_double; };

template<>
struct internal_precision<double, blas_prec_single> { using type = double; };
template<>
struct internal_precision<double, blas_prec_double> { using type = double; };
template<>
struct internal_precision<double, blas_prec_indigenous> { using type = double; };
template<>
struct internal_precision<double, blas_prec_extra> { using type = double_double; };

template<>
struct internal_precision<std::complex<float>, blas_prec_single> { using type = std::complex<float>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_double> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_indigenous> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_extra> { using type = std::complex<double_double>; };

template<>
struct internal_precision<std::complex<double>, blas_prec_single> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<double>, blas_prec_double> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<double>, blas_prec_indigenous> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<double>, blas_prec_extra> { using type = std::complex<double_double>; };

template<typename T, int prec>
using internal_precision_t = typename internal_precision<T, prec>::type;

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
 * y         (input/output) double*
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
    axpby<double, X, internal_precision_t<double, blas_prec_single>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    axpby<double, X, internal_precision_t<double, blas_prec_double>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    axpby<double, X, internal_precision_t<double, blas_prec_indigenous>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<double, X, internal_precision_t<double, blas_prec_extra>>(n, alpha, x, incx, beta, y, incy);
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
 * y         (input/output) float*
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
    axpby<float, float, internal_precision_t<float, blas_prec_single>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    axpby<float, float, internal_precision_t<float, blas_prec_double>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    axpby<float, float, internal_precision_t<float, blas_prec_indigenous>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<float, float, internal_precision_t<float, blas_prec_extra>>(n, alpha, x, incx, beta, y, incy);
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
 * y         (input/output) std::complex<double>*
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
    axpby<std::complex<double>, X, internal_precision_t<std::complex<double>, blas_prec_single>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    axpby<std::complex<double>, X, internal_precision_t<std::complex<double>, blas_prec_double>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    axpby<std::complex<double>, X, internal_precision_t<std::complex<double>, blas_prec_indigenous>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<std::complex<double>, X, internal_precision_t<std::complex<double>, blas_prec_extra>>(n, alpha, x, incx, beta, y, incy);
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
 * alpha     (input) std::complex<float>
 *
 * x         (input) const X*
 *           Array of length n.
 *
 * incx      (input) int
 *           The stride used to access components x[i].
 *
 * beta      (input) std::complex<float>
 *
 * y         (input/output) std::complex<float>*
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
    axpby<std::complex<float>, X, internal_precision_t<std::complex<float>, blas_prec_single>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    axpby<std::complex<float>, X, internal_precision_t<std::complex<float>, blas_prec_double>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    axpby<std::complex<float>, X, internal_precision_t<std::complex<float>, blas_prec_indigenous>>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    axpby<std::complex<float>, X, internal_precision_t<std::complex<float>, blas_prec_extra>>(n, alpha, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::axpby_x */

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_AXPBY_HPP
