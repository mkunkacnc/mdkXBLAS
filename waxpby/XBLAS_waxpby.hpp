#ifndef XBLAS_WAXPBY_HPP
#define XBLAS_WAXPBY_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename X,
         typename Y,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<X, T> &&
          impl::size_le_v<Y, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void waxpby(IdxType n,
                      T alpha,
                      const X *x,
                      IdxType incx,
                      T beta,
                      const Y *y,
                      IdxType incy,
                      T *w,
                      IdxType incw)
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
 * n      (input) IdxType
 *        The length of vectors x, y, and w.
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
 * w      (output) T*
 *        Array of length n.
 *
 * incw   (input) IdxType
 *        The stride used to write components w[i].
 *
 */
{
  static const char routine_name[] = "XBLAS::waxpby";

  FPU_FIX_DECL;

  /* Test the input parameters. */
  if (incx == 0)
    BLAS_error(routine_name, -4, incx, NULL);
  else if (incy == 0)
    BLAS_error(routine_name, -7, incy, NULL);
  else if (incw == 0)
    BLAS_error(routine_name, -9, incw, NULL);

  /* Immediate return */
  if (n <= 0) {
    return;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  IdxType ix = 0, iy = 0, iw = 0;
  if (incx < 0)
    ix = (-n + 1) * incx;
  if (incy < 0)
    iy = (-n + 1) * incy;
  if (incw < 0)
    iw = (-n + 1) * incw;

  for (IdxType i = 0; i < n; ++i) {
    TmpType tmpx = impl::mul<TmpType>(alpha, x[ix]);  /* tmpx = alpha * x[ix] */
    TmpType tmpy = impl::mul<TmpType>(beta, y[iy]);   /* tmpy = beta * y[iy] */
    w[iw] = impl::add<T>(tmpx, tmpy);
    ix += incx;
    iy += incy;
    iw += incw;
  } /* endfor */

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::waxpby */

//-----------------

inline
constexpr void waxpby_x(int n,
                        double alpha,
                        const double *x,
                        int incx,
                        double beta,
                        const float *y,
                        int incy,
                        double *w,
                        int incw,
                        enum blas_prec_type prec)
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
 * n      (input) int
 *        The length of vectors x, y, and w.
 *
 * alpha  (input) double
 *
 * x      (input) const double*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 * beta   (input) double
 *
 * y      (input) const float*
 *        Array of length n.
 *
 * incy   (input) int
 *        The stride used to access components y[i].
 *
 * w      (output) double*
 *        Array of length n.
 *
 * incw   (input) int
 *        The stride used to write components w[i].
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
  static const char routine_name[] = "BLAS_dwaxpby_d_s_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, ix = 0, iy = 0, iw = 0;
      double *w_i = w;
      const double *x_i = x;
      const float *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      double x_ii;
      float y_ii;
      double tmpx;
      double tmpy;



      /* Test the input parameters. */
      if (incx == 0)
        BLAS_error(routine_name, -4, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -7, incy, NULL);
      else if (incw == 0)
        BLAS_error(routine_name, -9, incw, NULL);


      /* Immediate return */
      if (n <= 0) {
        return;
      }






      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;
      if (incw < 0)
        iw = (-n + 1) * incw;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];
        tmpx = alpha_i * x_ii;        /* tmpx  = alpha * x[ix] */
        tmpy = beta_i * y_ii;        /* tmpy = beta * y[iy] */
        tmpy = tmpy + tmpx;
        w_i[iw] = tmpy;
        ix += incx;
        iy += incy;
        iw += incw;
      }                                /* endfor */



      break;
    }

  case blas_prec_extra:{

      int i, ix = 0, iy = 0, iw = 0;
      double *w_i = w;
      const double *x_i = x;
      const float *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      double x_ii;
      float y_ii;
      double head_tmpx, tail_tmpx;
      double head_tmpy, tail_tmpy;

      FPU_FIX_DECL;

      /* Test the input parameters. */
      if (incx == 0)
        BLAS_error(routine_name, -4, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -7, incy, NULL);
      else if (incw == 0)
        BLAS_error(routine_name, -9, incw, NULL);


      /* Immediate return */
      if (n <= 0) {
        return;
      }

      FPU_FIX_START;




      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;
      if (incw < 0)
        iw = (-n + 1) * incw;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];
        compute_doubledouble_eq_double_mul_double(&head_tmpx, &tail_tmpx, alpha_i, x_ii);
        {
          double dt = (double) y_ii;
          compute_doubledouble_eq_double_mul_double(&head_tmpy, &tail_tmpy, beta_i, dt);
        }                        /* tmpy = beta * y[iy] */
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmpy, &tail_tmpy, head_tmpy, tail_tmpy, head_tmpx, tail_tmpx);
        w_i[iw] = head_tmpy;
        ix += incx;
        iy += incy;
        iw += incw;
      }                                /* endfor */

      FPU_FIX_STOP;

      break;
    }
  }
} /* end XBLAS::waxpby_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_WAXPBY_HPP
