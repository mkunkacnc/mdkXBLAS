#ifndef XBLAS_HPMV_HPP
#define XBLAS_HPMV_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void hpmv(blas_order_type order,
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
 * Computes y = alpha * ap * x + beta * y, where ap is a hermitian
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
  static const char *routine_name = "XBLAS::hpmv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  enum blas_order_type order_i;

  IdxType ap_index, ap_start, x_index, x_start;
  IdxType y_start, y_index, incap;

  PrdType rowsum;
  PrdType rowtmp;
  A matval;
  X vecval;
  T resval;
  TmpType tmp1;
  TmpType tmp2;

  if (n < 1)
    return;
  if (alpha == T(0) && beta == T(1))
    return;

  /* Check for error conditions. */
  if (order != blas_colmajor && order != blas_rowmajor)
    BLAS_error(routine_name, -1, order, nullptr);
  if (uplo != blas_upper && uplo != blas_lower)
    BLAS_error(routine_name, -2, uplo, nullptr);
  if (incx == 0)
    BLAS_error(routine_name, -7, incx, nullptr);
  if (incy == 0) {
    BLAS_error(routine_name, -10, incy, nullptr);
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  incap = 1;

  if (incx < 0)
    x_start = (-n + 1) * incx;
  else
    x_start = 0;
  if (incy < 0)
    y_start = (-n + 1) * incy;
  else
    y_start = 0;

  if (uplo == blas_lower)
    order_i = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;
  else
    order_i = order;

  if (alpha == T(0)) {
    y_index = y_start;
    for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
      resval = y[y_index];
      tmp2 = impl::mul<TmpType>(beta, resval);
      y[y_index] = impl::to<T>(tmp2);
      y_index += incy;
    }
  } else {
    if (order_i == blas_rowmajor) {
      if (alpha == T(1)) {
        if (beta == T(0)) {
          y_index = y_start;
          ap_start = 0;
          if (uplo == blas_upper) {
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (n - step - 1) * incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              tmp1 = rowsum;
              y[y_index] = impl::to<T>(tmp1);
              y_index += incy;
              ap_start += incap;
            }
          } else { /* if uplo == ... */
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (n - step - 1) * incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              tmp1 = rowsum;
              y[y_index] = impl::to<T>(tmp1);
              y_index += incy;
              ap_start += incap;
            }
          } /* end if uplo == blas_upper ... */
        } else {
          y_index = y_start;
          ap_start = 0;
          if (uplo == blas_upper) {
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (n - step - 1) * incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              resval = y[y_index];
              tmp1 = rowsum;
              tmp2 = impl::mul<TmpType>(beta, resval);
              tmp2 = tmp1 + tmp2;
              y[y_index] = impl::to<T>(tmp2);
              y_index += incy;
              ap_start += incap;
            }
          } else {                /* if uplo == ... */
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (n - step - 1) * incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              resval = y[y_index];
              tmp1 = rowsum;
              tmp2 = impl::mul<TmpType>(beta, resval);
              tmp2 = tmp1 + tmp2;
              y[y_index] = impl::to<T>(tmp2);
              y_index += incy;
              ap_start += incap;
            }
          } /* end if uplo == blas_upper ... */
        }
      } else {
        if (beta == T(0)) {
          y_index = y_start;
          ap_start = 0;
          if (uplo == blas_upper) {
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (n - step - 1) * incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              tmp1 = impl::mul<TmpType>(rowsum, alpha);
              y[y_index] = impl::to<T>(tmp1);
              y_index += incy;
              ap_start += incap;
            }
          } else {                /* if uplo == ... */
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (n - step - 1) * incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              tmp1 = impl::mul<TmpType>(rowsum, alpha);
              y[y_index] = impl::to<T>(tmp1);
              y_index += incy;
              ap_start += incap;
            }
          } /* end if uplo == blas_upper ... */
        } else {
          y_index = y_start;
          ap_start = 0;
          if (uplo == blas_upper) {
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (n - step - 1) * incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              resval = y[y_index];
              tmp1 = impl::mul<TmpType>(rowsum, alpha);
              tmp2 = impl::mul<TmpType>(beta, resval);
              tmp2 = tmp1 + tmp2;
              y[y_index] = impl::to<T>(tmp2);
              y_index += incy;
              ap_start += incap;
            }
          } else {                /* if uplo == ... */
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (n - step - 1) * incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              resval = y[y_index];
              tmp1 = impl::mul<TmpType>(rowsum, alpha);
              tmp2 = impl::mul<TmpType>(beta, resval);
              tmp2 = tmp1 + tmp2;
              y[y_index] = impl::to<T>(tmp2);
              y_index += incy;
              ap_start += incap;
            }
          } /* end if uplo == blas_upper ... */
        }
      }
    } else {
      if (alpha == T(1)) {
        if (beta == T(0)) {
          y_index = y_start;
          ap_start = 0;
          if (uplo == blas_upper) {
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += (matrix_row + 1) * incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (step + 1) * incap;
                x_index += incx;
              }
              tmp1 = rowsum;
              y[y_index] = impl::to<T>(tmp1);
              y_index += incy;
              ap_start += (matrix_row + 1) * incap;
            }
          } else {                /* if uplo == ... */
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += (matrix_row + 1) * incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (step + 1) * incap;
                x_index += incx;
              }
              tmp1 = rowsum;
              y[y_index] = impl::to<T>(tmp1);
              y_index += incy;
              ap_start += (matrix_row + 1) * incap;
            }
          } /* end if uplo == blas_upper ... */
        } else {
          {
            y_index = y_start;
            ap_start = 0;
            if (uplo == blas_upper) {
              for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
                x_index = x_start;
                ap_index = ap_start;
                rowsum = impl::zero_v<PrdType>;
                rowtmp = impl::zero_v<PrdType>; // not needed
                for (IdxType step = 0; step < matrix_row; step++) {
                  matval = impl::Conj::func(ap[ap_index]);
                  vecval = x[x_index];
                  rowtmp = impl::mul<PrdType>(matval, vecval);
                  rowsum = rowsum + rowtmp;
                  ap_index += incap;
                  x_index += incx;
                }
                /* need to do diagonal element without referencing the imaginary part */
                auto matval_r = std::real(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval_r, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (matrix_row + 1) * incap;
                x_index += incx;
                for (IdxType step = matrix_row + 1; step < n; step++) {
                  matval = ap[ap_index];
                  vecval = x[x_index];
                  rowtmp = impl::mul<PrdType>(matval, vecval);
                  rowsum = rowsum + rowtmp;
                  ap_index += (step + 1) * incap;
                  x_index += incx;
                }
                resval = y[y_index];
                tmp1 = rowsum;
                tmp2 = impl::mul<TmpType>(beta, resval);
                tmp2 = tmp1 + tmp2;
                y[y_index] = impl::to<T>(tmp2);
                y_index += incy;
                ap_start += (matrix_row + 1) * incap;
              }
            } else {                /* if uplo == ... */
              for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
                x_index = x_start;
                ap_index = ap_start;
                rowsum = impl::zero_v<PrdType>;
                rowtmp = impl::zero_v<PrdType>; // not needed
                for (IdxType step = 0; step < matrix_row; step++) {
                  matval = ap[ap_index];
                  vecval = x[x_index];
                  rowtmp = impl::mul<PrdType>(matval, vecval);
                  rowsum = rowsum + rowtmp;
                  ap_index += incap;
                  x_index += incx;
                }
                /* need to do diagonal element without referencing the imaginary part */
                auto matval_r = std::real(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval_r, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (matrix_row + 1) * incap;
                x_index += incx;
                for (IdxType step = matrix_row + 1; step < n; step++) {
                  matval = impl::Conj::func(ap[ap_index]);
                  vecval = x[x_index];
                  rowtmp = impl::mul<PrdType>(matval, vecval);
                  rowsum = rowsum + rowtmp;
                  ap_index += (step + 1) * incap;
                  x_index += incx;
                }
                resval = y[y_index];
                tmp1 = rowsum;
                tmp2 = impl::mul<TmpType>(beta, resval);
                tmp2 = tmp1 + tmp2;
                y[y_index] = impl::to<T>(tmp2);
                y_index += incy;
                ap_start += (matrix_row + 1) * incap;
              }
            } /* end if uplo == blas_upper ... */
          }
        }
      } else {
        if (beta == T(0)) {
          y_index = y_start;
          ap_start = 0;
          if (uplo == blas_upper) {
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += (matrix_row + 1) * incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (step + 1) * incap;
                x_index += incx;
              }
              tmp1 = impl::mul<TmpType>(rowsum, alpha);
              y[y_index] = impl::to<T>(tmp1);
              y_index += incy;
              ap_start += (matrix_row + 1) * incap;
            }
          } else {                /* if uplo == ... */
            for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start;
              ap_index = ap_start;
              rowsum = impl::zero_v<PrdType>;
              rowtmp = impl::zero_v<PrdType>; // not needed
              for (IdxType step = 0; step < matrix_row; step++) {
                matval = ap[ap_index];
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += incap;
                x_index += incx;
              }
              /* need to do diagonal element without referencing the imaginary part */
              auto matval_r = std::real(ap[ap_index]);
              vecval = x[x_index];
              rowtmp = impl::mul<PrdType>(matval_r, vecval);
              rowsum = rowsum + rowtmp;
              ap_index += (matrix_row + 1) * incap;
              x_index += incx;
              for (IdxType step = matrix_row + 1; step < n; step++) {
                matval = impl::Conj::func(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (step + 1) * incap;
                x_index += incx;
              }
              tmp1 = impl::mul<TmpType>(rowsum, alpha);
              y[y_index] = impl::to<T>(tmp1);
              y_index += incy;
              ap_start += (matrix_row + 1) * incap;
            }
          } /* end if uplo == blas_upper ... */
        } else {
          {
            y_index = y_start;
            ap_start = 0;
            if (uplo == blas_upper) {
              for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
                x_index = x_start;
                ap_index = ap_start;
                rowsum = impl::zero_v<PrdType>;
                rowtmp = impl::zero_v<PrdType>; // not needed
                for (IdxType step = 0; step < matrix_row; step++) {
                  matval = impl::Conj::func(ap[ap_index]);
                  vecval = x[x_index];
                  rowtmp = impl::mul<PrdType>(matval, vecval);
                  rowsum = rowsum + rowtmp;
                  ap_index += incap;
                  x_index += incx;
                }
                /* need to do diagonal element without referencing the imaginary part */
                auto matval_r = std::real(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval_r, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (matrix_row + 1) * incap;
                x_index += incx;
                for (IdxType step = matrix_row + 1; step < n; step++) {
                  matval = ap[ap_index];
                  vecval = x[x_index];
                  rowtmp = impl::mul<PrdType>(matval, vecval);
                  rowsum = rowsum + rowtmp;
                  ap_index += (step + 1) * incap;
                  x_index += incx;
                }
                resval = y[y_index];
                tmp1 = impl::mul<TmpType>(rowsum, alpha);
                tmp2 = impl::mul<TmpType>(beta, resval);
                tmp2 = tmp1 + tmp2;
                y[y_index] = impl::to<T>(tmp2);
                y_index += incy;
                ap_start += (matrix_row + 1) * incap;
              }
            } else {                /* if uplo == ... */
              for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
                x_index = x_start;
                ap_index = ap_start;
                rowsum = impl::zero_v<PrdType>;
                rowtmp = impl::zero_v<PrdType>; // not needed
                for (IdxType step = 0; step < matrix_row; step++) {
                  matval = ap[ap_index];
                  vecval = x[x_index];
                  rowtmp = impl::mul<PrdType>(matval, vecval);
                  rowsum = rowsum + rowtmp;
                  ap_index += incap;
                  x_index += incx;
                }
                /* need to do diagonal element without referencing the imaginary part */
                auto matval_r = std::real(ap[ap_index]);
                vecval = x[x_index];
                rowtmp = impl::mul<PrdType>(matval_r, vecval);
                rowsum = rowsum + rowtmp;
                ap_index += (matrix_row + 1) * incap;
                x_index += incx;
                for (IdxType step = matrix_row + 1; step < n; step++) {
                  matval = impl::Conj::func(ap[ap_index]);
                  vecval = x[x_index];
                  rowtmp = impl::mul<PrdType>(matval, vecval);
                  rowsum = rowsum + rowtmp;
                  ap_index += (step + 1) * incap;
                  x_index += incx;
                }
                resval = y[y_index];
                tmp1 = impl::mul<TmpType>(rowsum, alpha);
                tmp2 = impl::mul<TmpType>(beta, resval);
                tmp2 = tmp1 + tmp2;
                y[y_index] = impl::to<T>(tmp2);
                y_index += incy;
                ap_start += (matrix_row + 1) * incap;
              }
            } /* end if uplo == blas_upper ... */
          }
        }
      }
    }
  } /* end alpha != 0 */

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::hpmv */

//-----------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void hpmv_x(blas_order_type order,
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
 * Computes y = alpha * ap * x + beta * y, where ap is a hermitian
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
  static const char *routine_name = "XBLAS::hpmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::hpmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::hpmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::hpmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::hpmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -11, prec, nullptr);
    break;
  }
} /* end XBLAS::hpmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HPMV_HPP
