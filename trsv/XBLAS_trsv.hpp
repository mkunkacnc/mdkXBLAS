#ifndef XBLAS_TRSV_HPP
#define XBLAS_TRSV_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"
#include <new>

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void trsv(blas_order_type order,
                    blas_uplo_type uplo,
                    blas_trans_type trans,
                    blas_diag_type diag,
                    N n,
                    T alpha,
                    const A *t,
                    N ldt,
                    T *x,
                    N incx)
/*
 * Purpose
 * =======
 *
 * This routine solve :
 *
 *     x <- alpha * inverse(T) * x
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        column major, row major
 *
 * uplo   (input) blas_uplo_type
 *        upper, lower
 *
 * trans  (input) blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) blas_diag_type
 *        unit, non unit
 *
 * n      (input) N
 *        the dimension of T
 *
 * alpha  (input) T
 *
 * t      (input) const A*
 *        Triangular matrix
 *
 * x      (input/output) T*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
 *
 */
{
  static const char *routine_name = "XBLAS::trsv";

  FPU_FIX_DECL;

  IdxType i, j;
  IdxType ix, jx;
  IdxType start_x;
  const A *t_i = t;
  T *x_i = x;
  A T_element;
  IdxType incT = 1;

  if ((order != blas_rowmajor && order != blas_colmajor) ||
      (uplo != blas_upper && uplo != blas_lower) ||
      (trans != blas_trans && trans !=
       blas_no_trans && trans != blas_conj_trans) ||
      (diag != blas_non_unit_diag && diag != blas_unit_diag) ||
      (ldt < n) || (incx == 0)) {
    BLAS_error(routine_name, 0, 0, nullptr);
  }

  if (n <= 0)
    return;

  /* configuring the vector starting idx */
  if (incx <= 0) {
    start_x = -(n - 1) * incx;
  } else {
    start_x = 0;
  }

  /* if alpha is zero, then return x as a zero vector */
  if (alpha == T(0)) {
    ix = start_x;
    for (i = 0; i < n; i++) {
      x_i[ix] = T(0);
      ix += incx;
    }
    return;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if constexpr (impl::is_complex_v<A>) {
    if constexpr (sizeof(TmpType) > sizeof(T)) {
      int inc_intx;                /* inc for intx */
      TmpType temp1;
      TmpType temp2;
      TmpType temp3;
      TmpType *intx;

      /* copy of x used for calculations */

      /* allocate space for intx */
      intx = new(std::nothrow) TmpType[n];

      if (n > 0 && intx == nullptr) {
        BLAS_error(routine_name, 0, 0, "allocation failed.\n");
      }

      /* since intx is for internal usage, set it to 1 and then adjust
         it if necessary */
      inc_intx = 1;

      /* copy x to intx */
      ix = start_x;
      jx = 0;
      for (i = 0; i < n; i++) {
        temp1 = impl::to<TmpType>(x_i[ix]);
        intx[jx] = temp1;
        ix += incx;
        jx += inc_intx;
      }

      if ((order == blas_rowmajor &&
           trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor &&
           trans != blas_no_trans && uplo == blas_lower)) {
        if (trans == blas_conj_trans) {

          jx = (n - 1) * inc_intx;
          for (j = n - 1; j >= 0; j--) {
            /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = intx[jx];
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = (n - 1) * inc_intx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = impl::Conj::func(t_i[i * incT + j * ldt * incT]);
              temp3 = intx[ix];
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= inc_intx;
            }                /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(t_i[j * incT + j * ldt * incT]);
              temp1 = impl::div(temp1, T_element);
            }
            /* if (diag == blas_non_unit_diag) */
            intx[jx] = temp1;
            jx -= inc_intx;
          }                        /* for j>=0 */
        } else {

          jx = (n - 1) * inc_intx;
          for (j = n - 1; j >= 0; j--) {

            /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = intx[jx];
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = (n - 1) * inc_intx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = t_i[i * incT + j * ldt * incT];
              temp3 = intx[ix];
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= inc_intx;
            }                /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[j * incT + j * ldt * incT];
              temp1 = impl::div(temp1, T_element);
            }
            /* if (diag == blas_non_unit_diag) */
            intx[jx] = temp1;
            jx -= inc_intx;
          }                        /* for j>=0 */
        }
      } else if ((order == blas_rowmajor &&
                  trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans != blas_no_trans && uplo == blas_upper)) {
        if (trans == blas_conj_trans) {
          jx = 0;
          for (j = 0; j < n; j++) {
            /* compute Xj = Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = intx[jx];
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = 0;
            for (i = 0; i < j; i++) {
              T_element = impl::Conj::func(t_i[i * incT + j * ldt * incT]);
              temp3 = intx[ix];
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += inc_intx;
            }                /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(t_i[j * incT + j * ldt * incT]);
              temp1 = impl::div(temp1, T_element);
            }
            /* if (diag == blas_non_unit_diag) */
            intx[jx] = temp1;
            jx += inc_intx;
          }                        /* for j<n */
        } else {
          jx = 0;
          for (j = 0; j < n; j++) {
            /* compute Xj = Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = intx[jx];
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = 0;
            for (i = 0; i < j; i++) {
              T_element = t_i[i * incT + j * ldt * incT];
              temp3 = intx[ix];
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += inc_intx;
            }                /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[j * incT + j * ldt * incT];
              temp1 = impl::div(temp1, T_element);
            }
            /* if (diag == blas_non_unit_diag) */
            intx[jx] = temp1;
            jx += inc_intx;
          }                        /* for j<n */
        }
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_upper)) {
        if (trans == blas_conj_trans) {

          jx = (n - 1) * inc_intx;
          for (j = n - 1; j >= 0; j--) {
            /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = intx[jx];
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = (n - 1) * inc_intx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = impl::Conj::func(t_i[j * incT + i * ldt * incT]);
              temp3 = intx[ix];
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= inc_intx;
            }                /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(t_i[j * incT + j * ldt * incT]);
              temp1 = impl::div(temp1, T_element);
            }
            /* if (diag == blas_non_unit_diag) */
            intx[jx] = temp1;
            jx -= inc_intx;
          }                        /* for j>=0 */
        } else {

          jx = (n - 1) * inc_intx;
          for (j = n - 1; j >= 0; j--) {
            /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = intx[jx];
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = (n - 1) * inc_intx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = t_i[j * incT + i * ldt * incT];
              temp3 = intx[ix];
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= inc_intx;
            }                /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[j * incT + j * ldt * incT];
              temp1 = impl::div(temp1, T_element);
            }
            /* if (diag == blas_non_unit_diag) */
            intx[jx] = temp1;
            jx -= inc_intx;
          }                        /* for j>=0 */
        }
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_lower)) {
        if (trans == blas_conj_trans) {

          jx = 0;
          for (j = 0; j < n; j++) {
            /* compute Xj = Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = intx[jx];
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = 0;
            for (i = 0; i < j; i++) {
              T_element = impl::Conj::func(t_i[j * incT + i * ldt * incT]);
              temp3 = intx[ix];
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += inc_intx;
            }                /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(t_i[j * incT + j * ldt * incT]);
              temp1 = impl::div(temp1, T_element);
            }
            /* if (diag == blas_non_unit_diag) */
            intx[jx] = temp1;
            jx += inc_intx;
          }                        /* for j<n */
        } else {

          jx = 0;
          for (j = 0; j < n; j++) {
            /* compute Xj = Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = intx[jx];
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = 0;
            for (i = 0; i < j; i++) {
              T_element = t_i[j * incT + i * ldt * incT];
              temp3 = intx[ix];
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += inc_intx;
            }                /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[j * incT + j * ldt * incT];
              temp1 = impl::div(temp1, T_element);
            }
            /* if (diag == blas_non_unit_diag) */
            intx[jx] = temp1;
            jx += inc_intx;
          }                        /* for j<n */
        }
      }

      /* copy the final results from intx to x */
      ix = start_x;
      jx = 0;
      for (i = 0; i < n; i++) {
        temp1 = intx[jx];
        x_i[ix] = impl::to<T>(temp1);
        ix += incx;
        jx += inc_intx;
      }

      delete[] intx;

    } else {
      TmpType temp1;                /* temporary variable for calculations */
      TmpType temp2;                /* temporary variable for calculations */
      TmpType temp3;                /* temporary variable for calculations */

      if ((order == blas_rowmajor &&
           trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor &&
           trans != blas_no_trans && uplo == blas_lower)) {
        if (trans == blas_conj_trans) {

          jx = start_x + (n - 1) * incx;
          for (j = n - 1; j >= 0; j--) {

            /* compute Xj = alpha*Xj - SUM Tij(or Tji) * Xi
               i=j+1 to n-1           */
            temp3 = impl::to<TmpType>(x_i[jx]);
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = start_x + (n - 1) * incx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = impl::Conj::func(t_i[i * incT + j * ldt * incT]);
              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= incx;
            }                        /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(t_i[j * incT + j * ldt * incT]);

              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);

            jx -= incx;
          }                        /* for j>=0 */
        } else {

          jx = start_x + (n - 1) * incx;
          for (j = n - 1; j >= 0; j--) {

            /* compute Xj = alpha*Xj - SUM Tij(or Tji) * Xi
               i=j+1 to n-1           */
            temp3 = impl::to<TmpType>(x_i[jx]);
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = start_x + (n - 1) * incx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = t_i[i * incT + j * ldt * incT];

              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= incx;
            }                        /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[j * incT + j * ldt * incT];


              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);

            jx -= incx;
          }                        /* for j>=0 */
        }
      } else if ((order == blas_rowmajor &&
                  trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans != blas_no_trans && uplo == blas_upper)) {
        if (trans == blas_conj_trans) {

          jx = start_x;
          for (j = 0; j < n; j++) {

            /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = impl::to<TmpType>(x_i[jx]);
            /* multiply by alpha */
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = start_x;
            for (i = 0; i < j; i++) {
              T_element = impl::Conj::func(t_i[i * incT + j * ldt * incT]);
              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += incx;
            }                        /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(t_i[j * incT + j * ldt * incT]);

              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);
            jx += incx;
          }                        /* for j<n */
        } else {

          jx = start_x;
          for (j = 0; j < n; j++) {

            /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = impl::to<TmpType>(x_i[jx]);
            /* multiply by alpha */
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = start_x;
            for (i = 0; i < j; i++) {
              T_element = t_i[i * incT + j * ldt * incT];

              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += incx;
            }                        /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[j * incT + j * ldt * incT];


              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);
            jx += incx;
          }                        /* for j<n */
        }
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_upper)) {
        if (trans == blas_conj_trans) {

          jx = start_x + (n - 1) * incx;
          for (j = n - 1; j >= 0; j--) {

            /* compute Xj = alpha*Xj - SUM Tij(or Tji) * Xi
               i=j+1 to n-1           */
            temp3 = impl::to<TmpType>(x_i[jx]);
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = start_x + (n - 1) * incx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = impl::Conj::func(t_i[j * incT + i * ldt * incT]);
              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= incx;
            }                        /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(t_i[j * incT + j * ldt * incT]);

              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);

            jx -= incx;
          }                        /* for j>=0 */
        } else {

          jx = start_x + (n - 1) * incx;
          for (j = n - 1; j >= 0; j--) {

            /* compute Xj = alpha*Xj - SUM Tij(or Tji) * Xi
               i=j+1 to n-1           */
            temp3 = impl::to<TmpType>(x_i[jx]);
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = start_x + (n - 1) * incx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = t_i[j * incT + i * ldt * incT];

              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= incx;
            }                        /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[j * incT + j * ldt * incT];


              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);

            jx -= incx;
          }                        /* for j>=0 */
        }
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_lower)) {
        if (trans == blas_conj_trans) {

          jx = start_x;
          for (j = 0; j < n; j++) {

            /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = impl::to<TmpType>(x_i[jx]);
            /* multiply by alpha */
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = start_x;
            for (i = 0; i < j; i++) {
              T_element = impl::Conj::func(t_i[j * incT + i * ldt * incT]);
              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += incx;
            }                        /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(t_i[j * incT + j * ldt * incT]);

              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);
            jx += incx;
          }                        /* for j<n */
        } else {

          jx = start_x;
          for (j = 0; j < n; j++) {

            /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
               i=j+1 to n-1           */
            temp3 = impl::to<TmpType>(x_i[jx]);
            /* multiply by alpha */
            temp1 = impl::mul<TmpType>(temp3, alpha);

            ix = start_x;
            for (i = 0; i < j; i++) {
              T_element = t_i[j * incT + i * ldt * incT];

              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += incx;
            }                        /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[j * incT + j * ldt * incT];


              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);
            jx += incx;
          }                        /* for j<n */
        }
      }
    }
  } else {
    if constexpr (sizeof(TmpType) > sizeof(T)) {
      IdxType inc_intx;             /* inc for intx */
      TmpType temp1;                /* temporary variable for calculations */
      TmpType temp2;                /* temporary variable for calculations */
      TmpType temp3;                /* temporary variable for calculations */
      TmpType *intx;                /* copy of x used for calculations */

      /* allocate space for intx */
      intx = new(std::nothrow) TmpType[n];
      if (n > 0 && intx == nullptr) {
        BLAS_error(routine_name, 0, 0, "allocation failed.\n");
      }

      /* since intx is for internal usage, set it to 1 and then adjust
         it if necessary */
      inc_intx = 1;

      /* copy x to intx */
      ix = start_x;
      jx = 0;
      for (i = 0; i < n; i++) {
        temp1 = TmpType(x_i[ix]);
        intx[jx] = temp1;
        ix += incx;
        jx += inc_intx;
      }

      if ((order == blas_rowmajor &&
           trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor &&
           trans != blas_no_trans && uplo == blas_lower)) {

        jx = (n - 1) * inc_intx;
        for (j = n - 1; j >= 0; j--) {

          /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3 = intx[jx];
          /* multiply by alpha */
          temp1 = impl::mul<TmpType>(temp3, alpha);

          ix = (n - 1) * inc_intx;
          for (i = n - 1; i >= j + 1; i--) {
            T_element = t_i[i * incT + j * ldt * incT];
            temp3 = intx[ix];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            ix -= inc_intx;
          }                        /* for j<n */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[j * incT + j * ldt * incT];
            temp1 = impl::div(temp1, T_element);
          }
          /* if (diag == blas_non_unit_diag) */
          intx[jx] = temp1;

          jx -= inc_intx;
        }                        /* for j>=0 */
      } else if ((order == blas_rowmajor &&
                  trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans != blas_no_trans && uplo == blas_upper)) {

        jx = 0;
        for (j = 0; j < n; j++) {
          /* compute Xj = Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3 = intx[jx];
          /* multiply by alpha */
          temp1 = impl::mul<TmpType>(temp3, alpha);

          ix = 0;
          for (i = 0; i < j; i++) {
            T_element = t_i[i * incT + j * ldt * incT];

            temp3 = intx[ix];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            ix += inc_intx;
          }                        /* for i<j */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[j * incT + j * ldt * incT];
            temp1 = impl::div(temp1, T_element);
          }
          /* if (diag == blas_non_unit_diag) */
          intx[jx] = temp1;
          jx += inc_intx;
        }                        /* for j<n */
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_upper)) {

        jx = (n - 1) * inc_intx;
        for (j = n - 1; j >= 0; j--) {
          /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3 = intx[jx];
          /* multiply by alpha */
          temp1 = impl::mul<TmpType>(temp3, alpha);

          ix = (n - 1) * inc_intx;
          for (i = n - 1; i >= j + 1; i--) {
            T_element = t_i[j * incT + i * ldt * incT];

            temp3 = intx[ix];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            ix -= inc_intx;
          }                        /* for j<n */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[j * incT + j * ldt * incT];
            temp1 = impl::div(temp1, T_element);
          }
          /* if (diag == blas_non_unit_diag) */
          intx[jx] = temp1;

          jx -= inc_intx;
        }                        /* for j>=0 */
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_lower)) {

        jx = 0;
        for (j = 0; j < n; j++) {

          /* compute Xj = Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3 = intx[jx];
          /* multiply by alpha */
          temp1 = impl::mul<TmpType>(temp3, alpha);

          ix = 0;
          for (i = 0; i < j; i++) {
            T_element = t_i[j * incT + i * ldt * incT];

            temp3 = intx[ix];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            ix += inc_intx;
          }                        /* for i<j */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[j * incT + j * ldt * incT];
            temp1 = impl::div(temp1, T_element);
          }
          /* if (diag == blas_non_unit_diag) */
          intx[jx] = temp1;
          jx += inc_intx;
        }                        /* for j<n */
      }

      /* copy the final results from intx to x */
      ix = start_x;
      jx = 0;
      for (i = 0; i < n; i++) {
        temp1 = intx[jx];
        x_i[ix] = impl::to<T>(temp1);
        ix += incx;
        jx += inc_intx;
      }

      delete[] intx;
    } else {
      TmpType temp1;
      TmpType temp2;
      TmpType temp3;

      if ((order == blas_rowmajor &&
           trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor &&
           trans != blas_no_trans && uplo == blas_lower)) {

        jx = start_x + (n - 1) * incx;
        for (j = n - 1; j >= 0; j--) {

          /* compute Xj = alpha*Xj - SUM Tij(or Tji) * Xi
             i=j+1 to n-1           */
          temp3 = x_i[jx];
          temp1 = impl::mul<TmpType>(temp3, alpha);

          ix = start_x + (n - 1) * incx;
          for (i = n - 1; i >= j + 1; i--) {
            T_element = t_i[i * incT + j * ldt * incT];

            temp3 = x_i[ix];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            ix -= incx;
          }                        /* for j<n */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[j * incT + j * ldt * incT];
            temp1 = impl::div(temp1, T_element);
          }
          /* if (diag == blas_non_unit_diag) */
          x_i[jx] = impl::to<T>(temp1);

          jx -= incx;
        }                                /* for j>=0 */
      } else if ((order == blas_rowmajor &&
                  trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans != blas_no_trans && uplo == blas_upper)) {

        jx = start_x;
        for (j = 0; j < n; j++) {

          /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3 = x_i[jx];
          /* multiply by alpha */
          temp1 = impl::mul<TmpType>(temp3, alpha);

          ix = start_x;
          for (i = 0; i < j; i++) {
            T_element = t_i[i * incT + j * ldt * incT];

            temp3 = x_i[ix];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            ix += incx;
          }                        /* for i<j */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[j * incT + j * ldt * incT];
            temp1 = impl::div(temp1, T_element);
          }
          /* if (diag == blas_non_unit_diag) */
          x_i[jx] = impl::to<T>(temp1);
          jx += incx;
        }                                /* for j<n */
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_upper)) {

        jx = start_x + (n - 1) * incx;
        for (j = n - 1; j >= 0; j--) {

          /* compute Xj = alpha*Xj - SUM Tij(or Tji) * Xi
             i=j+1 to n-1           */
          temp3 = x_i[jx];
          temp1 = impl::mul<TmpType>(temp3, alpha);

          ix = start_x + (n - 1) * incx;
          for (i = n - 1; i >= j + 1; i--) {
            T_element = t_i[j * incT + i * ldt * incT];

            temp3 = x_i[ix];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            ix -= incx;
          }                        /* for j<n */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[j * incT + j * ldt * incT];
            temp1 = impl::div(temp1, T_element);
          }
          /* if (diag == blas_non_unit_diag) */
          x_i[jx] = impl::to<T>(temp1);

          jx -= incx;
        }                                /* for j>=0 */
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_lower)) {

        jx = start_x;
        for (j = 0; j < n; j++) {

          /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3 = x_i[jx];
          /* multiply by alpha */
          temp1 = impl::mul<TmpType>(temp3, alpha);

          ix = start_x;
          for (i = 0; i < j; i++) {
            T_element = t_i[j * incT + i * ldt * incT];

            temp3 = x_i[ix];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            ix += incx;
          }                        /* for i<j */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[j * incT + j * ldt * incT];
            temp1 = impl::div(temp1, T_element);
          }
          /* if (diag == blas_non_unit_diag) */
          x_i[jx] = impl::to<T>(temp1);
          jx += incx;
        }                                /* for j<n */
      }
    }
  }
  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::trsv */

//-----------------

template<typename T,
         typename A,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void trsv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      blas_trans_type trans,
                      blas_diag_type diag,
                      N n,
                      T alpha,
                      const A *t,
                      N ldt,
                      T *x,
                      N incx,
                      blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine solve :
 *
 *     x <- alpha * inverse(T) * x
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        column major, row major
 *
 * uplo   (input) blas_uplo_type
 *        upper, lower
 *
 * trans  (input) blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) blas_diag_type
 *        unit, non unit
 *
 * n      (input) N
 *        the dimension of T
 *
 * alpha  (input) T
 *
 * t      (input) const A*
 *        Triangular matrix
 *
 * x      (input/output) T*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
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
  static const char *routine_name = "XBLAS::trsv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::trsv<T, A, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_double:
    XBLAS::trsv<T, A, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_indigenous:
    XBLAS::trsv<T, A, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_extra:
    XBLAS::trsv<T, A, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  default:
    BLAS_error(routine_name, -11, prec, nullptr);
    break;
  }
} /* end XBLAS::trsv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TRSV_HPP
