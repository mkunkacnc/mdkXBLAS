#ifndef XBLAS_TRSV_HPP
#define XBLAS_TRSV_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void trsv(blas_order_type order,
                    blas_uplo_type uplo,
                    blas_trans_type trans,
                    blas_diag_type diag,
                    IdxType n,
                    T alpha,
                    const A *t,
                    IdxType ldt,
                    T *x,
                    IdxType incx)
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
 * n      (input) IdxType
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
 * incx   (input) IdxType
 *        The stride used to access components x[i].
 *
 */
{
  static const char routine_name[] = "XBLAS::trsv";

  FPU_FIX_DECL;

  IdxType i, j;
  IdxType ix, jx;
  IdxType start_x;
  const A *t_i = t;
  T *x_i = x;
  T alpha_i = alpha;
  A T_element;
  IdxType incT = 1;

  if ((order != blas_rowmajor && order != blas_colmajor) ||
      (uplo != blas_upper && uplo != blas_lower) ||
      (trans != blas_trans && trans !=
       blas_no_trans && trans != blas_conj_trans) ||
      (diag != blas_non_unit_diag && diag != blas_unit_diag) ||
      (ldt < n) || (incx == 0)) {
    BLAS_error(routine_name, 0, 0, NULL);
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
  if (alpha_i == T(0)) {
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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
      intx = (TmpType *) blas_malloc(n * sizeof(TmpType));
      if (n > 0 && intx == NULL) {
        BLAS_error("blas_malloc", 0, 0, "malloc failed.\n");
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
          temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
          temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
          temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
          temp1 = impl::mul<TmpType>(temp3, alpha_i);

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

      blas_free(intx);
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
          temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
          temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
          temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
          temp1 = impl::mul<TmpType>(temp3, alpha_i);

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
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void trsv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      blas_trans_type trans,
                      blas_diag_type diag,
                      IdxType n,
                      T alpha,
                      const A *t,
                      IdxType ldt,
                      T *x,
                      IdxType incx,
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
 * n      (input) IdxType
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
 * incx   (input) IdxType
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
//static const char routine_name[] = "XBLAS::trsv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::trsv<T, A, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_double:
    XBLAS::trsv<T, A, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_indigenous:
    XBLAS::trsv<T, A, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_extra:
    XBLAS::trsv<T, A, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  }
} /* end XBLAS::trsv_x */




template<typename T,
         typename A,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void my_trsv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      blas_trans_type trans,
                      blas_diag_type diag,
                      IdxType n,
                      T alpha,
                      const A *t,
                      IdxType ldt,
                      T *x,
                      IdxType incx,
                      blas_prec_type prec)
/*void BLAS_ztrsv_c_x(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    enum blas_trans_type trans,
                    enum blas_diag_type diag,
                    int n,
                    const void *alpha,
                    const void *T,
                    int ldt,
                    void *x,
                    int incx,
                    enum blas_prec_type prec)*/
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
 * order  (input) enum blas_order_type
 *        column major, row major
 *
 * uplo   (input) enum blas_uplo_type
 *        upper, lower
 *
 * trans  (input) enum blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) enum blas_diag_type
 *        unit, non unit
 *
 * n      (input) int
 *        the dimension of T
 *
 * alpha  (input) const void*
 *
 * T      (input) const void*
 *        Triangular matrix
 *
 * x      (input/output) void*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
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
//static const char routine_name[] = "BLAS_ztrsv_c_x";
#if 0
  XBLAS::trsv_x(order,
                uplo,
                trans,
                diag,
                n,
                *static_cast<const std::complex<double> *>(alpha),
                static_cast<const std::complex<float> *>(T),
                ldt,
                static_cast<std::complex<double> *>(x),
                incx,
                prec);
} /* end BLAS_ztrsv_c_x */
#else
  static const char routine_name[] = "BLAS_ztrsv_c";
  static_assert(std::is_same_v<T, std::complex<double>>);
  static_assert(std::is_same_v<A, std::complex<float>>);

  int i, j;                        /* used to idx matrix */
  int ix, jx;                        /* used to idx vector x */
  int start_x;                        /* used as the starting idx to vector x */
  const A *T_i = t;        /* internal matrix T */
  T *x_i = x;        /* internal x */
  T alpha_i = alpha;        /* internal alpha */
  A T_element;                /* temporary variable for an element of matrix A */
  int incT = 1;                        /* internal ldt */

  if ((order != blas_rowmajor && order != blas_colmajor) ||
      (uplo != blas_upper && uplo != blas_lower) ||
      (trans != blas_trans && trans !=
       blas_no_trans && trans != blas_conj_trans) ||
      (diag != blas_non_unit_diag && diag != blas_unit_diag) ||
      (ldt < n) || (incx == 0)) {
    BLAS_error(routine_name, 0, 0, NULL);
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
  if (alpha_i == T(0)) {
    ix = start_x;
    for (i = 0; i < n; i++) {
      x_i[ix] = T(0);
      ix += incx;
    }
    return;
  }
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:

    {
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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

            ix = start_x + (n - 1) * incx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = impl::Conj::func(T_i[i * incT + j * ldt * incT]);
              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= incx;
            }                        /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(T_i[j * incT + j * ldt * incT]);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

            ix = start_x + (n - 1) * incx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = T_i[i * incT + j * ldt * incT];

              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= incx;
            }                        /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = T_i[j * incT + j * ldt * incT];


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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

            ix = start_x;
            for (i = 0; i < j; i++) {
              T_element = impl::Conj::func(T_i[i * incT + j * ldt * incT]);
              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += incx;
            }                        /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(T_i[j * incT + j * ldt * incT]);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

            ix = start_x;
            for (i = 0; i < j; i++) {
              T_element = T_i[i * incT + j * ldt * incT];

              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += incx;
            }                        /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = T_i[j * incT + j * ldt * incT];


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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

            ix = start_x + (n - 1) * incx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = impl::Conj::func(T_i[j * incT + i * ldt * incT]);
              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= incx;
            }                        /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(T_i[j * incT + j * ldt * incT]);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

            ix = start_x + (n - 1) * incx;
            for (i = n - 1; i >= j + 1; i--) {
              T_element = T_i[j * incT + i * ldt * incT];

              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix -= incx;
            }                        /* for j<n */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = T_i[j * incT + j * ldt * incT];


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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

            ix = start_x;
            for (i = 0; i < j; i++) {
              T_element = impl::Conj::func(T_i[j * incT + i * ldt * incT]);
              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += incx;
            }                        /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = impl::Conj::func(T_i[j * incT + j * ldt * incT]);

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
            temp1 = impl::mul<TmpType>(temp3, alpha_i);

            ix = start_x;
            for (i = 0; i < j; i++) {
              T_element = T_i[j * incT + i * ldt * incT];

              temp3 = impl::to<TmpType>(x_i[ix]);
              temp2 = impl::mul<TmpType>(temp3, T_element);
              temp1 = temp1 - temp2;
              ix += incx;
            }                        /* for i<j */

            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = T_i[j * incT + j * ldt * incT];


              temp1 = impl::div(temp1, T_element);

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[jx] = impl::to<T>(temp1);
            jx += incx;
          }                        /* for j<n */
        }
      }
    }
    break;
  case blas_prec_extra:
    {
      static_assert(std::is_same_v<TmpType, std::complex<double>>);
      using TmpType_t = std::complex<double_double>;

      FPU_FIX_DECL;
      FPU_FIX_START;
      {
        {
          int inc_intx;                /* inc for intx */
          double head_temp1[2], tail_temp1[2];        /* temporary variable for calculations */
          double head_temp2[2], tail_temp2[2];        /* temporary variable for calculations */
          double head_temp3[2], tail_temp3[2];        /* temporary variable for calculations */
          double *head_intx, *tail_intx;
          /* copy of x used for calculations */

          /* allocate space for intx */
          head_intx = (double *) blas_malloc(n * sizeof(double) * 2);
          tail_intx = (double *) blas_malloc(n * sizeof(double) * 2);
          if (n > 0 && (head_intx == NULL || tail_intx == NULL)) {
            BLAS_error("blas_malloc", 0, 0, "malloc failed.\n");
          }

          /* since intx is for internal usage, set it to 1 and then adjust
             it if necessary */
          inc_intx = 1;
          inc_intx *= 2;

          /* copy x to intx */
          ix = start_x;
          jx = 0;
          for (i = 0; i < n; i++) {
            head_temp1[0] = std::real(x_i[ix]);
            tail_temp1[0] = 0.0;
            head_temp1[1] = std::imag(x_i[ix]);
            tail_temp1[1] = 0.0;
            head_intx[jx] = head_temp1[0];
            tail_intx[jx] = tail_temp1[0];
            head_intx[1 + jx] = head_temp1[1];
            tail_intx[1 + jx] = tail_temp1[1];
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
                head_temp3[0] = head_intx[jx];
                head_temp3[1] = head_intx[1 + jx];
                tail_temp3[0] = tail_intx[jx];
                tail_temp3[1] = tail_intx[1 + jx];
                /* multiply by alpha */
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, std::imag(alpha_i));
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, std::imag(alpha_i));
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }


                ix = (n - 1) * inc_intx;
                for (i = n - 1; i >= j + 1; i--) {
                  T_element = impl::Conj::func(T_i[i * incT + j * ldt * incT]);
                  head_temp3[0] = head_intx[ix];
                  head_temp3[1] = head_intx[1 + ix];
                  tail_temp3[0] = tail_intx[ix];
                  tail_temp3[1] = tail_intx[1 + ix];
                  {
                    double cd[2];
                    cd[0] = (double) std::real(T_element);
                    cd[1] = (double) std::imag(T_element);
                    {
                      /* Compute complex-extra = complex-extra * complex-double. */
                      double head_a0, tail_a0;
                      double head_a1, tail_a1;
                      double head_t1, tail_t1;
                      double head_t2, tail_t2;
                      head_a0 = head_temp3[0];
                      tail_a0 = tail_temp3[0];
                      head_a1 = head_temp3[1];
                      tail_a1 = tail_temp3[1];
                      /* real part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                      head_t2 = -head_t2;
                      tail_t2 = -tail_t2;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[0] = head_t1;
                      tail_temp2[0] = tail_t1;
                      /* imaginary part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[1] = head_t1;
                      tail_temp2[1] = tail_t1;
                    }

                  }
                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  impl::inner_type_t<TmpType_t> temp2r(head_temp2[0], tail_temp2[0]);
                  impl::inner_type_t<TmpType_t> temp2i(head_temp2[1], tail_temp2[1]);
                  TmpType_t temp2(temp2r, temp2i);
                  temp1 = temp1 - temp2;
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                  ix -= inc_intx;
                }                /* for j<n */

                /* if the diagonal entry is not equal to one, then divide Xj by
                   the entry */
                if (diag == blas_non_unit_diag) {
                  T_element = impl::Conj::func(T_i[j * incT + j * ldt * incT]);

                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  temp1 = impl::div(temp1, T_element);
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                }
                /* if (diag == blas_non_unit_diag) */
                head_intx[jx] = head_temp1[0];
                tail_intx[jx] = tail_temp1[0];
                head_intx[1 + jx] = head_temp1[1];
                tail_intx[1 + jx] = tail_temp1[1];

                jx -= inc_intx;
              }                        /* for j>=0 */
            } else {

              jx = (n - 1) * inc_intx;
              for (j = n - 1; j >= 0; j--) {

                /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
                   i=j+1 to n-1           */
                head_temp3[0] = head_intx[jx];
                head_temp3[1] = head_intx[1 + jx];
                tail_temp3[0] = tail_intx[jx];
                tail_temp3[1] = tail_intx[1 + jx];
                /* multiply by alpha */
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, std::imag(alpha_i));
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, std::imag(alpha_i));
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }


                ix = (n - 1) * inc_intx;
                for (i = n - 1; i >= j + 1; i--) {
                  T_element = T_i[i * incT + j * ldt * incT];

                  head_temp3[0] = head_intx[ix];
                  head_temp3[1] = head_intx[1 + ix];
                  tail_temp3[0] = tail_intx[ix];
                  tail_temp3[1] = tail_intx[1 + ix];
                  {
                    double cd[2];
                    cd[0] = (double) std::real(T_element);
                    cd[1] = (double) std::imag(T_element);
                    {
                      /* Compute complex-extra = complex-extra * complex-double. */
                      double head_a0, tail_a0;
                      double head_a1, tail_a1;
                      double head_t1, tail_t1;
                      double head_t2, tail_t2;
                      head_a0 = head_temp3[0];
                      tail_a0 = tail_temp3[0];
                      head_a1 = head_temp3[1];
                      tail_a1 = tail_temp3[1];
                      /* real part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                      head_t2 = -head_t2;
                      tail_t2 = -tail_t2;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[0] = head_t1;
                      tail_temp2[0] = tail_t1;
                      /* imaginary part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[1] = head_t1;
                      tail_temp2[1] = tail_t1;
                    }

                  }
                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  impl::inner_type_t<TmpType_t> temp2r(head_temp2[0], tail_temp2[0]);
                  impl::inner_type_t<TmpType_t> temp2i(head_temp2[1], tail_temp2[1]);
                  TmpType_t temp2(temp2r, temp2i);
                  temp1 = temp1 - temp2;
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                  ix -= inc_intx;
                }                /* for j<n */

                /* if the diagonal entry is not equal to one, then divide Xj by
                   the entry */
                if (diag == blas_non_unit_diag) {
                  T_element = T_i[j * incT + j * ldt * incT];


                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  temp1 = impl::div(temp1, T_element);
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();

                }
                /* if (diag == blas_non_unit_diag) */
                head_intx[jx] = head_temp1[0];
                tail_intx[jx] = tail_temp1[0];
                head_intx[1 + jx] = head_temp1[1];
                tail_intx[1 + jx] = tail_temp1[1];

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
                head_temp3[0] = head_intx[jx];
                head_temp3[1] = head_intx[1 + jx];
                tail_temp3[0] = tail_intx[jx];
                tail_temp3[1] = tail_intx[1 + jx];
                /* multiply by alpha */
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, std::imag(alpha_i));
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, std::imag(alpha_i));
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }


                ix = 0;
                for (i = 0; i < j; i++) {
                  T_element = impl::Conj::func(T_i[i * incT + j * ldt * incT]);
                  head_temp3[0] = head_intx[ix];
                  head_temp3[1] = head_intx[1 + ix];
                  tail_temp3[0] = tail_intx[ix];
                  tail_temp3[1] = tail_intx[1 + ix];
                  {
                    double cd[2];
                    cd[0] = (double) std::real(T_element);
                    cd[1] = (double) std::imag(T_element);
                    {
                      /* Compute complex-extra = complex-extra * complex-double. */
                      double head_a0, tail_a0;
                      double head_a1, tail_a1;
                      double head_t1, tail_t1;
                      double head_t2, tail_t2;
                      head_a0 = head_temp3[0];
                      tail_a0 = tail_temp3[0];
                      head_a1 = head_temp3[1];
                      tail_a1 = tail_temp3[1];
                      /* real part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                      head_t2 = -head_t2;
                      tail_t2 = -tail_t2;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[0] = head_t1;
                      tail_temp2[0] = tail_t1;
                      /* imaginary part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[1] = head_t1;
                      tail_temp2[1] = tail_t1;
                    }

                  }
                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  impl::inner_type_t<TmpType_t> temp2r(head_temp2[0], tail_temp2[0]);
                  impl::inner_type_t<TmpType_t> temp2i(head_temp2[1], tail_temp2[1]);
                  TmpType_t temp2(temp2r, temp2i);
                  temp1 = temp1 - temp2;
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                  ix += inc_intx;
                }                /* for i<j */

                /* if the diagonal entry is not equal to one, then divide Xj by
                   the entry */
                if (diag == blas_non_unit_diag) {
                  T_element = impl::Conj::func(T_i[j * incT + j * ldt * incT]);

                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  temp1 = impl::div(temp1, T_element);
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();

                }
                /* if (diag == blas_non_unit_diag) */
                head_intx[jx] = head_temp1[0];
                tail_intx[jx] = tail_temp1[0];
                head_intx[1 + jx] = head_temp1[1];
                tail_intx[1 + jx] = tail_temp1[1];
                jx += inc_intx;
              }                        /* for j<n */
            } else {

              jx = 0;
              for (j = 0; j < n; j++) {

                /* compute Xj = Xj - SUM Aij(or Aji) * Xi
                   i=j+1 to n-1           */
                head_temp3[0] = head_intx[jx];
                head_temp3[1] = head_intx[1 + jx];
                tail_temp3[0] = tail_intx[jx];
                tail_temp3[1] = tail_intx[1 + jx];
                /* multiply by alpha */
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, std::imag(alpha_i));
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, std::imag(alpha_i));
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }


                ix = 0;
                for (i = 0; i < j; i++) {
                  T_element = T_i[i * incT + j * ldt * incT];

                  head_temp3[0] = head_intx[ix];
                  head_temp3[1] = head_intx[1 + ix];
                  tail_temp3[0] = tail_intx[ix];
                  tail_temp3[1] = tail_intx[1 + ix];
                  {
                    double cd[2];
                    cd[0] = (double) std::real(T_element);
                    cd[1] = (double) std::imag(T_element);
                    {
                      /* Compute complex-extra = complex-extra * complex-double. */
                      double head_a0, tail_a0;
                      double head_a1, tail_a1;
                      double head_t1, tail_t1;
                      double head_t2, tail_t2;
                      head_a0 = head_temp3[0];
                      tail_a0 = tail_temp3[0];
                      head_a1 = head_temp3[1];
                      tail_a1 = tail_temp3[1];
                      /* real part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                      head_t2 = -head_t2;
                      tail_t2 = -tail_t2;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[0] = head_t1;
                      tail_temp2[0] = tail_t1;
                      /* imaginary part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[1] = head_t1;
                      tail_temp2[1] = tail_t1;
                    }

                  }
                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  impl::inner_type_t<TmpType_t> temp2r(head_temp2[0], tail_temp2[0]);
                  impl::inner_type_t<TmpType_t> temp2i(head_temp2[1], tail_temp2[1]);
                  TmpType_t temp2(temp2r, temp2i);
                  temp1 = temp1 - temp2;
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                  ix += inc_intx;
                }                /* for i<j */

                /* if the diagonal entry is not equal to one, then divide Xj by
                   the entry */
                if (diag == blas_non_unit_diag) {
                  T_element = T_i[j * incT + j * ldt * incT];


                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  temp1 = impl::div(temp1, T_element);
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();

                }
                /* if (diag == blas_non_unit_diag) */
                head_intx[jx] = head_temp1[0];
                tail_intx[jx] = tail_temp1[0];
                head_intx[1 + jx] = head_temp1[1];
                tail_intx[1 + jx] = tail_temp1[1];
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
                head_temp3[0] = head_intx[jx];
                head_temp3[1] = head_intx[1 + jx];
                tail_temp3[0] = tail_intx[jx];
                tail_temp3[1] = tail_intx[1 + jx];
                /* multiply by alpha */
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, std::imag(alpha_i));
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, std::imag(alpha_i));
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }


                ix = (n - 1) * inc_intx;
                for (i = n - 1; i >= j + 1; i--) {
                  T_element = impl::Conj::func(T_i[j * incT + i * ldt * incT]);
                  head_temp3[0] = head_intx[ix];
                  head_temp3[1] = head_intx[1 + ix];
                  tail_temp3[0] = tail_intx[ix];
                  tail_temp3[1] = tail_intx[1 + ix];
                  {
                    double cd[2];
                    cd[0] = (double) std::real(T_element);
                    cd[1] = (double) std::imag(T_element);
                    {
                      /* Compute complex-extra = complex-extra * complex-double. */
                      double head_a0, tail_a0;
                      double head_a1, tail_a1;
                      double head_t1, tail_t1;
                      double head_t2, tail_t2;
                      head_a0 = head_temp3[0];
                      tail_a0 = tail_temp3[0];
                      head_a1 = head_temp3[1];
                      tail_a1 = tail_temp3[1];
                      /* real part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                      head_t2 = -head_t2;
                      tail_t2 = -tail_t2;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[0] = head_t1;
                      tail_temp2[0] = tail_t1;
                      /* imaginary part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[1] = head_t1;
                      tail_temp2[1] = tail_t1;
                    }

                  }
                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  impl::inner_type_t<TmpType_t> temp2r(head_temp2[0], tail_temp2[0]);
                  impl::inner_type_t<TmpType_t> temp2i(head_temp2[1], tail_temp2[1]);
                  TmpType_t temp2(temp2r, temp2i);
                  temp1 = temp1 - temp2;
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                  ix -= inc_intx;
                }                /* for j<n */

                /* if the diagonal entry is not equal to one, then divide Xj by
                   the entry */
                if (diag == blas_non_unit_diag) {
                  T_element = impl::Conj::func(T_i[j * incT + j * ldt * incT]);

                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  temp1 = impl::div(temp1, T_element);
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();

                }
                /* if (diag == blas_non_unit_diag) */
                head_intx[jx] = head_temp1[0];
                tail_intx[jx] = tail_temp1[0];
                head_intx[1 + jx] = head_temp1[1];
                tail_intx[1 + jx] = tail_temp1[1];

                jx -= inc_intx;
              }                        /* for j>=0 */
            } else {

              jx = (n - 1) * inc_intx;
              for (j = n - 1; j >= 0; j--) {

                /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
                   i=j+1 to n-1           */
                head_temp3[0] = head_intx[jx];
                head_temp3[1] = head_intx[1 + jx];
                tail_temp3[0] = tail_intx[jx];
                tail_temp3[1] = tail_intx[1 + jx];
                /* multiply by alpha */
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, std::imag(alpha_i));
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, std::imag(alpha_i));
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }


                ix = (n - 1) * inc_intx;
                for (i = n - 1; i >= j + 1; i--) {
                  T_element = T_i[j * incT + i * ldt * incT];

                  head_temp3[0] = head_intx[ix];
                  head_temp3[1] = head_intx[1 + ix];
                  tail_temp3[0] = tail_intx[ix];
                  tail_temp3[1] = tail_intx[1 + ix];
                  {
                    double cd[2];
                    cd[0] = (double) std::real(T_element);
                    cd[1] = (double) std::imag(T_element);
                    {
                      /* Compute complex-extra = complex-extra * complex-double. */
                      double head_a0, tail_a0;
                      double head_a1, tail_a1;
                      double head_t1, tail_t1;
                      double head_t2, tail_t2;
                      head_a0 = head_temp3[0];
                      tail_a0 = tail_temp3[0];
                      head_a1 = head_temp3[1];
                      tail_a1 = tail_temp3[1];
                      /* real part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                      head_t2 = -head_t2;
                      tail_t2 = -tail_t2;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[0] = head_t1;
                      tail_temp2[0] = tail_t1;
                      /* imaginary part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[1] = head_t1;
                      tail_temp2[1] = tail_t1;
                    }

                  }
                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  impl::inner_type_t<TmpType_t> temp2r(head_temp2[0], tail_temp2[0]);
                  impl::inner_type_t<TmpType_t> temp2i(head_temp2[1], tail_temp2[1]);
                  TmpType_t temp2(temp2r, temp2i);
                  temp1 = temp1 - temp2;
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                  ix -= inc_intx;
                }                /* for j<n */

                /* if the diagonal entry is not equal to one, then divide Xj by
                   the entry */
                if (diag == blas_non_unit_diag) {
                  T_element = T_i[j * incT + j * ldt * incT];


                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  temp1 = impl::div(temp1, T_element);
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();

                }
                /* if (diag == blas_non_unit_diag) */
                head_intx[jx] = head_temp1[0];
                tail_intx[jx] = tail_temp1[0];
                head_intx[1 + jx] = head_temp1[1];
                tail_intx[1 + jx] = tail_temp1[1];

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
                head_temp3[0] = head_intx[jx];
                head_temp3[1] = head_intx[1 + jx];
                tail_temp3[0] = tail_intx[jx];
                tail_temp3[1] = tail_intx[1 + jx];
                /* multiply by alpha */
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, std::imag(alpha_i));
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, std::imag(alpha_i));
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }


                ix = 0;
                for (i = 0; i < j; i++) {
                  T_element = impl::Conj::func(T_i[j * incT + i * ldt * incT]);
                  head_temp3[0] = head_intx[ix];
                  head_temp3[1] = head_intx[1 + ix];
                  tail_temp3[0] = tail_intx[ix];
                  tail_temp3[1] = tail_intx[1 + ix];
                  {
                    double cd[2];
                    cd[0] = (double) std::real(T_element);
                    cd[1] = (double) std::imag(T_element);
                    {
                      /* Compute complex-extra = complex-extra * complex-double. */
                      double head_a0, tail_a0;
                      double head_a1, tail_a1;
                      double head_t1, tail_t1;
                      double head_t2, tail_t2;
                      head_a0 = head_temp3[0];
                      tail_a0 = tail_temp3[0];
                      head_a1 = head_temp3[1];
                      tail_a1 = tail_temp3[1];
                      /* real part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                      head_t2 = -head_t2;
                      tail_t2 = -tail_t2;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[0] = head_t1;
                      tail_temp2[0] = tail_t1;
                      /* imaginary part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[1] = head_t1;
                      tail_temp2[1] = tail_t1;
                    }

                  }
                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  impl::inner_type_t<TmpType_t> temp2r(head_temp2[0], tail_temp2[0]);
                  impl::inner_type_t<TmpType_t> temp2i(head_temp2[1], tail_temp2[1]);
                  TmpType_t temp2(temp2r, temp2i);
                  temp1 = temp1 - temp2;
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                  ix += inc_intx;
                }                /* for i<j */

                /* if the diagonal entry is not equal to one, then divide Xj by
                   the entry */
                if (diag == blas_non_unit_diag) {
                  T_element = impl::Conj::func(T_i[j * incT + j * ldt * incT]);

                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  temp1 = impl::div(temp1, T_element);
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();

                }
                /* if (diag == blas_non_unit_diag) */
                head_intx[jx] = head_temp1[0];
                tail_intx[jx] = tail_temp1[0];
                head_intx[1 + jx] = head_temp1[1];
                tail_intx[1 + jx] = tail_temp1[1];
                jx += inc_intx;
              }                        /* for j<n */
            } else {

              jx = 0;
              for (j = 0; j < n; j++) {

                /* compute Xj = Xj - SUM Aij(or Aji) * Xi
                   i=j+1 to n-1           */
                head_temp3[0] = head_intx[jx];
                head_temp3[1] = head_intx[1 + jx];
                tail_temp3[0] = tail_intx[jx];
                tail_temp3[1] = tail_intx[1 + jx];
                /* multiply by alpha */
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, std::imag(alpha_i));
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, std::real(alpha_i));
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, std::imag(alpha_i));
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }


                ix = 0;
                for (i = 0; i < j; i++) {
                  T_element = T_i[j * incT + i * ldt * incT];

                  head_temp3[0] = head_intx[ix];
                  head_temp3[1] = head_intx[1 + ix];
                  tail_temp3[0] = tail_intx[ix];
                  tail_temp3[1] = tail_intx[1 + ix];
                  {
                    double cd[2];
                    cd[0] = (double) std::real(T_element);
                    cd[1] = (double) std::imag(T_element);
                    {
                      /* Compute complex-extra = complex-extra * complex-double. */
                      double head_a0, tail_a0;
                      double head_a1, tail_a1;
                      double head_t1, tail_t1;
                      double head_t2, tail_t2;
                      head_a0 = head_temp3[0];
                      tail_a0 = tail_temp3[0];
                      head_a1 = head_temp3[1];
                      tail_a1 = tail_temp3[1];
                      /* real part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                      head_t2 = -head_t2;
                      tail_t2 = -tail_t2;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[0] = head_t1;
                      tail_temp2[0] = tail_t1;
                      /* imaginary part */
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                      head_temp2[1] = head_t1;
                      tail_temp2[1] = tail_t1;
                    }

                  }
                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  impl::inner_type_t<TmpType_t> temp2r(head_temp2[0], tail_temp2[0]);
                  impl::inner_type_t<TmpType_t> temp2i(head_temp2[1], tail_temp2[1]);
                  TmpType_t temp2(temp2r, temp2i);
                  temp1 = temp1 - temp2;
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();
                  ix += inc_intx;
                }                /* for i<j */

                /* if the diagonal entry is not equal to one, then divide Xj by
                   the entry */
                if (diag == blas_non_unit_diag) {
                  T_element = T_i[j * incT + j * ldt * incT];


                  impl::inner_type_t<TmpType_t> temp1r(head_temp1[0], tail_temp1[0]);
                  impl::inner_type_t<TmpType_t> temp1i(head_temp1[1], tail_temp1[1]);
                  TmpType_t temp1(temp1r, temp1i);
                  temp1 = impl::div(temp1, T_element);
                  head_temp1[0] = std::real(temp1).head_();
                  head_temp1[1] = std::imag(temp1).head_();
                  tail_temp1[0] = std::real(temp1).tail_();
                  tail_temp1[1] = std::imag(temp1).tail_();

                }
                /* if (diag == blas_non_unit_diag) */
                head_intx[jx] = head_temp1[0];
                tail_intx[jx] = tail_temp1[0];
                head_intx[1 + jx] = head_temp1[1];
                tail_intx[1 + jx] = tail_temp1[1];
                jx += inc_intx;
              }                        /* for j<n */
            }
          }

          /* copy the final results from intx to x */
          ix = start_x;
          jx = 0;
          for (i = 0; i < n; i++) {
            head_temp1[0] = head_intx[jx];
            head_temp1[1] = head_intx[1 + jx];
            tail_temp1[0] = tail_intx[jx];
            tail_temp1[1] = tail_intx[1 + jx];
            x_i[ix] = T(head_temp1[0], head_temp1[1]);
            ix += incx;
            jx += inc_intx;
          }

          blas_free(head_intx);
          blas_free(tail_intx);
        }
      }
      FPU_FIX_STOP;
    }
    break;
  }
}
#endif



//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TRSV_HPP
