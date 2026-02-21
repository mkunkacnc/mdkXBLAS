#ifndef XBLAS_TBSV_HPP
#define XBLAS_TBSV_HPP

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
constexpr void tbsv(blas_order_type order,
                    blas_uplo_type uplo,
                    blas_trans_type trans,
                    blas_diag_type diag,
                    IdxType n,
                    IdxType k,
                    T alpha,
                    const A *t,
                    IdxType ldt,
                    T *x,
                    IdxType incx)
/*
 * Purpose
 * =======
 *
 * This routine solves :
 *
 *     x <- alpha * inverse(t) * x
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        column major, row major (blas_rowmajor, blas_colmajor)
 *
 * uplo   (input) blas_uplo_type
 *        upper, lower (blas_upper, blas_lower)
 *
 * trans  (input) blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) blas_diag_type
 *        unit, non unit (blas_unit_diag, blas_non_unit_diag)
 *
 * n      (input) IdxType
 *        the dimension of t
 *
 * k      (input) IdxType
 *        the number of subdiagonals/superdiagonals of t
 *
 * alpha  (input) T
 *
 * t      (input) const A*
 *        Triangular Banded matrix
 *
 * x      (input/output) T*
 *        Array of length n.
 *
 * incx   (input) IdxType
 *        The stride used to access components x[i].
 *
 */
{
  /* Routine name */
  static const char routine_name[] = "XBLAS::tbsv";

  using PrdType = impl::get_inner_type_t<A, T, TmpType>;

  FPU_FIX_DECL;

  IdxType i, j;
  IdxType xi;
  IdxType start_xi;
  IdxType incxi;
  IdxType Tij;
  IdxType dot_start, dot_start_inc1, dot_start_inc2, dot_inc;

  const A *t_i = t;
  T *x_i = x;
  T alpha_i = alpha;

  if (order != blas_rowmajor && order != blas_colmajor) {
    BLAS_error(routine_name, -1, order, 0);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, 0);
  }
  if ((trans != blas_trans) && (trans != blas_no_trans) &&
      (trans != blas_conj) && (trans != blas_conj_trans)) {
    BLAS_error(routine_name, -2, uplo, 0);
  }
  if (diag != blas_non_unit_diag && diag != blas_unit_diag) {
    BLAS_error(routine_name, -4, diag, 0);
  }
  if (n < 0) {
    BLAS_error(routine_name, -5, n, 0);
  }
  if (k >= n) {
    BLAS_error(routine_name, -6, k, 0);
  }
  if ((ldt < 1) || (ldt <= k)) {
    BLAS_error(routine_name, -9, ldt, 0);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -11, incx, 0);
  }

  if (n <= 0)
    return;

  incxi = incx;


  /* configuring the vector starting idx */
  if (incxi < 0) {
    start_xi = (1 - n) * incxi;
  } else {
    start_xi = 0;
  }

  /* if alpha is zero, then return x as a zero vector */
  if (alpha_i == T(0)) {
    xi = start_xi;
    for (i = 0; i < n; i++) {
      x_i[xi] = 0.0;
      xi += incxi;
    }
    return;
  }
  /* check to see if k=0.  if so, we can optimize somewhat */
  if (k == 0) {
    if ((alpha_i == T(1)) && (diag == blas_unit_diag)) {
      /* nothing to do */
      return;
    } else {
      /* just run the loops as is. */

    }
  }

  /* get index variables prepared */
  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (order == blas_rowmajor)) {
    dot_start = k;
  } else {
    dot_start = 0;
  }

  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (order == blas_rowmajor)) {
    dot_inc = 1;
    dot_start_inc1 = ldt - 1;
    dot_start_inc2 = ldt;
  } else {
    dot_inc = ldt - 1;
    dot_start_inc1 = 1;
    dot_start_inc2 = ldt;
  }

  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (uplo == blas_lower)) {
    /*start at the first element of x */
    /* substitution will proceed forwards (forwardsubstitution) */
  } else {
    /*start at the last element of x */
    /* substitution will proceed backwards (backsubstitution) */
    dot_inc = -dot_inc;
    dot_start_inc1 = -dot_start_inc1;
    dot_start_inc2 = -dot_start_inc2;
    dot_start = ldt * (n - 1) + k - dot_start;
    /*order of the following 2 statements matters! */
    start_xi = start_xi + (n - 1) * incxi;
    incxi = -incxi;
  }







  {

    {
      TmpType temp1;
      TmpType temp2;
      T x_elem;
      float T_element;






      /*loop 1 */
      xi = start_xi;
      for (j = 0; j < k; j++) {

        /* each time through loop, xi lands on next x to compute. */
        x_elem = x_i[xi];
        /* preform the multiplication -
           in this implementation we do not seperate the alpha = 1 case */
        temp1 = impl::mul<TmpType>(x_elem, alpha_i);

        xi = start_xi;

        Tij = dot_start;
        dot_start += dot_start_inc1;

        for (i = j; i > 0; i--) {
          T_element = t_i[Tij];

          x_elem = x_i[xi];
          temp2 = impl::mul<TmpType>(x_elem, T_element);
          temp1 = temp1 + (-temp2);
          xi += incxi;
          Tij += dot_inc;
        }                        /* for across row */


        /* if the diagonal entry is not equal to one, then divide Xj by
           the entry */
        if (diag == blas_non_unit_diag) {
          T_element = t_i[Tij];


          temp1 = temp1 / T_element;

        }
        /* if (diag == blas_non_unit_diag) */
        x_i[xi] = impl::to<T>(temp1);
        xi += incxi;
      }                                /* for j<k */
      /*end loop 1 */

      /*loop 2 continue without changing j to start */
      for (; j < n; j++) {

        /* each time through loop, xi lands on next x to compute. */
        x_elem = x_i[xi];
        temp1 = impl::mul<TmpType>(x_elem, alpha_i);

        xi = start_xi;
        start_xi += incxi;

        Tij = dot_start;
        dot_start += dot_start_inc2;

        for (i = k; i > 0; i--) {
          T_element = t_i[Tij];

          x_elem = x_i[xi];
          temp2 = impl::mul<TmpType>(x_elem, T_element);
          temp1 = temp1 + (-temp2);
          xi += incxi;
          Tij += dot_inc;
        }                        /* for across row */


        /* if the diagonal entry is not equal to one, then divide by
           the entry */
        if (diag == blas_non_unit_diag) {
          T_element = t_i[Tij];


          temp1 = temp1 / T_element;

        }
        /* if (diag == blas_non_unit_diag) */
        x_i[xi] = impl::to<T>(temp1);
        xi += incxi;
      }                                /* for j<n */


    }
  }
  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::tbsv */

//-----------------

inline
constexpr void tbsv_x(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    enum blas_trans_type trans,
                    enum blas_diag_type diag,
                    int n,
                    int k,
                    double alpha,
                    const float *t,
                    int ldt,
                    double *x,
                    int incx,
                    enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine solves :
 *
 *     x <- alpha * inverse(t) * x
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        column major, row major (blas_rowmajor, blas_colmajor)
 *
 * uplo   (input) enum blas_uplo_type
 *        upper, lower (blas_upper, blas_lower)
 *
 * trans  (input) enum blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) enum blas_diag_type
 *        unit, non unit (blas_unit_diag, blas_non_unit_diag)
 *
 * n      (input) int
 *        the dimension of t
 *
 * k      (input) int
 *        the number of subdiagonals/superdiagonals of t
 *
 * alpha  (input) double
 *
 * t      (input) const float*
 *        Triangular Banded matrix
 *
 * x      (input/output) double*
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
  /* Routine name */
  static const char routine_name[] = "BLAS_dtbsv_s_x";

  int i, j;                        /* used to keep track of loop counts */
  int xi;                        /* used to index vector x */
  int start_xi;                        /* used as the starting idx to vector x */
  int incxi;
  int Tij;                        /* index inside of Banded structure */
  int dot_start, dot_start_inc1, dot_start_inc2, dot_inc;

  const float *t_i = t;                /* internal matrix t */
  double *x_i = x;                /* internal x */
  double alpha_i = alpha;        /* internal alpha */

  if (order != blas_rowmajor && order != blas_colmajor) {
    BLAS_error(routine_name, -1, order, 0);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, 0);
  }
  if ((trans != blas_trans) && (trans != blas_no_trans) &&
      (trans != blas_conj) && (trans != blas_conj_trans)) {
    BLAS_error(routine_name, -2, uplo, 0);
  }
  if (diag != blas_non_unit_diag && diag != blas_unit_diag) {
    BLAS_error(routine_name, -4, diag, 0);
  }
  if (n < 0) {
    BLAS_error(routine_name, -5, n, 0);
  }
  if (k >= n) {
    BLAS_error(routine_name, -6, k, 0);
  }
  if ((ldt < 1) || (ldt <= k)) {
    BLAS_error(routine_name, -9, ldt, 0);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -11, incx, 0);
  }

  if (n <= 0)
    return;

  incxi = incx;


  /* configuring the vector starting idx */
  if (incxi < 0) {
    start_xi = (1 - n) * incxi;
  } else {
    start_xi = 0;
  }

  /* if alpha is zero, then return x as a zero vector */
  if (alpha_i == 0.0) {
    xi = start_xi;
    for (i = 0; i < n; i++) {
      x_i[xi] = 0.0;
      xi += incxi;
    }
    return;
  }
  /* check to see if k=0.  if so, we can optimize somewhat */
  if (k == 0) {
    if ((alpha_i == 1.0) && (diag == blas_unit_diag)) {
      /* nothing to do */
      return;
    } else {
      /* just run the loops as is. */
      /* must set prec to output. Ignore user input of prec */
      prec = blas_prec_double;
    }
  }

  /* get index variables prepared */
  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (order == blas_rowmajor)) {
    dot_start = k;
  } else {
    dot_start = 0;
  }

  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (order == blas_rowmajor)) {
    dot_inc = 1;
    dot_start_inc1 = ldt - 1;
    dot_start_inc2 = ldt;
  } else {
    dot_inc = ldt - 1;
    dot_start_inc1 = 1;
    dot_start_inc2 = ldt;
  }

  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (uplo == blas_lower)) {
    /*start at the first element of x */
    /* substitution will proceed forwards (forwardsubstitution) */
  } else {
    /*start at the last element of x */
    /* substitution will proceed backwards (backsubstitution) */
    dot_inc = -dot_inc;
    dot_start_inc1 = -dot_start_inc1;
    dot_start_inc2 = -dot_start_inc2;
    dot_start = ldt * (n - 1) + k - dot_start;
    /*order of the following 2 statements matters! */
    start_xi = start_xi + (n - 1) * incxi;
    incxi = -incxi;
  }







  switch (prec) {

  case blas_prec_single:
  case blas_prec_indigenous:
  case blas_prec_double:{
      {

        {
          double temp1;                /* temporary variable for calculations */
          double temp2;                /* temporary variable for calculations */
          double x_elem;
          float T_element;






          /*loop 1 */
          xi = start_xi;
          for (j = 0; j < k; j++) {

            /* each time through loop, xi lands on next x to compute. */
            x_elem = x_i[xi];
            /* preform the multiplication -
               in this implementation we do not seperate the alpha = 1 case */
            temp1 = x_elem * alpha_i;

            xi = start_xi;

            Tij = dot_start;
            dot_start += dot_start_inc1;

            for (i = j; i > 0; i--) {
              T_element = t_i[Tij];

              x_elem = x_i[xi];
              temp2 = x_elem * T_element;
              temp1 = temp1 + (-temp2);
              xi += incxi;
              Tij += dot_inc;
            }                        /* for across row */


            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[Tij];


              temp1 = temp1 / T_element;

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[xi] = temp1;
            xi += incxi;
          }                        /* for j<k */
          /*end loop 1 */

          /*loop 2 continue without changing j to start */
          for (; j < n; j++) {

            /* each time through loop, xi lands on next x to compute. */
            x_elem = x_i[xi];
            temp1 = x_elem * alpha_i;

            xi = start_xi;
            start_xi += incxi;

            Tij = dot_start;
            dot_start += dot_start_inc2;

            for (i = k; i > 0; i--) {
              T_element = t_i[Tij];

              x_elem = x_i[xi];
              temp2 = x_elem * T_element;
              temp1 = temp1 + (-temp2);
              xi += incxi;
              Tij += dot_inc;
            }                        /* for across row */


            /* if the diagonal entry is not equal to one, then divide by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[Tij];


              temp1 = temp1 / T_element;

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[xi] = temp1;
            xi += incxi;
          }                        /* for j<n */


        }
      }
      break;
    }

  case blas_prec_extra:{
      {

        {
          double head_temp1, tail_temp1;        /* temporary variable for calculations */
          double head_temp2, tail_temp2;        /* temporary variable for calculations */
          double head_temp3, tail_temp3;        /* temporary variable for calculations */
          double x_elem;
          float T_element;        /* temporary variable for an element of matrix T */

          int x_inti = 0, inc_x_inti = 1;
          int k_compare = k;        /*used for comparisons with x_inti */
          double *head_x_internal, *tail_x_internal;
          FPU_FIX_DECL;



          head_x_internal = (double *) blas_malloc(k * sizeof(double));
          tail_x_internal = (double *) blas_malloc(k * sizeof(double));
          if (k > 0 && (head_x_internal == NULL || tail_x_internal == NULL)) {
            BLAS_error("blas_malloc", 0, 0, "malloc failed.\n");
          }


          FPU_FIX_START;

          /*loop 1 */
          xi = start_xi;
          /* x_inti already initialized to 0 */
          for (j = 0; j < k; j++) {

            /* each time through loop, xi lands on next x to compute. */
            x_elem = x_i[xi];
            /* preform the multiplication -
               in this implementation we do not seperate the alpha = 1 case */
            compute_doubledouble_eq_double_mul_double(&head_temp1, &tail_temp1, x_elem, alpha_i);

            Tij = dot_start;
            dot_start += dot_start_inc1;

            /*start loop buffer over in loop 1 */
            x_inti = 0;
            for (i = j; i > 0; i--) {
              T_element = t_i[Tij];

              head_temp3 = head_x_internal[x_inti];
              tail_temp3 = tail_x_internal[x_inti];
              {
                double dt = (double) T_element;
                compute_doubledouble_eq_doubledouble_mul_double(&head_temp2, &tail_temp2, head_temp3, tail_temp3, dt);
              }
              {
                double head_bt, tail_bt;
                head_bt = -head_temp2;
                tail_bt = -tail_temp2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_temp1, &tail_temp1, head_temp1, tail_temp1, head_bt, tail_bt);
              }
              x_inti += inc_x_inti;
              Tij += dot_inc;
            }                        /* for across row */


            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[Tij];


              {
                double dt = (double) T_element;
                compute_doubledouble_eq_doubledouble_div_double(&head_temp1, &tail_temp1, head_temp1, tail_temp1, dt);
              }

            }

            /* if (diag == blas_non_unit_diag) */
            /* place internal precision result in internal buffer */
            head_x_internal[x_inti] = head_temp1;
            tail_x_internal[x_inti] = tail_temp1;

            /* place result x in same place as got x this loop */
            x_i[xi] = head_temp1;
            xi += incxi;
          }                        /* for j<k */
          /*end loop 1 */


          /* loop2 ***************************** */
          x_inti = 0;
          /*loop 2 continue without changing j to start */
          for (; j < n; j++) {

            /* each time through loop, xi lands on next x to compute. */
            x_elem = x_i[xi];
            compute_doubledouble_eq_double_mul_double(&head_temp1, &tail_temp1, x_elem, alpha_i);


            Tij = dot_start;
            dot_start += dot_start_inc2;

            for (i = k; i > 0 && (x_inti < k_compare); i--) {
              T_element = t_i[Tij];

              head_temp3 = head_x_internal[x_inti];
              tail_temp3 = tail_x_internal[x_inti];
              {
                double dt = (double) T_element;
                compute_doubledouble_eq_doubledouble_mul_double(&head_temp2, &tail_temp2, head_temp3, tail_temp3, dt);
              }
              {
                double head_bt, tail_bt;
                head_bt = -head_temp2;
                tail_bt = -tail_temp2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_temp1, &tail_temp1, head_temp1, tail_temp1, head_bt, tail_bt);
              }
              x_inti += inc_x_inti;
              Tij += dot_inc;
            }                        /* for across row */
            /*reset index to internal storage loop buffer. */
            x_inti = 0;
            for (; i > 0; i--) {
              T_element = t_i[Tij];

              head_temp3 = head_x_internal[x_inti];
              tail_temp3 = tail_x_internal[x_inti];
              {
                double dt = (double) T_element;
                compute_doubledouble_eq_doubledouble_mul_double(&head_temp2, &tail_temp2, head_temp3, tail_temp3, dt);
              }
              {
                double head_bt, tail_bt;
                head_bt = -head_temp2;
                tail_bt = -tail_temp2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_temp1, &tail_temp1, head_temp1, tail_temp1, head_bt, tail_bt);
              }
              x_inti += inc_x_inti;
              Tij += dot_inc;
            }                        /* for across row */


            /* if the diagonal entry is not equal to one, then divide by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[Tij];


              {
                double dt = (double) T_element;
                compute_doubledouble_eq_doubledouble_div_double(&head_temp1, &tail_temp1, head_temp1, tail_temp1, dt);
              }

            }

            /* if (diag == blas_non_unit_diag) */
            /* place internal precision result in internal buffer */
            head_x_internal[x_inti] = head_temp1;
            tail_x_internal[x_inti] = tail_temp1;
            x_inti += inc_x_inti;
            if (x_inti >= k_compare)
              x_inti = 0;

            /* place result x in same place as got x this loop */
            x_i[xi] = head_temp1;
            xi += incxi;
          }                        /* for j<n */

          FPU_FIX_STOP;

          blas_free(head_x_internal);
          blas_free(tail_x_internal);
        }
      }
      break;
    }

  default:
    BLAS_error(routine_name, -13, prec, 0);
    break;
  }                                /* end prec switch */
} /* end XBLAS::tbsv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TBSV_HPP
