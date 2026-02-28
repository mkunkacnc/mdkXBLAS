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
      (trans != static_cast<blas_trans_type>(blas_conj)) && (trans != blas_conj_trans)) {
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
      x_i[xi] = T(0);
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
    /* substitution will proceed forwards (forward substitution) */
  } else {
    /*start at the last element of x */
    /* substitution will proceed backwards (back substitution) */
    dot_inc = -dot_inc;
    dot_start_inc1 = -dot_start_inc1;
    dot_start_inc2 = -dot_start_inc2;
    dot_start = ldt * (n - 1) + k - dot_start;
    /*order of the following 2 statements matters! */
    start_xi = start_xi + (n - 1) * incxi;
    incxi = -incxi;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if constexpr (impl::is_complex_v<A>) {
    if constexpr (sizeof(TmpType) > sizeof(T)) {
      TmpType temp1;        /* temporary variable for calculations */
      TmpType temp2;        /* temporary variable for calculations */
      TmpType temp3;        /* temporary variable for calculations */
      T x_elem;
      A T_element;        /* temporary variable for an element of matrix T */

      IdxType x_inti = 0, inc_x_inti = 1;
      IdxType k_compare = k;        /*used for comparisons with x_inti */
      TmpType *x_internal;

      x_internal = new(std::nothrow) TmpType[k]; // *) blas_malloc(k * sizeof(TmpType));
      if (k > 0 && x_internal == NULL) {
        BLAS_error("blas_malloc", 0, 0, "temp allocation failed.\n");
      }

      if ((trans == static_cast<blas_trans_type>(blas_conj)) || (trans == blas_conj_trans)) {
        /* conjugated */
        /*loop 1 */
        xi = start_xi;
        /* x_inti already initialized to 0 */
        for (j = 0; j < k; j++) {

          /* each time through loop, xi lands on next x to compute. */
          x_elem = x_i[xi];
          /* preform the multiplication -
             in this implementation we do not separate the alpha = 1 case */
          temp1 = impl::mul<TmpType>(x_elem, alpha_i);

          Tij = dot_start;
          dot_start += dot_start_inc1;

          /*start loop buffer over in loop 1 */
          x_inti = 0;
          for (i = j; i > 0; i--) {
            T_element = impl::Conj::func(t_i[Tij]);
            temp3 = x_internal[x_inti];
            impl::mul<TmpType>(x_elem, T_element);
            temp1 = temp1 - temp2;
            x_inti += inc_x_inti;
            Tij += dot_inc;
          }                        /* for across row */


          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = impl::Conj::func(t_i[Tij]);

            if constexpr (std::is_same_v<impl::inner_type_t<A>, double>) {
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double_double t;
                double_double t1;
                double_double t2;
                double_double q[2];

                eps = pow(2.0, -53.0);        /* double precision */
                un = pow(2.0, -1022.0);
                ov = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps)) * 2.0 */
                eps1 = pow(2.0, -104.0);        /* extra precision */
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16.0;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  s = 2.0 / (eps1 * eps1);
                  temp1 *= s;
                  S = S / s;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  s = 2.0 / (eps * eps);
                  T_element *= s;
                  S = S * s;
                }

                /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = impl::mul<double_double>(r, std::imag(T_element));
                  //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, T_element[1]);
                  t = t + std::real(T_element);
                  //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, T_element[0]);
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  //tail_q[0] = tail_t2;
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  {
                    t2 = t1 - t2;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
                  }                /* b - a*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = impl::mul<double_double>(r, std::real(T_element));
                  //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, T_element[0]);
                  t = t + std::imag(T_element);
                  //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, T_element[1]);
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  {
                    t2 = t2 - t1;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
                  }                /* -a + b*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                }
                /* Scale back */
                if (S == 1.0) {
                  temp1 = std::complex<double_double>(q[0], q[1]);
                } else {
                  /* Compute complex-extra = complex-extra * real. */
                  double_double a0;
                  double_double a1;
                  double_double t;
                  a0 = q[0];
                  a1 = q[1];
                  temp1 = std::complex<double_double>(a0 * S, a1 * S);
                }
              } else {
                // scaled division to avoid overflow.
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -53.0);
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  t = 2.0 / (eps1 * eps1);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              }
            } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double_double t;
                double_double t1;
                double_double t2;
                double_double q[2];

                eps = pow(2.0, -24.0);        /* single precision */
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -104.0);        /* extra precision */
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16.0;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  s = 2.0 / (eps1 * eps1);
                  temp1 *= s;
                  S = S / s;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  s = 2.0 / (eps * eps);
                  T_element *= s;
                  S = S * s;
                }

                /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  {
                    double dt = (double) std::imag(T_element);
                    t = impl::mul<double_double>(r, dt);
                    //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
                  }
                  {
                    double dt = (double) std::real(T_element);
                    t = t + dt;
                    //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
                  }
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);

                  t1 = std::real(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  {
                    t2 = t1 - t2;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
                  }                /* b - a*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  {
                    double dt = std::real(T_element);
                    t = impl::mul<double_double>(r, dt);
                    //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
                  }
                  {
                    double dt = (double) std::imag(T_element);
                    t = t + dt;
                    //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
                  }
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  {
                    t2 = t2 - t1;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
                  }                /* -a + b*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                }
                /* Scale back */
                if (S == 1.0) {
                  temp1 = std::complex<double_double>(q[0], q[1]);
                } else {
                  /* Compute complex-extra = complex-extra * real. */
                  double_double a0;
                  double_double a1;
                  double_double t;
                  a0 = q[0];
                  a1 = q[1];
                  temp1 = std::complex<double_double>(a0 * S, a1 * S);
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, S);
                  //head_temp1[0] = head_t;
                  //tail_temp1[0] = tail_t;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, S);
                  //head_temp1[1] = head_t;
                  //tail_temp1[1] = tail_t;
                }
              } else if constexpr (std::is_same_v<TmpType, std::complex<double>>) {
                // scaled division to avoid overflow.
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -53.0);
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
                abs_a = fabs(std::real(temp1));
                abs_b = fabs(std::imag(temp1));
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  t = 2.0 / (eps1 * eps1);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
                double S = 1.0, eps, ov, un;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un / eps * 2) {        /* scale up a, b */
                  t = 2.0 / (eps * eps);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              } else {
                double S = 1.0, eps, ov, un;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un / eps * 2) {        /* scale up a, b */
                  t = 2.0 / (eps * eps);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              }
            } else {
              temp1 = impl::div(temp1, T_element);
            }
          }

          /* if (diag == blas_non_unit_diag) */
          /* place internal precision result in internal buffer */
          x_internal[x_inti] = temp1;

          /* place result x in same place as got x this loop */
          x_i[xi] = impl::to<T>(temp1);
          xi += incxi;
        }                        /* for j<k */
        /*end loop 1 */

        /* loop2 ***************************** */
        x_inti = 0;
        /*loop 2 continue without changing j to start */
        for (; j < n; j++) {

          /* each time through loop, xi lands on next x to compute. */
          x_elem = x_i[xi];
          temp1 = impl::mul<TmpType>(x_elem, alpha_i);

          Tij = dot_start;
          dot_start += dot_start_inc2;

          for (i = k; i > 0 && (x_inti < k_compare); i--) {
            T_element = impl::Conj::func(t_i[Tij]);
            temp3 = x_internal[x_inti];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            x_inti += inc_x_inti;
            Tij += dot_inc;
          }                        /* for across row */
          /*reset index to internal storage loop buffer. */
          x_inti = 0;
          for (; i > 0; i--) {
            T_element = impl::Conj::func(t_i[Tij]);
            temp3 = x_internal[x_inti];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            x_inti += inc_x_inti;
            Tij += dot_inc;
          }                        /* for across row */


          /* if the diagonal entry is not equal to one, then divide by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = impl::Conj::func(t_i[Tij]);

            if constexpr (std::is_same_v<impl::inner_type_t<A>, double>) {
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double_double t;
                double_double t1;
                double_double t2;
                double_double q[2];

                eps = pow(2.0, -53.0);        /* double precision */
                un = pow(2.0, -1022.0);
                ov = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps)) * 2.0 */
                eps1 = pow(2.0, -104.0);        /* extra precision */
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16.0;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  s = 2.0 / (eps1 * eps1);
                  temp1 *= s;
                  S = S / s;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  s = 2.0 / (eps * eps);
                  T_element *= s;
                  S = S * s;
                }

                /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = impl::mul<double_double>(r, std::imag(T_element));
                  //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, T_element[1]);
                  t = t + std::real(T_element);
                  //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, T_element[0]);
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  //tail_q[0] = tail_t2;
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  {
                    t2 = t1 - t2;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
                  }                /* b - a*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = impl::mul<double_double>(r, std::real(T_element));
                  //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, T_element[0]);
                  t = t + std::imag(T_element);
                  //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, T_element[1]);
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  {
                    t2 = t2 - t1;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
                  }                /* -a + b*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                }
                /* Scale back */
                if (S == 1.0) {
                  temp1 = std::complex<double_double>(q[0], q[1]);
                } else {
                  /* Compute complex-extra = complex-extra * real. */
                  double_double a0;
                  double_double a1;
                  double_double t;
                  a0 = q[0];
                  a1 = q[1];
                  temp1 = std::complex<double_double>(a0 * S, a1 * S);
                }
              } else {
                // scaled division to avoid overflow.
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -53.0);
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  t = 2.0 / (eps1 * eps1);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              }
            } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double_double t;
                double_double t1;
                double_double t2;
                double_double q[2];

                eps = pow(2.0, -24.0);        /* single precision */
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -104.0);        /* extra precision */
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16.0;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  s = 2.0 / (eps1 * eps1);
                  temp1 *= s;
                  S = S / s;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  s = 2.0 / (eps * eps);
                  T_element *= s;
                  S = S * s;
                }

                /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  {
                    double dt = (double) std::imag(T_element);
                    t = impl::mul<double_double>(r, dt);
                    //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
                  }
                  {
                    double dt = (double) std::real(T_element);
                    t = t + dt;
                    //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
                  }
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);

                  t1 = std::real(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  {
                    t2 = t1 - t2;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
                  }                /* b - a*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  {
                    double dt = std::real(T_element);
                    t = impl::mul<double_double>(r, dt);
                    //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
                  }
                  {
                    double dt = (double) std::imag(T_element);
                    t = t + dt;
                    //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
                  }
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  {
                    t2 = t2 - t1;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
                  }                /* -a + b*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                }
                /* Scale back */
                if (S == 1.0) {
                  temp1 = std::complex<double_double>(q[0], q[1]);
                } else {
                  /* Compute complex-extra = complex-extra * real. */
                  double_double a0;
                  double_double a1;
                  double_double t;
                  a0 = q[0];
                  a1 = q[1];
                  temp1 = std::complex<double_double>(a0 * S, a1 * S);
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, S);
                  //head_temp1[0] = head_t;
                  //tail_temp1[0] = tail_t;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, S);
                  //head_temp1[1] = head_t;
                  //tail_temp1[1] = tail_t;
                }
              } else if constexpr (std::is_same_v<TmpType, std::complex<double>>) {
                // scaled division to avoid overflow.
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -53.0);
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
                abs_a = fabs(std::real(temp1));
                abs_b = fabs(std::imag(temp1));
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  t = 2.0 / (eps1 * eps1);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
                double S = 1.0, eps, ov, un;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un / eps * 2) {        /* scale up a, b */
                  t = 2.0 / (eps * eps);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              } else {
                double S = 1.0, eps, ov, un;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un / eps * 2) {        /* scale up a, b */
                  t = 2.0 / (eps * eps);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              }
            } else {
              temp1 = impl::div(temp1, T_element);
            }
          }

          /* if (diag == blas_non_unit_diag) */
          /* place internal precision result in internal buffer */
          x_internal[x_inti] = temp1;
          x_inti += inc_x_inti;
          if (x_inti >= k_compare)
            x_inti = 0;

          /* place result x in same place as got x this loop */
          x_i[xi] = impl::to<T>(temp1);
          xi += incxi;
        }                        /* for j<n */

      } else {
        /* not conjugated */
        /*loop 1 */
        xi = start_xi;
        /* x_inti already initialized to 0 */
        for (j = 0; j < k; j++) {

          /* each time through loop, xi lands on next x to compute. */
          x_elem = x_i[xi];
          /* preform the multiplication -
             in this implementation we do not separate the alpha = 1 case */
          temp1 = impl::mul<TmpType>(x_elem, alpha_i);

          Tij = dot_start;
          dot_start += dot_start_inc1;

          /*start loop buffer over in loop 1 */
          x_inti = 0;
          for (i = j; i > 0; i--) {
            T_element = t_i[Tij];

            temp3 = x_internal[x_inti];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            x_inti += inc_x_inti;
            Tij += dot_inc;
          }                        /* for across row */


          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[Tij];

            if constexpr (std::is_same_v<impl::inner_type_t<A>, double>) {
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double_double t;
                double_double t1;
                double_double t2;
                double_double q[2];

                eps = pow(2.0, -53.0);        /* double precision */
                un = pow(2.0, -1022.0);
                ov = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps)) * 2.0 */
                eps1 = pow(2.0, -104.0);        /* extra precision */
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16.0;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  s = 2.0 / (eps1 * eps1);
                  temp1 *= s;
                  S = S / s;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  s = 2.0 / (eps * eps);
                  T_element *= s;
                  S = S * s;
                }

                /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = impl::mul<double_double>(r, std::imag(T_element));
                  //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, T_element[1]);
                  t = t + std::real(T_element);
                  //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, T_element[0]);
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  //tail_q[0] = tail_t2;
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  {
                    t2 = t1 - t2;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
                  }                /* b - a*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = impl::mul<double_double>(r, std::real(T_element));
                  //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, T_element[0]);
                  t = t + std::imag(T_element);
                  //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, T_element[1]);
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  {
                    t2 = t2 - t1;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
                  }                /* -a + b*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                }
                /* Scale back */
                if (S == 1.0) {
                  temp1 = std::complex<double_double>(q[0], q[1]);
                } else {
                  /* Compute complex-extra = complex-extra * real. */
                  double_double a0;
                  double_double a1;
                  double_double t;
                  a0 = q[0];
                  a1 = q[1];
                  temp1 = std::complex<double_double>(a0 * S, a1 * S);
                }
              } else {
                // scaled division to avoid overflow.
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -53.0);
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  t = 2.0 / (eps1 * eps1);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              }
            } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double_double t;
                double_double t1;
                double_double t2;
                double_double q[2];

                eps = pow(2.0, -24.0);        /* single precision */
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -104.0);        /* extra precision */
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16.0;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  s = 2.0 / (eps1 * eps1);
                  temp1 *= s;
                  S = S / s;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  s = 2.0 / (eps * eps);
                  T_element *= s;
                  S = S * s;
                }

                /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  {
                    double dt = (double) std::imag(T_element);
                    t = impl::mul<double_double>(r, dt);
                    //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
                  }
                  {
                    double dt = (double) std::real(T_element);
                    t = t + dt;
                    //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
                  }
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);

                  t1 = std::real(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  {
                    t2 = t1 - t2;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
                  }                /* b - a*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  {
                    double dt = std::real(T_element);
                    t = impl::mul<double_double>(r, dt);
                    //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
                  }
                  {
                    double dt = (double) std::imag(T_element);
                    t = t + dt;
                    //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
                  }
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  {
                    t2 = t2 - t1;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
                  }                /* -a + b*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                }
                /* Scale back */
                if (S == 1.0) {
                  temp1 = std::complex<double_double>(q[0], q[1]);
                } else {
                  /* Compute complex-extra = complex-extra * real. */
                  double_double a0;
                  double_double a1;
                  double_double t;
                  a0 = q[0];
                  a1 = q[1];
                  temp1 = std::complex<double_double>(a0 * S, a1 * S);
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, S);
                  //head_temp1[0] = head_t;
                  //tail_temp1[0] = tail_t;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, S);
                  //head_temp1[1] = head_t;
                  //tail_temp1[1] = tail_t;
                }
              } else if constexpr (std::is_same_v<TmpType, std::complex<double>>) {
                // scaled division to avoid overflow.
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -53.0);
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
                abs_a = fabs(std::real(temp1));
                abs_b = fabs(std::imag(temp1));
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  t = 2.0 / (eps1 * eps1);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
                double S = 1.0, eps, ov, un;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un / eps * 2) {        /* scale up a, b */
                  t = 2.0 / (eps * eps);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              } else {
                double S = 1.0, eps, ov, un;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un / eps * 2) {        /* scale up a, b */
                  t = 2.0 / (eps * eps);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              }
            } else {
              temp1 = impl::div(temp1, T_element);
            }
          }

          /* if (diag == blas_non_unit_diag) */
          /* place internal precision result in internal buffer */
          x_internal[x_inti] = temp1;

          /* place result x in same place as got x this loop */
          x_i[xi] = impl::to<T>(temp1);
          xi += incxi;
        }                        /* for j<k */
        /*end loop 1 */


        /* loop2 ***************************** */
        x_inti = 0;
        /*loop 2 continue without changing j to start */
        for (; j < n; j++) {

          /* each time through loop, xi lands on next x to compute. */
          x_elem = x_i[xi];
          temp1 = impl::mul<TmpType>(x_elem, alpha_i);

          Tij = dot_start;
          dot_start += dot_start_inc2;

          for (i = k; i > 0 && (x_inti < k_compare); i--) {
            T_element = t_i[Tij];
            temp3 = x_internal[x_inti];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            x_inti += inc_x_inti;
            Tij += dot_inc;
          }                        /* for across row */
          /*reset index to internal storage loop buffer. */
          x_inti = 0;
          for (; i > 0; i--) {
            T_element = t_i[Tij];
            temp3 = x_internal[x_inti];
            temp2 = impl::mul<TmpType>(temp3, T_element);
            temp1 = temp1 - temp2;
            x_inti += inc_x_inti;
            Tij += dot_inc;
          }                        /* for across row */


          /* if the diagonal entry is not equal to one, then divide by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[Tij];

            if constexpr (std::is_same_v<impl::inner_type_t<A>, double>) {
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double_double t;
                double_double t1;
                double_double t2;
                double_double q[2];

                eps = pow(2.0, -53.0);        /* double precision */
                un = pow(2.0, -1022.0);
                ov = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps)) * 2.0 */
                eps1 = pow(2.0, -104.0);        /* extra precision */
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16.0;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  s = 2.0 / (eps1 * eps1);
                  temp1 *= s;
                  S = S / s;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  s = 2.0 / (eps * eps);
                  T_element *= s;
                  S = S * s;
                }

                /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = impl::mul<double_double>(r, std::imag(T_element));
                  //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, T_element[1]);
                  t = t + std::real(T_element);
                  //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, T_element[0]);
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  //tail_q[0] = tail_t2;
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  {
                    t2 = t1 - t2;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
                  }                /* b - a*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = impl::mul<double_double>(r, std::real(T_element));
                  //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, T_element[0]);
                  t = t + std::imag(T_element);
                  //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, T_element[1]);
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  {
                    t2 = t2 - t1;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
                  }                /* -a + b*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                }
                /* Scale back */
                if (S == 1.0) {
                  temp1 = std::complex<double_double>(q[0], q[1]);
                } else {
                  /* Compute complex-extra = complex-extra * real. */
                  double_double a0;
                  double_double a1;
                  double_double t;
                  a0 = q[0];
                  a1 = q[1];
                  temp1 = std::complex<double_double>(a0 * S, a1 * S);
                }
              } else {
                // scaled division to avoid overflow.
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -53.0);
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  t = 2.0 / (eps1 * eps1);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              }
            } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double s;
                double r;
                double_double t;
                double_double t1;
                double_double t2;
                double_double q[2];

                eps = pow(2.0, -24.0);        /* single precision */
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -104.0);        /* extra precision */
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16.0;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  s = 2.0 / (eps1 * eps1);
                  temp1 *= s;
                  S = S / s;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  s = 2.0 / (eps * eps);
                  T_element *= s;
                  S = S * s;
                }

                /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  {
                    double dt = (double) std::imag(T_element);
                    t = impl::mul<double_double>(r, dt);
                    //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
                  }
                  {
                    double dt = (double) std::real(T_element);
                    t = t + dt;
                    //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
                  }
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);

                  t1 = std::real(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  {
                    t2 = t1 - t2;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
                  }                /* b - a*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  {
                    double dt = std::real(T_element);
                    t = impl::mul<double_double>(r, dt);
                    //compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
                  }
                  {
                    double dt = (double) std::imag(T_element);
                    t = t + dt;
                    //compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
                  }
                  t1 = std::real(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::imag(temp1);
                  t2 = t2 + t1;
                  //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[0] = t2;
                  t1 = std::imag(temp1);
                  t2 = t1 * r;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
                  t1 = std::real(temp1);
                  {
                    t2 = t2 - t1;
                    //compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
                  }                /* -a + b*r */
                  t2 = t2 / t;
                  //compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
                  q[1] = t2;
                }
                /* Scale back */
                if (S == 1.0) {
                  temp1 = std::complex<double_double>(q[0], q[1]);
                } else {
                  /* Compute complex-extra = complex-extra * real. */
                  double_double a0;
                  double_double a1;
                  double_double t;
                  a0 = q[0];
                  a1 = q[1];
                  temp1 = std::complex<double_double>(a0 * S, a1 * S);
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, S);
                  //head_temp1[0] = head_t;
                  //tail_temp1[0] = tail_t;
                  //compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, S);
                  //head_temp1[1] = head_t;
                  //tail_temp1[1] = tail_t;
                }
              } else if constexpr (std::is_same_v<TmpType, std::complex<double>>) {
                // scaled division to avoid overflow.
                double S = 1.0, eps, ov, un, eps1, ov1, un1;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                eps1 = pow(2.0, -53.0);
                un1 = pow(2.0, -1022.0);
                ov1 = 1.79769313486231571e+308;
                /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
                abs_a = fabs(std::real(temp1));
                abs_b = fabs(std::imag(temp1));
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov1 / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                  t = 2.0 / (eps1 * eps1);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
                double S = 1.0, eps, ov, un;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un / eps * 2) {        /* scale up a, b */
                  t = 2.0 / (eps * eps);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              } else {
                double S = 1.0, eps, ov, un;
                double abs_a, abs_b, abs_c, abs_d, ab, cd;
                double r;
                double t;
                double q[2];

                eps = pow(2.0, -24.0);
                un = pow(2.0, -126.0);
                ov = pow(2.0, 128.0) * (1 - eps);
                if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                  abs_a = fabs(std::real(temp1).to_double());
                  abs_b = fabs(std::imag(temp1).to_double());
                } else {
                  abs_a = fabs(std::real(temp1));
                  abs_b = fabs(std::imag(temp1));
                }
                abs_c = fabs(static_cast<double>(std::real(T_element)));
                abs_d = fabs(static_cast<double>(std::imag(T_element)));
                ab = std::max(abs_a, abs_b);
                cd = std::max(abs_c, abs_d);

                /* Scaling */
                if (ab > ov / 16) {        /* scale down a, b */
                  temp1 /= 16;
                  S = S * 16;
                }
                if (cd > ov / 16) {        /* scale down c, d */
                  T_element /= 16;
                  S = S / 16;
                }
                if (ab < un / eps * 2) {        /* scale up a, b */
                  t = 2.0 / (eps * eps);
                  temp1 *= t;
                  S = S / t;
                }
                if (cd < un / eps * 2) {        /* scale up c, d */
                  t = 2.0 / (eps * eps);
                  T_element *= t;
                  S = S * t;
                }

                /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
                if (abs_c > abs_d) {
                  r = std::imag(T_element) / std::real(T_element);
                  t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                  q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                  q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
                } else {
                  r = std::real(T_element) / std::imag(T_element);
                  t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                  q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                  q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
                }
                /* Scale back */
                temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
              }
            } else {
              temp1 = impl::div(temp1, T_element);
            }
          }

          /* if (diag == blas_non_unit_diag) */
          /* place internal precision result in internal buffer */
          x_internal[x_inti] = temp1;
          x_inti += inc_x_inti;
          if (x_inti >= k_compare)
            x_inti = 0;

          /* place result x in same place as got x this loop */
          x_i[xi] = impl::to<T>(temp1);
          xi += incxi;
        }                        /* for j<n */
      }

      delete[] x_internal;
      //blas_free(x_internal);

    } else {
      TmpType temp1;
      TmpType temp2;
      T x_elem;
      A T_element;

      if (trans == static_cast<blas_trans_type>(blas_conj) || trans == blas_conj_trans) {
        /* conjugated */

        /*loop 1 */
        xi = start_xi;
        for (j = 0; j < k; j++) {

          /* each time through loop, xi lands on next x to compute. */
          x_elem = x_i[xi];
          /* preform the multiplication -
             in this implementation we do not separate the alpha = 1 case */
          temp1 = impl::mul<TmpType>(x_elem, alpha_i);

          xi = start_xi;

          Tij = dot_start;
          dot_start += dot_start_inc1;

          for (i = j; i > 0; i--) {
            T_element = impl::Conj::func(t_i[Tij]);
            x_elem = x_i[xi];
            temp2 = impl::mul<TmpType>(x_elem, T_element);
            temp1 = temp1 - temp2;
            xi += incxi;
            Tij += dot_inc;
          }                        /* for across row */


          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = impl::Conj::func(t_i[Tij]);

            if constexpr (std::is_same_v<impl::inner_type_t<A>, double>) {
              // scaled division to avoid overflow.
              double S = 1.0, eps, ov, un, eps1, ov1, un1;
              double abs_a, abs_b, abs_c, abs_d, ab, cd;
              double r;
              double t;
              double q[2];

              eps = pow(2.0, -24.0);
              un = pow(2.0, -126.0);
              ov = pow(2.0, 128.0) * (1 - eps);
              eps1 = pow(2.0, -53.0);
              un1 = pow(2.0, -1022.0);
              ov1 = 1.79769313486231571e+308;
              /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
              abs_a = fabs(std::real(temp1));
              abs_b = fabs(std::imag(temp1));
              abs_c = fabs(static_cast<double>(std::real(T_element)));
              abs_d = fabs(static_cast<double>(std::imag(T_element)));
              ab = std::max(abs_a, abs_b);
              cd = std::max(abs_c, abs_d);

              /* Scaling */
              if (ab > ov1 / 16) {        /* scale down a, b */
                temp1 /= 16;
                S = S * 16;
              }
              if (cd > ov / 16) {        /* scale down c, d */
                T_element /= 16;
                S = S / 16;
              }
              if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                t = 2.0 / (eps1 * eps1);
                temp1 *= t;
                S = S / t;
              }
              if (cd < un / eps * 2) {        /* scale up c, d */
                t = 2.0 / (eps * eps);
                T_element *= t;
                S = S * t;
              }

              /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
              if (abs_c > abs_d) {
                r = std::imag(T_element) / std::real(T_element);
                t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
              } else {
                r = std::real(T_element) / std::imag(T_element);
                t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
              }
              /* Scale back */
              temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
            } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
              double S = 1.0, eps, ov, un;
              double abs_a, abs_b, abs_c, abs_d, ab, cd;
              double r;
              double t;
              double q[2];

              eps = pow(2.0, -24.0);
              un = pow(2.0, -126.0);
              ov = pow(2.0, 128.0) * (1 - eps);
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
              } else {
                abs_a = fabs(std::real(temp1));
                abs_b = fabs(std::imag(temp1));
              }
              abs_c = fabs(static_cast<double>(std::real(T_element)));
              abs_d = fabs(static_cast<double>(std::imag(T_element)));
              ab = std::max(abs_a, abs_b);
              cd = std::max(abs_c, abs_d);

              /* Scaling */
              if (ab > ov / 16) {        /* scale down a, b */
                temp1 /= 16;
                S = S * 16;
              }
              if (cd > ov / 16) {        /* scale down c, d */
                T_element /= 16;
                S = S / 16;
              }
              if (ab < un / eps * 2) {        /* scale up a, b */
                t = 2.0 / (eps * eps);
                temp1 *= t;
                S = S / t;
              }
              if (cd < un / eps * 2) {        /* scale up c, d */
                t = 2.0 / (eps * eps);
                T_element *= t;
                S = S * t;
              }

              /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
              if (abs_c > abs_d) {
                r = std::imag(T_element) / std::real(T_element);
                t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
              } else {
                r = std::real(T_element) / std::imag(T_element);
                t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
              }
              /* Scale back */
              temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
            } else {
              temp1 = impl::div(temp1, T_element);
            }
          }
          /* if (diag == blas_non_unit_diag) */
          x_i[xi] = impl::to<T>(temp1);
          xi += incxi;
        }                        /* for j<k */
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
            T_element = impl::Conj::func(t_i[Tij]);
            x_elem = x_i[xi];
            temp2 = impl::mul<TmpType>(x_elem, T_element);
            temp1 = temp1 - temp2;
            xi += incxi;
            Tij += dot_inc;
          }                        /* for across row */


          /* if the diagonal entry is not equal to one, then divide by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = impl::Conj::func(t_i[Tij]);

            if constexpr (std::is_same_v<impl::inner_type_t<A>, double>) {
              // scaled division to avoid overflow.
              double S = 1.0, eps, ov, un, eps1, ov1, un1;
              double abs_a, abs_b, abs_c, abs_d, ab, cd;
              double r;
              double t;
              double q[2];

              eps = pow(2.0, -24.0);
              un = pow(2.0, -126.0);
              ov = pow(2.0, 128.0) * (1 - eps);
              eps1 = pow(2.0, -53.0);
              un1 = pow(2.0, -1022.0);
              ov1 = 1.79769313486231571e+308;
              /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
              abs_a = fabs(std::real(temp1));
              abs_b = fabs(std::imag(temp1));
              abs_c = fabs(static_cast<double>(std::real(T_element)));
              abs_d = fabs(static_cast<double>(std::imag(T_element)));
              ab = std::max(abs_a, abs_b);
              cd = std::max(abs_c, abs_d);

              /* Scaling */
              if (ab > ov1 / 16) {        /* scale down a, b */
                temp1 /= 16;
                S = S * 16;
              }
              if (cd > ov / 16) {        /* scale down c, d */
                T_element /= 16;
                S = S / 16;
              }
              if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                t = 2.0 / (eps1 * eps1);
                temp1 *= t;
                S = S / t;
              }
              if (cd < un / eps * 2) {        /* scale up c, d */
                t = 2.0 / (eps * eps);
                T_element *= t;
                S = S * t;
              }

              /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
              if (abs_c > abs_d) {
                r = std::imag(T_element) / std::real(T_element);
                t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
              } else {
                r = std::real(T_element) / std::imag(T_element);
                t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
              }
              /* Scale back */
              temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
            } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
              double S = 1.0, eps, ov, un;
              double abs_a, abs_b, abs_c, abs_d, ab, cd;
              double r;
              double t;
              double q[2];

              eps = pow(2.0, -24.0);
              un = pow(2.0, -126.0);
              ov = pow(2.0, 128.0) * (1 - eps);
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
              } else {
                abs_a = fabs(std::real(temp1));
                abs_b = fabs(std::imag(temp1));
              }
              abs_c = fabs(static_cast<double>(std::real(T_element)));
              abs_d = fabs(static_cast<double>(std::imag(T_element)));
              ab = std::max(abs_a, abs_b);
              cd = std::max(abs_c, abs_d);

              /* Scaling */
              if (ab > ov / 16) {        /* scale down a, b */
                temp1 /= 16;
                S = S * 16;
              }
              if (cd > ov / 16) {        /* scale down c, d */
                T_element /= 16;
                S = S / 16;
              }
              if (ab < un / eps * 2) {        /* scale up a, b */
                t = 2.0 / (eps * eps);
                temp1 *= t;
                S = S / t;
              }
              if (cd < un / eps * 2) {        /* scale up c, d */
                t = 2.0 / (eps * eps);
                T_element *= t;
                S = S * t;
              }

              /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
              if (abs_c > abs_d) {
                r = std::imag(T_element) / std::real(T_element);
                t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
              } else {
                r = std::real(T_element) / std::imag(T_element);
                t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
              }
              /* Scale back */
              temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
            } else {
              temp1 = impl::div(temp1, T_element);
            }
          }
          /* if (diag == blas_non_unit_diag) */
          x_i[xi] = impl::to<T>(temp1);
          xi += incxi;
        }                        /* for j<n */

      } else {
        /* not conjugated */
        /* loop 1 */
        xi = start_xi;
        for (j = 0; j < k; j++) {
          /* each time through loop, xi lands on next x to compute. */
          x_elem = x_i[xi];
          /* preform the multiplication -
             in this implementation we do not separate the alpha = 1 case */
          temp1 = impl::mul<TmpType>(x_elem, alpha_i);

          xi = start_xi;

          Tij = dot_start;
          dot_start += dot_start_inc1;

          for (i = j; i > 0; i--) {
            T_element = t_i[Tij];
            x_elem = x_i[xi];
            temp2 = impl::mul<TmpType>(x_elem, T_element);
            temp1 = temp1 - temp2;
            xi += incxi;
            Tij += dot_inc;
          }                        /* for across row */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[Tij];

            if constexpr (std::is_same_v<impl::inner_type_t<A>, double>) {
              // scaled division to avoid overflow.
              double S = 1.0, eps, ov, un, eps1, ov1, un1;
              double abs_a, abs_b, abs_c, abs_d, ab, cd;
              double r;
              double t;
              double q[2];

              eps = pow(2.0, -24.0);
              un = pow(2.0, -126.0);
              ov = pow(2.0, 128.0) * (1 - eps);
              eps1 = pow(2.0, -53.0);
              un1 = pow(2.0, -1022.0);
              ov1 = 1.79769313486231571e+308;
              /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
              abs_a = fabs(std::real(temp1));
              abs_b = fabs(std::imag(temp1));
              abs_c = fabs(static_cast<double>(std::real(T_element)));
              abs_d = fabs(static_cast<double>(std::imag(T_element)));
              ab = std::max(abs_a, abs_b);
              cd = std::max(abs_c, abs_d);

              /* Scaling */
              if (ab > ov1 / 16) {        /* scale down a, b */
                temp1 /= 16;
                S = S * 16;
              }
              if (cd > ov / 16) {        /* scale down c, d */
                T_element /= 16;
                S = S / 16;
              }
              if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                t = 2.0 / (eps1 * eps1);
                temp1 *= t;
                S = S / t;
              }
              if (cd < un / eps * 2) {        /* scale up c, d */
                t = 2.0 / (eps * eps);
                T_element *= t;
                S = S * t;
              }

              /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
              if (abs_c > abs_d) {
                r = std::imag(T_element) / std::real(T_element);
                t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
              } else {
                r = std::real(T_element) / std::imag(T_element);
                t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
              }
              /* Scale back */
              temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
            } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
              double S = 1.0, eps, ov, un;
              double abs_a, abs_b, abs_c, abs_d, ab, cd;
              double r;
              double t;
              double q[2];

              eps = pow(2.0, -24.0);
              un = pow(2.0, -126.0);
              ov = pow(2.0, 128.0) * (1 - eps);
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
              } else {
                abs_a = fabs(std::real(temp1));
                abs_b = fabs(std::imag(temp1));
              }
              abs_c = fabs(static_cast<double>(std::real(T_element)));
              abs_d = fabs(static_cast<double>(std::imag(T_element)));
              ab = std::max(abs_a, abs_b);
              cd = std::max(abs_c, abs_d);

              /* Scaling */
              if (ab > ov / 16) {        /* scale down a, b */
                temp1 /= 16;
                S = S * 16;
              }
              if (cd > ov / 16) {        /* scale down c, d */
                T_element /= 16;
                S = S / 16;
              }
              if (ab < un / eps * 2) {        /* scale up a, b */
                t = 2.0 / (eps * eps);
                temp1 *= t;
                S = S / t;
              }
              if (cd < un / eps * 2) {        /* scale up c, d */
                t = 2.0 / (eps * eps);
                T_element *= t;
                S = S * t;
              }

              /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
              if (abs_c > abs_d) {
                r = std::imag(T_element) / std::real(T_element);
                t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
              } else {
                r = std::real(T_element) / std::imag(T_element);
                t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
              }
              /* Scale back */
              temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
            } else {
              temp1 = impl::div(temp1, T_element);
            }
          }
          /* if (diag == blas_non_unit_diag) */
          x_i[xi] = impl::to<T>(temp1);
          xi += incxi;
        }                        /* for j<k */
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
            temp1 = temp1 - temp2;
            xi += incxi;
            Tij += dot_inc;
          }                        /* for across row */


          /* if the diagonal entry is not equal to one, then divide by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = t_i[Tij];

            if constexpr (std::is_same_v<impl::inner_type_t<A>, double>) {
              // scaled division to avoid overflow.
              double S = 1.0, eps, ov, un, eps1, ov1, un1;
              double abs_a, abs_b, abs_c, abs_d, ab, cd;
              double r;
              double t;
              double q[2];

              eps = pow(2.0, -24.0);
              un = pow(2.0, -126.0);
              ov = pow(2.0, 128.0) * (1 - eps);
              eps1 = pow(2.0, -53.0);
              un1 = pow(2.0, -1022.0);
              ov1 = 1.79769313486231571e+308;
              /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
              abs_a = fabs(std::real(temp1));
              abs_b = fabs(std::imag(temp1));
              abs_c = fabs(static_cast<double>(std::real(T_element)));
              abs_d = fabs(static_cast<double>(std::imag(T_element)));
              ab = std::max(abs_a, abs_b);
              cd = std::max(abs_c, abs_d);

              /* Scaling */
              if (ab > ov1 / 16) {        /* scale down a, b */
                temp1 /= 16;
                S = S * 16;
              }
              if (cd > ov / 16) {        /* scale down c, d */
                T_element /= 16;
                S = S / 16;
              }
              if (ab < un1 / eps1 * 2) {        /* scale up a, b */
                t = 2.0 / (eps1 * eps1);
                temp1 *= t;
                S = S / t;
              }
              if (cd < un / eps * 2) {        /* scale up c, d */
                t = 2.0 / (eps * eps);
                T_element *= t;
                S = S * t;
              }

              /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
              if (abs_c > abs_d) {
                r = std::imag(T_element) / std::real(T_element);
                t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
              } else {
                r = std::real(T_element) / std::imag(T_element);
                t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
              }
              /* Scale back */
              temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
            } else if constexpr (std::is_same_v<impl::inner_type_t<A>, float>) {
              double S = 1.0, eps, ov, un;
              double abs_a, abs_b, abs_c, abs_d, ab, cd;
              double r;
              double t;
              double q[2];

              eps = pow(2.0, -24.0);
              un = pow(2.0, -126.0);
              ov = pow(2.0, 128.0) * (1 - eps);
              if constexpr (std::is_same_v<TmpType, std::complex<double_double>>) {
                abs_a = fabs(std::real(temp1).to_double());
                abs_b = fabs(std::imag(temp1).to_double());
              } else {
                abs_a = fabs(std::real(temp1));
                abs_b = fabs(std::imag(temp1));
              }
              abs_c = fabs(static_cast<double>(std::real(T_element)));
              abs_d = fabs(static_cast<double>(std::imag(T_element)));
              ab = std::max(abs_a, abs_b);
              cd = std::max(abs_c, abs_d);

              /* Scaling */
              if (ab > ov / 16) {        /* scale down a, b */
                temp1 /= 16;
                S = S * 16;
              }
              if (cd > ov / 16) {        /* scale down c, d */
                T_element /= 16;
                S = S / 16;
              }
              if (ab < un / eps * 2) {        /* scale up a, b */
                t = 2.0 / (eps * eps);
                temp1 *= t;
                S = S / t;
              }
              if (cd < un / eps * 2) {        /* scale up c, d */
                t = 2.0 / (eps * eps);
                T_element *= t;
                S = S * t;
              }

              /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
              if (abs_c > abs_d) {
                r = std::imag(T_element) / std::real(T_element);
                t = 1 / (std::real(T_element) + std::imag(T_element) * r);
                q[0] = (std::real(temp1) + std::imag(temp1) * r) * t;
                q[1] = (std::imag(temp1) - std::real(temp1) * r) * t;
              } else {
                r = std::real(T_element) / std::imag(T_element);
                t = 1 / (std::imag(T_element) + std::real(T_element) * r);
                q[0] = ( std::imag(temp1) + std::real(temp1) * r) * t;
                q[1] = (-std::real(temp1) + std::imag(temp1) * r) * t;
              }
              /* Scale back */
              temp1 = impl::mul<TmpType>(A(q[0], q[1]), S);
            } else {
              temp1 = impl::div(temp1, T_element);
            }
          }
          /* if (diag == blas_non_unit_diag) */
          x_i[xi] = impl::to<T>(temp1);
          xi += incxi;
        }                        /* for j<n */

      }
    }
  } else {
    if constexpr (sizeof(TmpType) > sizeof(T)) {
      TmpType temp1;      /* temporary variable for calculations */
      TmpType temp2;      /* temporary variable for calculations */
      TmpType temp3;      /* temporary variable for calculations */
      T x_elem;
      A T_element;        /* temporary variable for an element of matrix T */

      IdxType x_inti = 0, inc_x_inti = 1;
      IdxType k_compare = k;        /*used for comparisons with x_inti */
      TmpType *x_internal;

      x_internal = new(std::nothrow) TmpType[k]; // *) blas_malloc(k * sizeof(TmpType));
      if (k > 0 && x_internal == NULL) {
        BLAS_error("blas_malloc", 0, 0, "temp allocation failed.\n");
      }

      /*loop 1 */
      xi = start_xi;
      /* x_inti already initialized to 0 */
      for (j = 0; j < k; j++) {

        /* each time through loop, xi lands on next x to compute. */
        x_elem = x_i[xi];
        /* preform the multiplication -
           in this implementation we do not separate the alpha = 1 case */
        temp1 = impl::mul<TmpType>(x_elem, alpha_i);

        Tij = dot_start;
        dot_start += dot_start_inc1;

        /*start loop buffer over in loop 1 */
        x_inti = 0;
        for (i = j; i > 0; i--) {
          T_element = t_i[Tij];

          temp3 = x_internal[x_inti];
          temp2 = impl::mul<TmpType>(temp3, T_element);
          temp1 = temp1 - temp2;
          x_inti += inc_x_inti;
          Tij += dot_inc;
        }                        /* for across row */

        /* if the diagonal entry is not equal to one, then divide Xj by
           the entry */
        if (diag == blas_non_unit_diag) {
          T_element = t_i[Tij];
          temp1 = impl::div(temp1, T_element);
        }

        /* if (diag == blas_non_unit_diag) */
        /* place internal precision result in internal buffer */
        x_internal[x_inti] = temp1;

        /* place result x in same place as got x this loop */
        x_i[xi] = impl::to<T>(temp1);
        xi += incxi;
      }                        /* for j<k */
      /*end loop 1 */


      /* loop2 ***************************** */
      x_inti = 0;
      /*loop 2 continue without changing j to start */
      for (; j < n; j++) {

        /* each time through loop, xi lands on next x to compute. */
        x_elem = x_i[xi];
        temp1 = impl::mul<TmpType>(x_elem, alpha_i);

        Tij = dot_start;
        dot_start += dot_start_inc2;

        for (i = k; i > 0 && (x_inti < k_compare); i--) {
          T_element = t_i[Tij];

          temp3 = x_internal[x_inti];
          temp2 = impl::mul<TmpType>(temp3, T_element);
          temp1 = temp1 - temp2;
          x_inti += inc_x_inti;
          Tij += dot_inc;
        }                        /* for across row */
        /*reset index to internal storage loop buffer. */
        x_inti = 0;
        for (; i > 0; i--) {
          T_element = t_i[Tij];

          temp3 = x_internal[x_inti];
          temp2 = impl::mul<TmpType>(temp3, T_element);
          temp1 = temp1 - temp2;
          x_inti += inc_x_inti;
          Tij += dot_inc;
        }                        /* for across row */


        /* if the diagonal entry is not equal to one, then divide by
           the entry */
        if (diag == blas_non_unit_diag) {
          T_element = t_i[Tij];
          temp1 = impl::div(temp1, T_element);
        }

        /* if (diag == blas_non_unit_diag) */
        /* place internal precision result in internal buffer */
        x_internal[x_inti] = temp1;
        x_inti += inc_x_inti;
        if (x_inti >= k_compare)
          x_inti = 0;

        /* place result x in same place as got x this loop */
        x_i[xi] = impl::to<T>(temp1);
        xi += incxi;
      }                        /* for j<n */

      //blas_free(x_internal);
      delete[] x_internal;
    } else {
      TmpType temp1;
      TmpType temp2;
      T x_elem;
      A T_element;

      /*loop 1 */
      xi = start_xi;
      for (j = 0; j < k; j++) {

        /* each time through loop, xi lands on next x to compute. */
        x_elem = x_i[xi];
        /* preform the multiplication -
           in this implementation we do not separate the alpha = 1 case */
        temp1 = impl::mul<TmpType>(x_elem, alpha_i);

        xi = start_xi;

        Tij = dot_start;
        dot_start += dot_start_inc1;

        for (i = j; i > 0; i--) {
          T_element = t_i[Tij];

          x_elem = x_i[xi];
          temp2 = impl::mul<TmpType>(x_elem, T_element);
          temp1 = temp1 - temp2;
          xi += incxi;
          Tij += dot_inc;
        }                        /* for across row */


        /* if the diagonal entry is not equal to one, then divide Xj by
           the entry */
        if (diag == blas_non_unit_diag) {
          T_element = t_i[Tij];
          temp1 = impl::div(temp1, T_element);
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
          temp1 = temp1 - temp2;
          xi += incxi;
          Tij += dot_inc;
        }                        /* for across row */


        /* if the diagonal entry is not equal to one, then divide by
           the entry */
        if (diag == blas_non_unit_diag) {
          T_element = t_i[Tij];


          temp1 = temp1 / static_cast<TmpType>(T_element);

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

template<typename T,
         typename A,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void tbsv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      blas_trans_type trans,
                      blas_diag_type diag,
                      IdxType n,
                      IdxType k,
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
//static const char routine_name[] = "XBLAS::tbsv_x";
  if (k == 0) {
    if constexpr (std::is_same_v<impl::inner_type_t<T>, float>) {
      /* must set prec to output. Ignore user input of prec */
      prec = blas_prec_single;
    } else if constexpr (std::is_same_v<impl::inner_type_t<T>, double>) {
      /* must set prec to output. Ignore user input of prec */
      prec = blas_prec_double;
    }
  }

  switch (prec) {
  case blas_prec_single:
    XBLAS::tbsv<T, A, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, trans, diag, n, k, alpha, t, ldt, x, incx);
    break;
  case blas_prec_double:
    XBLAS::tbsv<T, A, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, trans, diag, n, k, alpha, t, ldt, x, incx);
    break;
  case blas_prec_indigenous:
    XBLAS::tbsv<T, A, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, trans, diag, n, k, alpha, t, ldt, x, incx);
    break;
  case blas_prec_extra:
    XBLAS::tbsv<T, A, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, trans, diag, n, k, alpha, t, ldt, x, incx);
    break;
  }
} /* end XBLAS::tbsv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TBSV_HPP
