#include "blas_extended_private.h"

/* compute c = a * b; */
void compute_doubledouble_eq_double_mul_double(double* head_c,
                                               double* tail_c,
                                               double a,
                                               double b)
{
  /* Compute double_double = double * double. */
  double a1, a2, b1, b2, con;

#define SPLIT_VAR(a)  \
  con = a * SPLIT;    \
  a##1 = con - a;     \
  a##1 = con - a##1;  \
  a##2 = a - a##1;

  SPLIT_VAR(a)
  SPLIT_VAR(b)

#undef SPLIT_VAR

  *head_c = a * b;
  *tail_c = (((a1 * b1 - *head_c) + a1 * b2) + a2 * b1) + a2 * b2;
}

/* compute c = a + b */
void compute_doubledouble_eq_doubledouble_add_doubledouble(double* head_c,
                                                           double* tail_c,
                                                           double head_a,
                                                           double tail_a,
                                                           double head_b,
                                                           double tail_b)
{
  /* Compute double-double = double-double + double-double. */
  double bv;
  double s1, s2, t1, t2;

  /* Add two hi words. */
  s1 = head_b + head_a;
  bv = s1 - head_b;
  s2 = ((head_a - bv) + (head_b - (s1 - bv)));

  /* Add two lo words. */
  t1 = tail_b + tail_a;
  bv = t1 - tail_b;
  t2 = ((tail_a - bv) + (tail_b - (t1 - bv)));

  s2 += t1;

  /* Renormalize (s1, s2)  to  (t1, s2) */
  t1 = s1 + s2;
  s2 = s2 - (t1 - s1);

  t2 += s2;

  /* Renormalize (t1, t2)  */
  *head_c = t1 + t2;
  *tail_c = t2 - (head_b - t1);
}
