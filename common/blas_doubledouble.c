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

/* compute c = a * b */
void compute_doubledouble_eq_doubledouble_mul_double(double* head_c,
                                                     double* tail_c,
                                                     double head_a,
                                                     double tail_a,
                                                     double b)
{
  /* Compute double-double = double-double * double. */
  double a11, a21, b1, b2, c11, c21, c2, con, t1, t2;

  con = head_a * SPLIT;
  a11 = con - head_a;
  a11 = con - a11;
  a21 = head_a - a11;

  con = b * SPLIT;
  b1 = con - b;
  b1 = con - b1;
  b2 = b - b1;

  c11 = head_a * b;
  c21 = (((a11 * b1 - c11) + a11 * b2) + a21 * b1) + a21 * b2;

  c2 = tail_a * b;
  t1 = c11 + c2;
  t2 = (c2 - (t1 - c11)) + c21;

  *head_c = t1 + t2;
  *tail_c = t2 - (*head_c - t1);
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
  s1 = head_a + head_b;
  bv = s1 - head_a;
  s2 = ((head_b - bv) + (head_a - (s1 - bv)));

  /* Add two lo words. */
  t1 = tail_a + tail_b;
  bv = t1 - tail_a;
  t2 = ((tail_b - bv) + (tail_a - (t1 - bv)));

  s2 += t1;

  /* Renormalize (s1, s2)  to  (t1, s2) */
  t1 = s1 + s2;
  s2 = s2 - (t1 - s1);

  t2 += s2;

  /* Renormalize (t1, t2)  */
  *head_c = t1 + t2;
  *tail_c = t2 - (*head_c - t1);
}

/* compute c = a + b */
double compute_double_eq_doubledouble_add_doubledouble(double head_a,
                                                       double tail_a,
                                                       double head_b,
                                                       double tail_b)
{
  /* Compute double = double-double + double-double. */
  double bv;
  double s1, s2, t1, t2;

  /* Add two hi words. */
  s1 = head_a + head_b;
  bv = s1 - head_a;
  s2 = ((head_b - bv) + (head_a - (s1 - bv)));

  /* Add two lo words. */
  t1 = tail_a + tail_b;
  bv = t1 - tail_a;
  t2 = ((tail_b - bv) + (tail_a - (t1 - bv)));

  s2 += t1;

  /* Renormalize (s1, s2)  to  (t1, s2) */
  t1 = s1 + s2;
  s2 = s2 - (t1 - s1);

  t2 += s2;

  /* Renormalize (t1, t2) */
  return t1 + t2;
}

/* compute c = a + b */
void compute_doubledouble_eq_double_add_double(double* head_c,
                                               double* tail_c,
                                               double a,
                                               double b)
{
  /* Compute double-double = double + double. */
  double e, t1, t2;

  /* Knuth trick. */
  t1 = a + b;
  e = t1 - a;
  t2 = ((b - e) + (a - (t1 - e)));

  /* The result is t1 + t2, after normalization. */
  *head_c = t1 + t2;
  *tail_c = t2 - (*head_c - t1);
}
