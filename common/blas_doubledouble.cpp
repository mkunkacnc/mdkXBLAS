#include "blas_extended_private.h"
#include "common/BLAS_doubledouble.hpp"

/* compute c = a * b; */
void compute_doubledouble_eq_double_mul_double(double* head_c,
                                               double* tail_c,
                                               double a,
                                               double b)
{
#if 1
  /* Compute double-double = double * double. */
  DoubleDouble res = DoubleDouble::mul(a, b);
  *head_c = res.head;
  *tail_c = res.tail;
#else
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
#endif
}

/* compute c = a * b; */
void compute_doubledouble_eq_float_mul_float(double* head_c,
                                             double* tail_c,
                                             float a,
                                             float b)
{
  /* Compute double-double = float * float. */
  DoubleDouble res = DoubleDouble::mul(a, b);
  *head_c = res.head;
  *tail_c = res.tail;
#if 0
  *head_c = static_cast<double>(a) * b;
  *tail_c = 0.0;
#endif
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

/* compute c = a + b */
void compute_doubledouble_eq_doubledouble_add_double(double* head_c,
                                                     double* tail_c,
                                                     double head_a,
                                                     double tail_a,
                                                     double b)
{
  /* Compute double-double = double-double + double. */
  double e, t1, t2;

  /* Knuth trick. */
  t1 = head_a + b;
  e = t1 - head_a;
  t2 = ((b - e) + (head_a - (t1 - e))) + tail_a;

  /* The result is t1 + t2, after normalization. */
  *head_c = t1 + t2;
  *tail_c = t2 - (*head_c - t1);
}

/* compute c = a / b */
void compute_doubledouble_eq_doubledouble_div_double(double* head_c,
                                                     double* tail_c,
                                                     double head_a,
                                                     double tail_a,
                                                     double b)
{
  /* Compute double-double = double-double / double,
     using a Newton iteration scheme. */
  double b1, b2, con, e, t1, t2, t11, t21, t12, t22;

  /* Compute a DP approximation to the quotient. */
  t1 = head_a / b;

  /* Split t1 and b into two parts with at most 26 bits each,
     using the Dekker-Veltkamp method. */
  con = t1 * SPLIT;
  t11 = con - (con - t1);
  t21 = t1 - t11;
  con = b * SPLIT;
  b1 = con - (con - b);
  b2 = b - b1;

  /* Compute t1 * b using Dekker method. */
  t12 = t1 * b;
  t22 = (((t11 * b1 - t12) + t11 * b2) + t21 * b1) + t21 * b2;

  /* Compute dda - (t12, t22) using Knuth trick. */
  t11 = head_a - t12;
  e = t11 - head_a;
  t21 = ((-t12 - e) + (head_a - (t11 - e))) + tail_a - t22;

  /* Compute high-order word of (t11, t21) and divide by b. */
  t2 = (t11 + t21) / b;

  /* The result is t1 + t2, after normalization. */
  *head_c = t1 + t2;
  *tail_c = t2 - (*head_c - t1);
}

/* compute c = a / b */
void compute_doubledouble_eq_doubledouble_div_doubledouble(double* head_c,
                                                           double* tail_c,
                                                           double head_a,
                                                           double tail_a,
                                                           double head_b,
                                                           double tail_b)
{
  double q1, q2, q3;
  double a1, a2, b1, b2;
  double p1, p2, c;
  double s1, s2, v;
  double t1, t2;
  double r1, r2;
  double cona, conb;

  q1 = head_a / head_b;        /*  approximate quotient */

  /*  Compute  q1 * b  */
  cona = q1 * SPLIT;
  conb = head_b * SPLIT;
  a1 = cona - (cona - q1);
  b1 = conb - (conb - head_b);
  a2 = q1 - a1;
  b2 = head_b - b1;

  /*  (p1, p2) is the product of high order terms. */
  p1 = q1 * head_b;
  p2 = (((a1 * b1 - p1) + a1 * b2) + a2 * b1) + a2 * b2;

  /*  Compute the low-order term */
  c = q1 * tail_b;

  /*  Compute  (s1, s2) = (p1, p2) + c */
  s1 = p1 + c;
  v = s1 - p1;
  s2 = ((c - v) + (p1 - (s1 - v))) + p2;

  /*  Renormalize. */
  p1 = s1 + s2;
  p2 = s2 - (p1 - s1);

  /*  Compute  a - (p1, p2)    */
  s1 = head_a - p1;
  v = s1 - head_a;
  s2 = (head_a - (s1 - v)) - (p1 + v);

  t1 = tail_a - p2;
  v = t1 - tail_a;
  t2 = (tail_a - (t1 - v)) - (p2 + v);

  s2 += t1;
  t1 = s1 + s2;
  s2 = s2 - (t1 - s1);

  t2 += s2;
  r1 = t1 + t2;
  r2 = t2 - (r1 - t1);

  /*  Compute the next quotient. */
  q2 = r1 / head_b;

  /*  Compute residual   r1 - q2 * b          */
  cona = q2 * SPLIT;
  a1 = cona - (cona - q2);
  a2 = q2 - a1;

  /*  (p1, p2) is the product of high order terms. */
  p1 = q2 * head_b;
  p2 = (((a1 * b1 - p1) + a1 * b2) + a2 * b1) + a2 * b2;

  /*  Compute the low-order term */
  c = q2 * tail_b;

  /*  Compute  (s1, s2) = (p1, p2) + c */
  s1 = p1 + c;
  v = s1 - p1;
  s2 = ((c - v) + (p1 - (s1 - v))) + p2;

  /*  Renormalize. */
  p1 = s1 + s2;
  p2 = s2 - (p1 - s1);

  /*  Compute  (r1, r2) - (p1, p2)    */
  s1 = r1 - p1;
  v = s1 - r1;
  s2 = (r1 - (s1 - v)) - (p1 + v);

  t1 = r2 - p2;
  v = t1 - r2;
  t2 = (r2 - (t1 - v)) - (p2 + v);

  s2 += t1;
  t1 = s1 + s2;
  s2 = s2 - (t1 - s1);

  t2 += s2;
  s1 = t1 + t2;

  /*  Compute the last correction. */
  q3 = s1 / head_b;

  /* Renormalize q1, q2, q3. */
  s1 = q2 + q3;
  s2 = q3 - (s1 - q2);

  *head_c = q1 + s1;
  t1 = s1 - (*head_c - q1);

  *tail_c = s2 + t1;
}

/* compute c = a * b */
void compute_complex_double_double_eq_double_mul_double(double head_c[],
                                                        double tail_c[],
                                                        double a,
                                                        double b)
{
  /* Compute complex double-double = double * double. */
  compute_doubledouble_eq_double_mul_double(&head_c[0], &tail_c[0], a, b);
  head_c[1] = 0.0;
  tail_c[1] = 0.0;
}
