#include "blas_extended_private.h"
#include "common/XBLAS_double_double.hpp"

using XBLAS::double_double;

/* compute c = a * b; */
void compute_doubledouble_eq_double_mul_double(double* head_c,
                                               double* tail_c,
                                               double a,
                                               double b)
{
  /* Compute double-double = double * double. */
  double_double c = double_double::mul(a, b);
  *head_c = c.head_();
  *tail_c = c.tail_();
}

/* compute c = a * b; */
void compute_doubledouble_eq_float_mul_float(double* head_c,
                                             double* tail_c,
                                             float a,
                                             float b)
{
  /* Compute double-double = float * float. */
  double_double c = double_double::mul(a, b);
  *head_c = c.head_();
  *tail_c = c.tail_();
}

/* compute c = a * b */
void compute_doubledouble_eq_doubledouble_mul_double(double* head_c,
                                                     double* tail_c,
                                                     double head_a,
                                                     double tail_a,
                                                     double b)
{
  /* Compute double-double = double-double * double. */
  double_double a(head_a, tail_a);
  double_double c = a * b;
  *head_c = c.head_();
  *tail_c = c.tail_();
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
  double_double a(head_a, tail_a);
  double_double b(head_b, tail_b);
  double_double c = a + b;
  *head_c = c.head_();
  *tail_c = c.tail_();
}

/* compute c = a + b */
double compute_double_eq_doubledouble_add_doubledouble(double head_a,
                                                       double tail_a,
                                                       double head_b,
                                                       double tail_b)
{
  /* Compute double = double-double + double-double. */
  double_double a(head_a, tail_a);
  double_double b(head_b, tail_b);
  return double_double::add(a, b);
}

/* compute c = a + b */
void compute_doubledouble_eq_double_add_double(double* head_c,
                                               double* tail_c,
                                               double a,
                                               double b)
{
  /* Compute double-double = double + double. */
  double_double c = double_double::add(a, b);
  *head_c = c.head_();
  *tail_c = c.tail_();
}

/* compute c = a + b */
void compute_doubledouble_eq_doubledouble_add_double(double* head_c,
                                                     double* tail_c,
                                                     double head_a,
                                                     double tail_a,
                                                     double b)
{
  /* Compute double-double = double-double + double. */
  double_double a(head_a, tail_a);
  double_double c = a + b;
  *head_c = c.head_();
  *tail_c = c.tail_();
}

/* compute c = a / b */
void compute_doubledouble_eq_doubledouble_div_double(double* head_c,
                                                     double* tail_c,
                                                     double head_a,
                                                     double tail_a,
                                                     double b)
{
  /* Compute double-double = double-double / double. */
  double_double a(head_a, tail_a);
  double_double c = a / b;
  *head_c = c.head_();
  *tail_c = c.tail_();
}

/* compute c = a / b */
void compute_doubledouble_eq_doubledouble_div_doubledouble(double* head_c,
                                                           double* tail_c,
                                                           double head_a,
                                                           double tail_a,
                                                           double head_b,
                                                           double tail_b)
{
  /* Compute double-double = double-double / double-double. */
  double_double a(head_a, tail_a);
  double_double b(head_b, tail_b);
  double_double c = a / b;
  *head_c = c.head_();
  *tail_c = c.tail_();
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
