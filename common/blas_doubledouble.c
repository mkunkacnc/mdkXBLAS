#include "blas_extended_private.h"

/* compute c = a * b; */
void compute_doubledouble_eq_double_mul_double(double* head_c,
                                               double* tail_c,
                                               double a,
                                               double b)
{
    /* Compute double_double = double * double. */
    double a1, a2, b1, b2, con;

#define SPLIT_VAR(a)    \
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
