#ifndef BLAS_MALLOC_H
#define BLAS_MALLOC_H

#ifdef XBLAS_NO_BLAS_MALLOC

#include <stdlib.h>

#define blas_malloc(s) malloc(s)
#define blas_realloc(p, s) realloc(p,s)
#define blas_free(p) free(p)

#else

/* stddef is needed for size_t */
#include <stddef.h>

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif


void *blas_malloc(size_t size);
void *blas_realloc(void *p, size_t size);
void blas_free(void *p);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#endif

#endif
