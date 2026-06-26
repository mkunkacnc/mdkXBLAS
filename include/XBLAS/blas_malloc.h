#ifndef BLAS_MALLOC_H
#define BLAS_MALLOC_H 1

#ifdef MDKXBLAS_NO_BLASMALLOC

#include <malloc.h>
#define blas_malloc(s) malloc(s)
#define blas_realloc(p, s) realloc(p,s)
#define blas_free(p) free(p)

#else

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* stddef is needed for size_t */
#include <stddef.h>
void  *blas_malloc(size_t size);
void *blas_realloc(void *p, size_t size);
void blas_free(void *p);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#endif

#endif
