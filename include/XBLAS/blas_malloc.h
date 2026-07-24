#ifndef BLAS_MALLOC_H
#define BLAS_MALLOC_H

#ifdef XBLAS_NO_BLAS_MALLOC

#include <stdlib.h>

#define blas_malloc(s) malloc(s)
#define blas_realloc(p, s) realloc(p,s)
#define blas_free(p) free(p)

#elifdef XBLAS_NO_INLINE
/* function definitions are in blas_malloc.cpp, client
 * code needs to link to the library.
 */

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

#elifdef __cplusplus /* !XBLAS_NO_INLINE */
/* function definitions are here so we don't
 * need to link to the library. Only applies
 * to C++, since C will still need to link to
 * the library.
 */

#include <cstdlib>

/* Set up for C function definitions, even when using C++ */
extern "C" {

inline
void *blas_malloc(std::size_t size)
{
#ifdef XBLAS_DEBUG_BLAS_MALLOC
  void *ptr = std::malloc(size);
  if (size % sizeof(float) == 0) {
    int n = size / sizeof(float);
    int i;
    for (i = 0; i < n; i++) {
      static_cast<float *>(ptr)[i] = 0.0 / 0.0;
    }
  }
  return ptr;
#else
  return std::malloc(size);
#endif
}

inline
void blas_free(void *ptr)
{
  std::free(ptr);
}

inline
void *blas_realloc(void *ptr, std::size_t size)
{
  return std::realloc(ptr, size);
}

/* Ends C function definitions when using C++ */
}

#endif /* XBLAS_NO_BLAS_MALLOC */

#endif
