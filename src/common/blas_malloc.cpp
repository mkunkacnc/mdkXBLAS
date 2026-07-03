#include "blas_malloc.h"
#include <cstdlib>

void *blas_malloc(std::size_t size)
{
#ifdef XBLAS_DEBUG_BLAS_MALLOC
  void *ptr = std::malloc(size);
  if (size % sizeof(float) == 0) {
    int n = size / sizeof(float);
    int i;
    for (i = 0; i < n; i++) {
      ((float *) ptr)[i] = 0.0 / 0.0;
    }
  }
  return ptr;
#else
  return std::malloc(size);
#endif
}

void blas_free(void *ptr)
{
  std::free(ptr);
}

void *blas_realloc(void *ptr, std::size_t size)
{
  return std::realloc(ptr, size);
}
