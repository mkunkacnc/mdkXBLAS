#ifndef BLAS_ERROR_H
#define BLAS_ERROR_H

#ifdef XBLAS_NO_INLINE
/* function definition is in BLAS_error.cpp, client
 * code needs to link to the library.
 */

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

void BLAS_error(const char *rname,
                int iflag,
                int ival,
                const char *form,
                ...);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#elifdef __cplusplus /* !XBLAS_NO_INLINE */
/* function definition is here so we don't
 * need to link to the library. Only applies
 * to C++, since C will still need to link to
 * the library.
 */

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>

inline
void BLAS_error(const char *rname,
                int iflag,
                int ival,
                const char *form,
                ...)
/*
 * Argument
 * ========
 * rname     (input) routine name
 *
 * iflag     (input) a negative value indicates that parameter number -IFLAG
 *                   caused the error; a nonnegative value is an
 *                   implementation-specific error code.
 *
 * ival      (input) the value of parameter number -IFLAG.
 */
{
  va_list argptr;
  va_start(argptr, form);
  std::fprintf(stderr, "Error #%d from routine %s:\n", iflag, rname);
  if (form) {
    std::vfprintf(stderr, form, argptr);
  } else if (iflag < 0) {
    std::fprintf(stderr,
                 "  Parameter number %d to routine %s had the illegal value %d\n",
                 -iflag, rname, ival);
  } else {
    std::fprintf(stderr, "  Unknown error code %d from routine %s\n",
                 iflag, rname);
  }
  std::exit(iflag);
}

#endif /* !XBLAS_NO_INLINE */

#endif /* BLAS_ERROR_H */
