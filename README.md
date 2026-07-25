# mdkXBLAS Library

This is a C++ reimplementation of the netlib xblas library created from the
original sources, with some improvements.  Go to https://www.netlib.org/xblas/
if you want the original library instead.

When used as a C++ library, it is a header-only library and you do not
need to compile the xblas library or link to it. Just include the relevant
.hpp files and compile your code. You will need at least C++23 to compile
the headers.

I have preserved the C and Fortran APIs from the original library.
The compiled version of this library can be linked to C or Fortran
programs, replacing the original xblas library if you want to go that
route.

In addition to running the unit tests from the netlib sources (which all pass),
I have also compiled and tested it with Lapack 3.12.1 (https://www.netlib.org/lapack/)
and it passes and fails the same number of tests that the original library
passes and fails.

I developed this code on Windows using winlibs Win64 compilers
GCC 15.1.0 (with POSIX threads) + MinGW-w64 13.0.0 (UCRT) - release 2 and
then GCC 16.1.0 (with POSIX threads) + MinGW-w64 14.0.0 (UCRT) - release 3
(https://winlibs.com/).

# Code Organization

## include/XBLAS

This contains all the .hpp files implementing the xblas functions, as well as
the original .h files from the netlib version. The C++ code is heavily templated,
with concepts and constexpr used throughout.

## src

This contains the same directory structure as the original netlib sources.
The .cpp sources contain the C API functions, which now just forward to the
C++ implementations in the .hpp files. It's all very generic.

The Fortran API is implemented in the .c files, which are essentially
unchanged from the original netlib sources. The Fortran API calls the C API,
which calls the C++ API.

## test/ctests

These are the original netlib tests, unchanged except I added a compile-time
option to have one main program instead of 25. These tests call the C API,
which in turn call the C++ code.

I have not yet written C++ tests that will specifically test compilation in
constexpr contexts, or testing long double or quadruple precision types.

# Building the Library

Cmake is the build tool used (https://cmake.org/). The CMakeLists.txt specifies
version 4.3.3. You'll want to look at the CMakePresets.json file for options to
specify or change. Add the "installDir" directive to the json, if needed. To
compile the library, you will need C, C++, and Fortran compilers. At least C++23
is needed.

Suppose the repository is in the mdkXBLAS directory:
* cd path/to/mdkXBLAS
---
* cmake -B build --preset mdkXBLAS-gcc
or
* cmake -B build --preset mdkXBLAS-gcc --fresh
---
* cmake --build build
or
* cmake --build build --clean-first
---
* ctest --test-dir build
---
* cmake --install build (--strip if desired)
or
* cmake --install build --prefix "C:/Program Files/mdkXBLAS" or some other path

## Build Options

All options are OFF by default.

### MDKXBLAS_USE_FLOAT128

<b>OFF</b>: The internal extended-precision calculations are performed using the
 XBLAS::double_double type.

<b>ON</b>: The internal extended-precision calculations are performed using the
 std::float128_t quadruple-precision floating-point type. The library won't compile
 unless the type is available, as determined by __STDCPP_FLOAT128_T.

I find double_double to be faster than float128_t, but your experience might be
different.

### MDKXBLAS_USE_INT64_INTERNAL_INDEX_TYPE

The level-2 and level-3 blas functions in Fortran operate on 2D arrays and so have two
indexes. These 2D Fortran arrays are mapped to 1D C arrays in the C and C++ implementations.
This creates a problem for large arrays. It is entirely possible for a 2D array indexed
by two 32 bit signed integer indexes to overflow the single 32 bit signed integer index of
a C array.  For example, a 50,000 x 50,000 array has 2,500,000,000 elements, which
cannot be indexed by a 32 bit signed integer.  If you are working with arrays of this
size, you'll need to turn on this option.

<b>OFF</b>: Use int as the internal index type.

<b>ON</b>: Use std::int64_t as the internal index type. The library won't compile unless
 the type is available.

NOTE: This only applies to the internal index type. The interface types for n,
lda, incx, etc. in the C and Fortran APIs are still int. In the C++ API, the type
for n, lda, etc. is a template parameter and can be whatever signed integral type you
give it.

### MDKXBLAS_USE_XERBLA

If there is an error in the inputs, such as n < 0, the BLAS_error function is called
to report the error and exit the program.

<b>OFF</b>: BLAS_error reports the error and exits the program.

<b>ON</b>: BLAS_error calls xerbla_array and lets xerbla_array handle the error.
 It is expected that xerbla_array exits the program. Xerbla_array needs to be linked
 to the program.

NOTE: This option is ignored by the C++ API; BLAS_error always reports the error
and exits.

### MDKXBLAS_NO_BLAS_MALLOC

<b>OFF</b>: Blas_malloc, blas_free, blas_realloc are functions. They call
 malloc, free, realloc.

<b>ON</b>: Blas_malloc, blas_free, blas_realloc are macros. They are replaced by
 malloc, free, realloc.

### MDKXBLAS_DEBUG_BLAS_MALLOC

<b>OFF</b>: There is no debugging added to blas_malloc.

<b>ON</b>: Debugging is added to blas_malloc. The memory allocated by the
 call to malloc is initialized with nans. Only applies if MDKXBLAS_NO_BLAS_MALLOC
 is <b>OFF</b>.

### MDKXBLAS_COMBINE_TESTS

<b>OFF</b>: The unit tests in test/ctests are in 25 separate executables and need to
 be built and run separately. Cmake/ctests does this easily.

<b>ON</b>: The tests are combined into a single executable. Before I had cmake set up
 to build everything, I was using an IDE build environment and it was just easier to
 have a single project for the tests building a single executable to run every test.

## Using the Library

### C++ program using the C++ API

As stated previously, you do not need to build the library to use it.
Just include the relevant .hpp files in include/XBLAS and build your
program. You will need at least C++23 to compile the headers. The functions
are in the XBLAS namespace. If your code would have previously called a C API
function, such as:

BLAS_dgemv_d_s(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);

you would now call:

XBLAS::gemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);

and the template will deduce the types.  Furthermore, since it is a template,
you <i>can</i> use 64 bit integers as the type for m, n, lda, incx, and incy
if you need them that big.

### C++ or C program using the C API

In your code:

\#include <XBLAS/blas_extended.h>

and call the desired BLAS_\* functions. The function prototypes themselves
are listed in blas_extended_proto.h, which is included by blas_extended.h.
If necessary, modify your development environment so it can find the XBLAS
directory that was installed via the cmake --install build step.

You will need to link to the xblas library.

### Fortran program using the Fortran API

Just call the functions you need and link to the xblas library. Recall that
the Fortran API does not have an order parameter. (Everything is column-major
order in Fortran.) If the C API function is:

BLAS_dgemv_d_s(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);

The Fortran API function is:

BLAS_dgemv_d_s(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
