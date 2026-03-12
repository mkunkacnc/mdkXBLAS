#ifndef XBLAS_IMPL_HPP
#define XBLAS_IMPL_HPP

// Need at least C++23
static_assert(__cplusplus >= 202302L, "Need at least C++23");
// Both of these may be defined starting with C++23
//static_assert(__STDCPP_FLOAT16_T__, "Need float16");
//static_assert(__STDCPP_FLOAT128_T__, "Need float128");

#include "common/XBLAS_double_double.hpp"

#include <complex>
#include <type_traits>

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

//---------------------------
// IS_COMPLEX

template<typename T>
struct is_complex { static constexpr auto value = false; };

template<typename T>
struct is_complex<std::complex<T>> { static constexpr auto value = true; };

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

//---------------------------
// INNER_TYPE
// T or T::value_type

template<typename T>
struct inner_type { using type = T; };

template<typename T>
requires is_complex_v<T>
struct inner_type<T> { using type = T::value_type; };

template<typename T>
using inner_type_t = inner_type<T>::type;

//---------------------------
// USES_DOUBLE_DOUBLE

template<typename T>
struct uses_double_double { static constexpr auto value = std::is_same_v<inner_type_t<T>, double_double>; };

template<typename T>
inline constexpr bool uses_double_double_v = uses_double_double<T>::value;

//---------------------------
// SIZE_LE

template<typename X, typename Y>
struct size_le { static constexpr auto value = sizeof(inner_type_t<X>) <= sizeof(inner_type_t<Y>); };

template<typename X, typename Y>
inline constexpr bool size_le_v = size_le<X, Y>::value;

//---------------------------
// GET_INNER_TYPE
// T or T::value_type, depending on types X and Y

template<typename X,
         typename Y,
         typename T>
struct get_inner_type { using type = T; };

// if X and Y are both real, we want the Z in T = complex<Z>
template<typename X,
         typename Y,
         typename T>
requires (std::floating_point<X> && std::floating_point<Y>)
struct get_inner_type<X, Y, T> { using type = inner_type_t<T>; };

template<typename X,
         typename Y,
         typename T>
using get_inner_type_t = get_inner_type<X, Y, T>::type;

//-------------------------------------
// INTERNAL_PRECISION

template<typename T, int prec>
struct internal_precision { using type = T; };

template<>
struct internal_precision<float, blas_prec_double> { using type = double; };
template<>
struct internal_precision<float, blas_prec_indigenous> { using type = double; };
template<>
struct internal_precision<float, blas_prec_extra> { using type = double_double; };

template<>
struct internal_precision<double, blas_prec_extra> { using type = double_double; };

template<>
struct internal_precision<std::complex<float>, blas_prec_double> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_indigenous> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_extra> { using type = std::complex<double_double>; };

template<>
struct internal_precision<std::complex<double>, blas_prec_extra> { using type = std::complex<double_double>; };

template<typename T, int prec>
using internal_precision_t = typename internal_precision<T, prec>::type;

//-------------------------------------
// MUL

template<typename C,
         typename A,
         typename B>
constexpr inline C mul(A a, B b)
{
  // C mul(C, C)
  if constexpr (std::is_same_v<C, A> &&
                std::is_same_v<C, B>) {
    return a * b;

  // double_double mul(A, B), A, B floating-point
  } else if constexpr (std::is_same_v<C, double_double> &&
                       std::floating_point<A> &&
                       std::floating_point<B>) {
    return double_double::mul(a, b);

  // complex<double_double> mul(A, B), A, B real, complex, double_double, etc.
  } else if constexpr (std::is_same_v<C, std::complex<double_double>>) {
    return double_double::mul(a, b);

  // All complex<double_double> should have been handled above.
  // complex<T> mul(A, B)
  } else if constexpr (is_complex_v<C> &&
                       std::floating_point<A> &&
                       std::floating_point<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return static_cast<inner_type_t<C>>(a) * b;

  // complex<T> mul(complex<U>, B)
  } else if constexpr (is_complex_v<C> &&
                       is_complex_v<A> &&
                       std::floating_point<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return static_cast<C>(a) * static_cast<inner_type_t<C>>(b);

  // complex<T> mul(A, complex<U>)
  } else if constexpr (is_complex_v<C> &&
                       std::floating_point<A> &&
                       is_complex_v<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return static_cast<inner_type_t<C>>(a) * static_cast<C>(b);

  // complex<T> mul(complex<U>, complex<V>)
  } else if constexpr (is_complex_v<C> &&
                       is_complex_v<A> &&
                       is_complex_v<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return static_cast<C>(a) * static_cast<C>(b);

  // At this point C, A, B should all be real (possibly double_double), not complex.
  // C mul(C, B) or C mul(A, C)
  } else if constexpr (std::is_same_v<C, A> || std::is_same_v<C, B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return a * b;

  // C mul(A, B)
  } else {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    static_assert(!std::is_same_v<C, double_double>);
    return static_cast<C>(a) * b;
  }
}

//-------------------------------------
// TO

template<typename To,
         typename From>
constexpr inline To to(From from)
{
  return static_cast<To>(from);
}

template<typename To>
constexpr inline To to(double_double from)
{
  return static_cast<To>(from.to_double());
}

template<typename To>
requires is_complex_v<To>
constexpr inline To to(std::complex<double_double> from)
{
  return To(to<inner_type_t<To>>(std::real(from)), to<inner_type_t<To>>(std::imag(from)));
}

//-------------------------------------
// ADD

template<typename C,
         typename A,
         typename B>
constexpr inline C add(A a, B b)
{
  if constexpr (std::floating_point<C> &&
                std::is_same_v<A, double_double> &&
                std::is_same_v<B, double_double>) {
    return double_double::add(a, b);

  } else if constexpr (is_complex_v<C> &&
                       std::floating_point<inner_type_t<C>> &&
                       std::is_same_v<A, std::complex<double_double>> &&
                       std::is_same_v<B, std::complex<double_double>>) {
    return C(double_double::add(std::real(a), std::real(b)), double_double::add(std::imag(a), std::imag(b)));

  } else if constexpr (is_complex_v<C> &&
                       is_complex_v<A> &&
                       is_complex_v<B>) {
    return C(std::real(a) + std::real(b), std::imag(a) + std::imag(b));

  } else {
    return to<C>(a + b);
  }
}

//---------------------------
// CONJ
// use nothing (real) or conj (complex)

struct Conj
{
    template<typename T> // for real types
    static constexpr auto func(T x) { return x; }

    template<typename T> // for complex types
    static constexpr auto func(const std::complex<T>& z) { return std::conj(z); }
};

//---------------------------
// ZERO
// return 0 with the correct type

template<typename T>
struct zero { static constexpr auto value = T(0); };

template<typename T>
struct zero<std::complex<T>> { static constexpr auto value = std::complex<T>(T(0), T(0)); };

template<typename T>
inline constexpr auto zero_v = zero<T>::value;

//---------------------------
// DIV

template<typename A,
         typename B>
constexpr A div(A a, B b)
{
  if constexpr (std::is_same_v<A, std::complex<double>> &&
                std::is_same_v<B, std::complex<double>>) {
    double S = 1.0, eps, ov, un;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double r;
    double t;
    double q[2];

    eps = std::pow(2.0, -24.0);
    un = std::pow(2.0, -126.0);
    ov = std::pow(2.0, 128.0) * (1 - eps);
    abs_a = std::fabs(std::real(a));
    abs_b = std::fabs(std::imag(a));
    abs_c = std::fabs(static_cast<double>(std::real(b)));
    abs_d = std::fabs(static_cast<double>(std::imag(b)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov / 16) {        /* scale down a, b */
      a /= 16;
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      b /= 16;
      S = S / 16;
    }
    if (ab < un / eps * 2) {        /* scale up a, b */
      t = 2.0 / (eps * eps);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(b) / std::real(b);
      t = 1 / (std::real(b) + std::imag(b) * r);
      q[0] = (std::real(a) + std::imag(a) * r) * t;
      q[1] = (std::imag(a) - std::real(a) * r) * t;
    } else {
      r = std::real(b) / std::imag(b);
      t = 1 / (std::imag(b) + std::real(b) * r);
      q[0] = ( std::imag(a) + std::real(a) * r) * t;
      q[1] = (-std::real(a) + std::imag(a) * r) * t;
    }
    /* Scale back */
    return A(q[0] * S, q[1] * S);

  } else if constexpr (std::is_same_v<A, std::complex<double>> &&
                       std::is_same_v<B, std::complex<float>>) {
    double S = 1.0, eps, ov, un, eps1, ov1, un1;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double r;
    double t;
    double q[2];

    eps = std::pow(2.0, -24.0);
    un = std::pow(2.0, -126.0);
    ov = std::pow(2.0, 128.0) * (1 - eps);
    eps1 = std::pow(2.0, -53.0);
    un1 = std::pow(2.0, -1022.0);
    ov1 = 1.79769313486231571e+308;
    /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
    abs_a = std::fabs(std::real(a));
    abs_b = std::fabs(std::imag(a));
    abs_c = std::fabs(static_cast<double>(std::real(b)));
    abs_d = std::fabs(static_cast<double>(std::imag(b)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov1 / 16) {        /* scale down a, b */
      a /= 16;
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      b /= 16;
      S = S / 16;
    }
    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      t = 2.0 / (eps1 * eps1);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(b) / std::real(b);
      t = 1 / (std::real(b) + std::imag(b) * r);
      q[0] = (std::real(a) + std::imag(a) * r) * t;
      q[1] = (std::imag(a) - std::real(a) * r) * t;
    } else {
      r = std::real(b) / std::imag(b);
      t = 1 / (std::imag(b) + std::real(b) * r);
      q[0] = ( std::imag(a) + std::real(a) * r) * t;
      q[1] = (-std::real(a) + std::imag(a) * r) * t;
    }
    /* Scale back */
    return A(q[0] * S, q[1] * S);

  } else if constexpr (std::is_same_v<A, std::complex<float>> &&
                       std::is_same_v<B, std::complex<float>>) {

    double S = 1.0, eps, ov, un;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double r;
    double t;
    double q[2];

    eps = std::pow(2.0, -24.0);
    un = std::pow(2.0, -126.0);
    ov = std::pow(2.0, 128.0) * (1 - eps);
    abs_a = std::fabs(static_cast<double>(std::real(a)));
    abs_b = std::fabs(static_cast<double>(std::imag(a)));
    abs_c = std::fabs(static_cast<double>(std::real(b)));
    abs_d = std::fabs(static_cast<double>(std::imag(b)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov / 16) {        /* scale down a, b */
      a /= 16;
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      b /= 16;
      S = S / 16;
    }
    if (ab < un / eps * 2) {        /* scale up a, b */
      t = 2.0 / (eps * eps);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(b) / std::real(b);
      t = 1 / (std::real(b) + std::imag(b) * r);
      q[0] = (std::real(a) + std::imag(a) * r) * t;
      q[1] = (std::imag(a) - std::real(a) * r) * t;
    } else {
      r = std::real(b) / std::imag(b);
      t = 1 / (std::imag(b) + std::real(b) * r);
      q[0] = ( std::imag(a) + std::real(a) * r) * t;
      q[1] = (-std::real(a) + std::imag(a) * r) * t;
    }
    /* Scale back */
    return A(q[0] * S, q[1] * S);

  } else if constexpr (std::is_same_v<A, std::complex<double_double>> &&
                       std::is_same_v<B, std::complex<double>>) {
    double head_temp1[2], tail_temp1[2];
    head_temp1[0] = std::real(a).head_();
    head_temp1[1] = std::imag(a).head_();
    tail_temp1[0] = std::real(a).tail_();
    tail_temp1[1] = std::imag(a).tail_();
    B T_element = b;

    double S = 1.0, eps, ov, un, eps1, ov1, un1;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double s;
    double r;
    double head_t, tail_t;
    double head_t1, tail_t1;
    double head_t2, tail_t2;
    double head_q[2], tail_q[2];

    eps = std::pow(2.0, -53.0);        /* double precision */
    un = std::pow(2.0, -1022.0);
    ov = 1.79769313486231571e+308;
    /* = (pow(2.0, 1023.0) * (1 - eps)) * 2.0 */
    eps1 = std::pow(2.0, -104.0);        /* extra precision */
    un1 = std::pow(2.0, -1022.0);
    ov1 = 1.79769313486231571e+308;
    /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
    abs_a = std::fabs(head_temp1[0]);
    abs_b = std::fabs(head_temp1[1]);
    abs_c = std::fabs(static_cast<double>(std::real(T_element)));
    abs_d = std::fabs(static_cast<double>(std::imag(T_element)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov1 / 16) {        /* scale down a, b */
      {
        double head_a, tail_a;
        double head_b, tail_b;
        head_a = head_temp1[0];
        tail_a = tail_temp1[0];
        compute_doubledouble_eq_doubledouble_div_double(&head_b, &tail_b, head_a, tail_a, 16.0);
        head_temp1[0] = head_b;
        tail_temp1[0] = tail_b;
        head_a = head_temp1[1];
        tail_a = tail_temp1[1];
        compute_doubledouble_eq_doubledouble_div_double(&head_b, &tail_b, head_a, tail_a, 16.0);
        head_temp1[1] = head_b;
        tail_temp1[1] = tail_b;
      }
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      T_element /= 16;
      S = S / 16;
    }
    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      s = 2.0 / (eps1 * eps1);
      {
        /* Compute complex-extra = complex-extra * real. */
        double head_a0, tail_a0;
        double head_a1, tail_a1;
        double head_t, tail_t;
        head_a0 = head_temp1[0];
        tail_a0 = tail_temp1[0];
        head_a1 = head_temp1[1];
        tail_a1 = tail_temp1[1];
        compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, s);
        head_temp1[0] = head_t;
        tail_temp1[0] = tail_t;
        compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, s);
        head_temp1[1] = head_t;
        tail_temp1[1] = tail_t;
      }

      S = S / s;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      s = 2.0 / (eps * eps);
      T_element *= s;
      S = S * s;
    }

    /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(T_element) / std::real(T_element);
      compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, std::imag(T_element));
      compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, std::real(T_element));
      head_t1 = head_temp1[1];
      tail_t1 = tail_temp1[1];        /* b */
      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
      head_t1 = head_temp1[0];
      tail_t1 = tail_temp1[0];        /* a */
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
      compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
      head_q[0] = head_t2;
      tail_q[0] = tail_t2;
      head_t1 = head_temp1[0];
      tail_t1 = tail_temp1[0];        /* a */
      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
      head_t1 = head_temp1[1];
      tail_t1 = tail_temp1[1];        /* b */
      {
        double head_bt, tail_bt;
        head_bt = -head_t2;
        tail_bt = -tail_t2;
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
      }                /* b - a*r */
      compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
      head_q[1] = head_t2;
      tail_q[1] = tail_t2;
    } else {
      r = std::real(T_element) / std::imag(T_element);
      compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, std::real(T_element));
      compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, std::imag(T_element));
      head_t1 = head_temp1[0];
      tail_t1 = tail_temp1[0];        /* a */
      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
      head_t1 = head_temp1[1];
      tail_t1 = tail_temp1[1];        /* b */
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
      compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
      head_q[0] = head_t2;
      tail_q[0] = tail_t2;
      head_t1 = head_temp1[1];
      tail_t1 = tail_temp1[1];        /* b */
      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
      head_t1 = head_temp1[0];
      tail_t1 = tail_temp1[0];        /* a */
      {
        double head_bt, tail_bt;
        head_bt = -head_t1;
        tail_bt = -tail_t1;
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
      }                /* -a + b*r */
      compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
      head_q[1] = head_t2;
      tail_q[1] = tail_t2;
    }
    /* Scale back */
    if (S == 1.0) {
      head_temp1[0] = head_q[0];
      tail_temp1[0] = tail_q[0];
      head_temp1[1] = head_q[1];
      tail_temp1[1] = tail_q[1];
    } else {
      /* Compute complex-extra = complex-extra * real. */
      double head_a0, tail_a0;
      double head_a1, tail_a1;
      double head_t, tail_t;
      head_a0 = head_q[0];
      tail_a0 = tail_q[0];
      head_a1 = head_q[1];
      tail_a1 = tail_q[1];
      compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, S);
      head_temp1[0] = head_t;
      tail_temp1[0] = tail_t;
      compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, S);
      head_temp1[1] = head_t;
      tail_temp1[1] = tail_t;
    }

    impl::inner_type_t<A> cr(head_temp1[0], tail_temp1[0]);
    impl::inner_type_t<A> ci(head_temp1[1], tail_temp1[1]);
    return A(cr, ci);

  } else if constexpr (std::is_same_v<A, std::complex<double_double>> &&
                       std::is_same_v<B, std::complex<float>>) {
    double head_temp1[2], tail_temp1[2];
    head_temp1[0] = std::real(a).head_();
    head_temp1[1] = std::imag(a).head_();
    tail_temp1[0] = std::real(a).tail_();
    tail_temp1[1] = std::imag(a).tail_();
    B T_element = b;

    double S = 1.0, eps, ov, un, eps1, ov1, un1;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double s;
    double r;
    double head_t, tail_t;
    double head_t1, tail_t1;
    double head_t2, tail_t2;
    double head_q[2], tail_q[2];

    eps = std::pow(2.0, -24.0);        /* single precision */
    un = std::pow(2.0, -126.0);
    ov = std::pow(2.0, 128.0) * (1 - eps);
    eps1 = std::pow(2.0, -104.0);        /* extra precision */
    un1 = std::pow(2.0, -1022.0);
    ov1 = 1.79769313486231571e+308;
    /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
    abs_a = std::fabs(head_temp1[0]);
    abs_b = std::fabs(head_temp1[1]);
    abs_c = std::fabs(static_cast<double>(std::real(T_element)));
    abs_d = std::fabs(static_cast<double>(std::imag(T_element)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov1 / 16) {        /* scale down a, b */
      {
        double head_a, tail_a;
        double head_b, tail_b;
        head_a = head_temp1[0];
        tail_a = tail_temp1[0];
        compute_doubledouble_eq_doubledouble_div_double(&head_b, &tail_b, head_a, tail_a, 16.0);
        head_temp1[0] = head_b;
        tail_temp1[0] = tail_b;
        head_a = head_temp1[1];
        tail_a = tail_temp1[1];
        compute_doubledouble_eq_doubledouble_div_double(&head_b, &tail_b, head_a, tail_a, 16.0);
        head_temp1[1] = head_b;
        tail_temp1[1] = tail_b;
      }
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      T_element /= 16;
      S = S / 16;
    }
    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      s = 2.0 / (eps1 * eps1);
      {
        /* Compute complex-extra = complex-extra * real. */
        double head_a0, tail_a0;
        double head_a1, tail_a1;
        double head_t, tail_t;
        head_a0 = head_temp1[0];
        tail_a0 = tail_temp1[0];
        head_a1 = head_temp1[1];
        tail_a1 = tail_temp1[1];
        compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, s);
        head_temp1[0] = head_t;
        tail_temp1[0] = tail_t;
        compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, s);
        head_temp1[1] = head_t;
        tail_temp1[1] = tail_t;
      }

      S = S / s;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      s = 2.0 / (eps * eps);
      T_element *= s;
      S = S * s;
    }

    /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(T_element) / std::real(T_element);
      {
        double dt = (double) std::imag(T_element);
        compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
      }
      {
        double dt = (double) std::real(T_element);
        compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
      }
      head_t1 = head_temp1[1];
      tail_t1 = tail_temp1[1];        /* b */
      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
      head_t1 = head_temp1[0];
      tail_t1 = tail_temp1[0];        /* a */
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
      compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
      head_q[0] = head_t2;
      tail_q[0] = tail_t2;
      head_t1 = head_temp1[0];
      tail_t1 = tail_temp1[0];        /* a */
      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
      head_t1 = head_temp1[1];
      tail_t1 = tail_temp1[1];        /* b */
      {
        double head_bt, tail_bt;
        head_bt = -head_t2;
        tail_bt = -tail_t2;
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t1, tail_t1, head_bt, tail_bt);
      }                /* b - a*r */
      compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
      head_q[1] = head_t2;
      tail_q[1] = tail_t2;
    } else {
      r = std::real(T_element) / std::imag(T_element);
      {
        double dt = (double) std::real(T_element);
        compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, r, dt);
      }
      {
        double dt = (double) std::imag(T_element);
        compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_t, tail_t, dt);
      }
      head_t1 = head_temp1[0];
      tail_t1 = tail_temp1[0];        /* a */
      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
      head_t1 = head_temp1[1];
      tail_t1 = tail_temp1[1];        /* b */
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t1, tail_t1);
      compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
      head_q[0] = head_t2;
      tail_q[0] = tail_t2;
      head_t1 = head_temp1[1];
      tail_t1 = tail_temp1[1];        /* b */
      compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_t1, tail_t1, r);
      head_t1 = head_temp1[0];
      tail_t1 = tail_temp1[0];        /* a */
      {
        double head_bt, tail_bt;
        head_bt = -head_t1;
        tail_bt = -tail_t1;
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_bt, tail_bt);
      }                /* -a + b*r */
      compute_doubledouble_eq_doubledouble_div_doubledouble(&head_t2, &tail_t2, head_t2, tail_t2, head_t, tail_t);
      head_q[1] = head_t2;
      tail_q[1] = tail_t2;
    }
    /* Scale back */
    if (S == 1.0) {
      head_temp1[0] = head_q[0];
      tail_temp1[0] = tail_q[0];
      head_temp1[1] = head_q[1];
      tail_temp1[1] = tail_q[1];
    } else {
      /* Compute complex-extra = complex-extra * real. */
      double head_a0, tail_a0;
      double head_a1, tail_a1;
      double head_t, tail_t;
      head_a0 = head_q[0];
      tail_a0 = tail_q[0];
      head_a1 = head_q[1];
      tail_a1 = tail_q[1];
      compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, S);
      head_temp1[0] = head_t;
      tail_temp1[0] = tail_t;
      compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, S);
      head_temp1[1] = head_t;
      tail_temp1[1] = tail_t;
    }

    impl::inner_type_t<A> cr(head_temp1[0], tail_temp1[0]);
    impl::inner_type_t<A> ci(head_temp1[1], tail_temp1[1]);
    return A(cr, ci);
  } else if constexpr (std::is_same_v<inner_type_t<A>, double_double>) {
    return a / b;
  } else if constexpr (impl::is_complex_v<A>) {
    static_assert(!impl::is_complex_v<B>);
    return a / static_cast<impl::inner_type_t<A>>(b);
  } else {
    return a / static_cast<A>(b);
  }
}

//-----------------
} // namespace impl
//-----------------

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_IMPL_HPP
