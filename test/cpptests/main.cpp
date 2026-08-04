#include <stdio.h>
#include <stdlib.h>

#ifdef XBLAS_COMBINE_TESTS

extern int main_axpby(int, const char**);
extern int main_dot(int, const char**);
extern int main_dot2(int, const char**);
extern int main_gbmv(int, const char**);
extern int main_gbmv2(int, const char**);
extern int main_ge_sum_mv(int, const char**);
extern int main_gemm(int, const char**);
extern int main_gemv(int, const char**);
extern int main_gemv2(int, const char**);
extern int main_hbmv(int, const char**);
extern int main_hemm(int, const char**);
extern int main_hemv(int, const char**);
extern int main_hemv2(int, const char**);
extern int main_hpmv(int, const char**);
extern int main_sbmv(int, const char**);
extern int main_spmv(int, const char**);
extern int main_sum(int, const char**);
extern int main_symm(int, const char**);
extern int main_symv(int, const char**);
extern int main_symv2(int, const char**);
extern int main_tbsv(int, const char**);
extern int main_tpmv(int, const char**);
extern int main_trmv(int, const char**);
extern int main_trsv(int, const char**);
extern int main_waxpby(int, const char**);

int main()
{
    int nr_failed_routines = 0;

    printf("###############################\n");
    const char* axpby_arr[] = { "main_axpby", "10", "1", "1.0", "1", "1" };
    nr_failed_routines += main_axpby(6, axpby_arr);

    printf("###############################\n");
    const char* dot_arr[] = { "main_dot", "20", "1", "1.0", "0", "1" };
    nr_failed_routines += main_dot(6, dot_arr);

    printf("###############################\n");
    const char* dot2_arr[] = { "main_dot2", "20", "1", "1.0", "0", "1" };
    nr_failed_routines += main_dot2(6, dot2_arr);

    printf("###############################\n");
    const char* gbmv_arr[] = { "main_gbmv", "12", "1", "1.0", "0", "0.1" };
    nr_failed_routines += main_gbmv(6, gbmv_arr);

    printf("###############################\n");
    const char* gbmv2_arr[] = { "main_gbmv2", "12", "1", "1.0", "0", "0.1" };
    nr_failed_routines += main_gbmv2(6, gbmv2_arr);

    printf("###############################\n");
    const char* ge_sum_mv_arr[] = { "main_ge_sum_mv", "7", "1", "1.0", "3", "0.1" };
    main_ge_sum_mv(6, ge_sum_mv_arr);

    printf("###############################\n");
    const char* gemm_arr[] = { "main_gemm", "9", "1", "1.0", "0", "0.01" };
    nr_failed_routines += main_gemm(6, gemm_arr);

    printf("###############################\n");
    const char* gemv_arr[] = { "main_gemv", "12", "1", "1.0", "0", "0.1" };
    nr_failed_routines += main_gemv(6, gemv_arr);

    printf("###############################\n");
    const char* gemv2_arr[] = { "main_gemv2", "12", "1", "1.0", "0", "0.1" };
    nr_failed_routines += main_gemv2(6, gemv2_arr);

    printf("###############################\n");
    const char* hbmv_arr[] = { "main_hmbv", "7", "1", "1.0", "3", "0.1" };
    nr_failed_routines += main_hbmv(6, hbmv_arr);

    printf("###############################\n");
    const char* hemm_arr[] = { "main_hemm", "7", "1", "1.0", "0", "0.1" };
    nr_failed_routines += main_hemm(6, hemm_arr);

    printf("###############################\n");
    const char* hemv_arr[] = { "main_hemv", "7", "1", "1.0", "3", "0.1" };
    nr_failed_routines += main_hemv(6, hemv_arr);

    printf("###############################\n");
    const char* hemv2_arr[] = { "main_hemv2", "7", "1", "1.0", "3", "1" };
    nr_failed_routines += main_hemv2(6, hemv2_arr);

    printf("###############################\n");
    const char* hpmv_arr[] = { "main_hpmv", "7", "1", "1.0", "3", "0.1" };
    nr_failed_routines += main_hpmv(6, hpmv_arr);

    printf("###############################\n");
    const char* sbmv_arr[] = { "main_sbmv", "7", "1", "1.0", "3", "0.01" };
    nr_failed_routines += main_sbmv(6, sbmv_arr);

    printf("###############################\n");
    const char* spmv_arr[] = { "main_spmv", "7", "1", "1.0", "3", "0.1" };
    nr_failed_routines += main_spmv(6, spmv_arr);

    printf("###############################\n");
    const char* sum_arr[] = { "main_sum_arr", "20", "1", "1.0", "0", "1.0" };
    nr_failed_routines += main_sum(6, sum_arr);

    printf("###############################\n");
    const char* symm_arr[] = { "main_symm", "7", "1", "1.0", "3", "0.01" };
    nr_failed_routines += main_symm(6, symm_arr);

    printf("###############################\n");
    const char* symv_arr[] = { "main_symv", "7", "1", "1.0", "3", "0.1" };
    nr_failed_routines += main_symv(6, symv_arr);

    printf("###############################\n");
    const char* symv2_arr[] = { "main_symv2", "7", "1", "1.0", "0", "1" };
    nr_failed_routines += main_symv2(6, symv2_arr);

    printf("###############################\n");
    const char* tbsv_arr[] = { "main_tbsv", "9", "1", "1.0", "3", "0.1" };
    nr_failed_routines += main_tbsv(6, tbsv_arr);

    printf("###############################\n");
    const char* tpmv_arr[] = { "main_tpmv", "12", "1", "1.0", "0", "0.1" };
    nr_failed_routines += main_tpmv(6, tpmv_arr);

    printf("###############################\n");
    const char* trmv_arr[] = { "main_trmv", "12", "1", "1.0", "0", "0.1" };
    nr_failed_routines += main_trmv(6, trmv_arr);

    printf("###############################\n");
    const char* trsv_arr[] = { "main_trsv", "10", "1", "1.0", "3", "0.1" };
    nr_failed_routines += main_trsv(6, trsv_arr);

    printf("###############################\n");
    const char* waxpby_arr[] = { "main_waxpby", "20", "1", "1.0", "0", "1" };
    nr_failed_routines += main_waxpby(6, waxpby_arr);

    return nr_failed_routines;
}

#endif
