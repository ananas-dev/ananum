#include<stdio.h>
#define TUNIT_IMPLEMENTATION
#include "devoir_2.h"
#include "tunit.h"

void test_Matvec1() {
    int n = 3;
    int nnz = 9; // full matrix
    double A[9] = {
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, //
    };
    int cols_idx[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int rows_idx[4] = {0, 3, 6, 9};
    double v[3] = {1.0, 2.0, 3.0};
    double Av[3] = {0.0, 0.0, 0.0};
  
    // Matvec(int n, int nnz, double *A, int *cols_idx, int *rows_idx, double *v,
    // double *Av)
    

    Matvec(n, nnz, rows_idx, cols_idx, A, v, Av);
    printf("%f, %f, %f\n", Av[0], Av[1], Av[2]);
    t_assert_double(Av[0], ==, 14.0);
    t_assert_double(Av[1], ==, 32.0);
    t_assert_double(Av[2], ==, 50.0);
}

//
void test_CG1() {

  int n = 3;
  int nnz = 4;
  double A[4] = {1.0, 1.0, 1.0};
  int cols_idx[4] = {0, 1, 2};
  int rows_idx[4] = {0, 1, 2, 3};
  // A is the identity matrix

  double v[3] = {1.0, 2.0, 3.0};
  double Av[3] = {100101.0, 0.0,
                  0.0}; // should be set as (0, 0, 0) at the beginning of CG

  // int CG(int n, int nnz, double eps, double *A, int *cols, int *rows_idx,
  //        double *b, double *x);

  //int n,int nnz,const int *rows_idx,const int *cols,const double *A,const double *b,double *x, double eps)
  CG(n,nnz, rows_idx, cols_idx, A, Av, v, 1.0);

  t_assert_double(Av[0], ==, v[0]);
  t_assert_double(Av[1], ==, v[1]);
  t_assert_double(Av[2], ==, v[2]);
}

void test_ILU() {
  int n = 4;
  int nnz = 10;

  // CSR representation
  double A[10] = {4, -1,         // Row 0
                  -1, 4, -1,     // Row 1
                  -1, 4, -1,     // Row 2
                  -1, 3};        // Row 3

  int cols[10] = {0, 1,          // Row 0
                  0, 1, 2,       // Row 1
                  1, 2, 3,       // Row 2
                  2, 3};         // Row 3

  int rows_idx[5] = {0, 2, 5, 8, 10}; // Row start indices (n+1 entries)

  double L[10]; // ILU output in same CSR pattern

  ILU(n, nnz, rows_idx, cols, A, L);

  for (int i = 0; i < nnz; i++) {
    printf("%lf\n", L[i]);
  }

  t_assert_double(L[0], ==, 1.0);
  // t_assert_double(L[0], ==, 1.0);
}
  

int main(int argc, char **argv) {
  testsuite_t *csr_suite = t_registerTestSuite("Matvec");
  t_addTestToSuite(csr_suite, "matvec CSR multiplication 1", test_Matvec1);
  t_addTestToSuite(csr_suite, "CG test", test_CG1);
  t_addTestToSuite(csr_suite, "ILU test", test_ILU);

  return t_runSuites(argc, argv);
}


