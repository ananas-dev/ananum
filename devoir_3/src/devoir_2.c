#include "devoir_2.h"
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// SYM = 1 --> Vous ne stockez que la partie inferieure de la matrice
// SYM = 0 --> Vous stockez toutes les entrées non nulles de la matrice
#define SYM 1 
int csr_sym() { return SYM; }

/**
 * Si vous avez installé la librairie BLAS, vous pouvez l'ajouter au Makefile
 * et supprimer les implémentations de cblas_* ci-dessous.
 */

// ----------------------- BLAS start -----------------------

void cblas_dscal(int n, double alpha, double *x, int incx) {
    for (int i = 0; i < n; i++) {
        x[i * incx] *= alpha;
    }
}
void cblas_dcopy(int n, const double *x, int incx, double *y, int incy) {
    for (int i = 0; i < n; i++) {
        y[i * incy] = x[i * incx];
    }
}
double cblas_ddot(int n, const double *x, int incx, const double *y, int incy) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += x[i * incx] * y[i * incy];
    }
    return result;
}
void cblas_daxpy(
    int n, double alpha, const double *x, int incx, double *y, int incy
) {
    for (int i = 0; i < n; i++) {
        y[i * incy] += alpha * x[i * incx];
    }
}
double cblas_dnrm2(int n, const double *x, int incx) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += x[i * incx] * x[i * incx];
    }
    return sqrt(result);
}

// ------------------------ BLAS end ------------------------

void Matvec(
    int n,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *v,
    double *Av
) {

    cblas_dscal(n, 0, Av, 1);

    for (int i = 0; i < n; i++) {
#if SYM
        for (int j = rows_idx[i]; j < rows_idx[i + 1] - 1; j++) {
            Av[i] += A[j] * v[cols[j]];
            Av[cols[j]] += A[j] * v[i];
        }
        Av[i] += A[rows_idx[i + 1] - 1] * v[i];
#else
        for (int j = rows_idx[i]; j < rows_idx[i + 1]; j++) {
            Av[i] += A[j] * v[cols[j]];
        }
#endif
    }
}

int CG(
    int n,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *b,
    double *x,
    double eps
) {
    int idx = 0;
    double alpha, beta;
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));
    double *r = (double *)malloc(n * sizeof(double));

    cblas_dscal(n, 0, x, 1);
    cblas_dcopy(n, b, 1, r, 1);
    cblas_dcopy(n, r, 1, p, 1);
    double r0 = cblas_dnrm2(n, b, 1);
    double r_norm2 = cblas_ddot(n, r, 1, r, 1);
    // printf("r0 = %9.3le\n", r0);

    while (sqrt(r_norm2) / r0 > eps) {
        // if (idx %100 == 0) printf("idx : %4d\n", idx);
        Matvec(n, rows_idx, cols, A, p, Ap);
        alpha = r_norm2 / cblas_ddot(n, p, 1, Ap, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, Ap, 1, r, 1);
        beta = 1 / r_norm2;
        r_norm2 = cblas_ddot(n, r, 1, r, 1);
        beta *= r_norm2;
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);
        idx++;
        // printf("it : %3d  -> res = %9.3le\n", idx, sqrt(r_norm2) / r0);
    }

    // printf("cg it : %d\n", idx);
    free(p);
    free(Ap);
    free(r);
    return idx;
}
