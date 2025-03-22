#include "devoir_1.h"
#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "debug.h"

double sign(double x) {
    return (x >= 0) - (x < 0);
}

void tridiagonalize_full(double *A, int n, int k, double *d, double *e) {
    // since they are the same size
    double *x = d;

    // Temporary matricies with upper bound sizes
    double *Q = malloc((n - 1) * (n - 1) * sizeof(double));
    double *A_local = malloc((n - 1) * n * sizeof(double));

    for (int i = 0; i < n - 2; i++) {
        int x_size = n - 1 - i;

        // x = A_{i+1:n,i}
        cblas_dcopy(x_size, &A[(i + 1) * n + i], n, x, 1);

        double x_norm = cblas_dnrm2(x_size, x, 1);

        // vk = sign(x1) * ||x|| * e1 + x
        double *vk = x;
        vk[0] += sign(x[0]) * x_norm;

        // vk normalization
        double vk_norm = cblas_dnrm2(x_size, vk, 1);
        cblas_dscal(x_size, 1 / vk_norm, vk, 1);

        // Q = I - 2 * vk * vk^T
        for (int row = 0; row < x_size; row++) {
            for (int col = 0; col < x_size; col++) {
                if (row == col) {
                    Q[row * x_size + col] = 1.0 - 2.0 * vk[row] * vk[col];
                } else {
                    Q[row * x_size + col] = -2.0 * vk[row] * vk[col];
                }
            }
        }

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            x_size,
            x_size + 1,
            x_size,
            1.0,
            Q,
            x_size,
            &A[n*(i+1)+i],
            n,
            0.0,
            A_local,
            x_size + 1
        );

        // copy A_local into A
        for (int row = 0; row < x_size; row++) {
            for (int col = 0; col < x_size + 1; col++) {
                A[(n * (i + 1 + row)) + i + col] = A_local[row * (x_size + 1) + col];
            }
        }

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            n,
            x_size,
            x_size,
            1.0,
            &A[i+1],
            n,
            Q,
            x_size,
            0.0,
            A_local,
            x_size
        );

        // copy A_local into A
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < x_size; col++) {
                A[(n * row) + i + 1 + col] = A_local[row * x_size + col];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (i > 0) {
            e[i] = A[(i * n) + i - 1];
        }

        d[i] = A[(i * n) + i];
    }

    free(A_local);
    free(Q);
}

//test