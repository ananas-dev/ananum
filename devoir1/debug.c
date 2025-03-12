#include "debug.h"

#include <stdio.h>
#include <math.h>

#define TOL_PRINT 1e-8

void print_mat(double *A, int n, int m, char *name) {
    printf("\nMatrix %s\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (fabs(A[i * m + j]) < TOL_PRINT)
                printf("%6s ", "0.00");
            else
                printf("%6.2lf ", A[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vec(double *x, int n, char *name) {
    printf("\nVector %s\n", name);
    for (int i = 0; i < n; i++) {
        if (fabs(x[i]) < TOL_PRINT)
            printf("%6s ", "0.0");
        else
            printf("%6.2lf ", x[i]);
    }
    printf("\n");
}