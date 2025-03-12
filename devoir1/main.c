#include "devoir_1.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "debug.h"

void fill_symmetric_matrix(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[i * n + j] = A[j * n + i] = i + j;
        }
    }
}

void fill_symmetric_matrix_rand(double *A, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double value = (double)(rand() % 10);
            A[i * n + j] = value;
            A[j * n + i] = value;
        }
    }
}


int main() {
    int n = 5;

    // double *A = malloc(sizeof(double) * n * n);

    // double *A;

    double A[] = {
        6.0,  0.0,  5.0,  0.0,  9.0,
        0.0,  5.0,  5.0,  0.0,  7.0,
        5.0,  5.0,  9.0,  0.0,  2.0,
        0.0,  0.0,  0.0,  6.0,  2.0,
        9.0,  7.0,  2.0,  2.0,  8.0
    };


    double *d = malloc(sizeof(double) * n);
    double *e = malloc(sizeof(double) * n);

    // fill_symmetric_matrix(A, n);

    // fill_symmetric_matrix_rand(A, n);


    print_mat(A, n, n, "A");

    tridiagonalize_full(A, n, 0, d, e);

    print_mat(A, n, n, "H");

    free(e);
    free(d);
    // free(A);
}
