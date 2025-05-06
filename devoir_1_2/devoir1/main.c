#include "devoir_1.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "debug.h"
#include <stdio.h>
#include <endian.h>
#include <assert.h>
#include <stdint.h>
#include <float.h>

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

void print_array(double *arr, int n, const char *name) { // écrit par chat gpt
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%.4f", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

// Fonction pour tester l'algorithme QR complet (plusieurs itérations) écrit par chat gpt 
void test_complete_qr(double *d, double *e, int n, double eps, int max_iter) {
    printf("\nTest de l'algorithme QR complet :\n");
    printf("=================================\n");
    
    // Sauvegarde des tableaux originaux
    double *d_orig = malloc(n * sizeof(double));
    double *e_orig = malloc(n * sizeof(double));
    memcpy(d_orig, d, n * sizeof(double));
    memcpy(e_orig, e, n * sizeof(double));
    
    printf("Valeurs diagonales initiales :\n");
    print_array(d, n, "d");
    printf("Valeurs super-diagonales initiales :\n");
    print_array(e, n, "e");
    
    int m = n;
    int total_iter = 0;
    
    while (m > 1 && total_iter < max_iter) {
        printf("\nItération %d, taille du problème : %d\n", total_iter + 1, m);
        
        int m_new = step_qr_tridiag(d, e, m, eps);
        
        printf("Après step_qr_tridiag :\n");
        print_array(d, n, "d");
        print_array(e, n, "e");
        
        if (m_new == m) {
            // Pas de convergence lors de cette étape
            total_iter++;
            printf("Pas de convergence, continue...\n");
        } else {
            // Une valeur propre a été isolée
            printf("Valeur propre isolée à l'indice %d : %.6f\n", m_new, d[m_new]);
            m = m_new;
            total_iter = 0; // Réinitialise le compteur d'itérations
        }
    }
    
    if (m == 1 || total_iter < max_iter) {
        printf("\nConvergence atteinte après %d itérations !\n", total_iter);
        printf("Valeurs propres calculées :\n");
        for (int i = 0; i < n; i++) {
            printf("λ%d = %.6f\n", i+1, d[i]);
        }
    } else {
        printf("\nL'algorithme n'a pas convergé après %d itérations.\n", max_iter);
    }
    
    // Restauration des tableaux originaux pour tests ultérieurs si nécessaire
    memcpy(d, d_orig, n * sizeof(double));
    memcpy(e, e_orig, n * sizeof(double));
    
    free(d_orig);
    free(e_orig);
}

int eigenvalues_qr(double *d, double *e, int n, double eps, int max_iter) {
    int m = n;
    int total_iter = 0;
    int global_iter = 0;
    
    while (m > 1 && total_iter < max_iter) {
        int m_new = step_qr_tridiag(d, e, m, eps);
        
        if (m_new == m) {
            total_iter++;
        } else {
            m = m_new;
            total_iter = 0;
        }

        global_iter++;
    }
    
    if (m == 1 || total_iter < max_iter) {
        return global_iter;
        // for (int i = 0; i < n; i++) {
        //     printf("%lf\n", d[i]);
        // }
    } else {
        printf("Pas convergé :(\n");
        exit(1);
    }
}

double *load_matrix(const char *filename, int *n) {
    FILE *file = fopen(filename, "r");
    assert(file != NULL);

    uint32_t n_temp;
    fread(&n_temp, sizeof(uint32_t), 1, file);

    *n = n_temp;

    double *A = malloc(n_temp * n_temp * sizeof(double));
    assert(A != NULL);

    fread(A, sizeof(double), n_temp * n_temp, file);

    return A;
}

#define SQUARE(x) ((x) * (x))

double *create_matrix_laplace_2d(int nx, int ny, double lx, double ly) {
    int lda, k;
    int size = nx * ny; // Number of nodes/unknowns
    double dx2 = SQUARE(lx / (nx + 1));
    double dy2 = SQUARE(ly / (ny + 1));
    double alpha, beta, gamma;
    double *L;

    // Choice of node numbering, here nx=4, ny=5
    // j\i    0   1   2   3
    //     .  .   .   .   .  .
    // 0   .  0   1   2   3  .
    // 1   .  4   5   6   7  .
    // 2   .  8   9  10  11  .
    // 3   . 12  13  14  15  .
    // 4   . 16  17  18  19  .
    //     .  .   .   .   .  .

    k = nx;
    alpha = 1. / dx2;
    beta = 1. / dy2;
    gamma = 2 * (alpha + beta);

    lda = size;
    L = (double *)calloc(size * lda, sizeof(double));
    for (int idx, i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            idx = i * k + j;
            L[idx * lda + idx] = gamma; // (i,j)->(i  ,j)
            if (0 < i)
                L[idx * lda + idx - k] = -beta; // (i,j)->(i-1,j)
            if (i < ny - 1)
                L[idx * lda + idx + k] = -beta; // (i,j)->(i+1,j)
            if (0 < j)
                L[idx * lda + idx - 1] = -alpha; // (i,j)->(i,j-1)
            if (j < nx - 1)
                L[idx * lda + idx + 1] = -alpha; // (i,j)->(i,j+1)
        }
    }
    return L;
}

double *create_matrix_laplace_1d(int n, double lx) {
    double dx2 = SQUARE(lx / (n + 2));
    double *L;

    double coeff = 1. / dx2;

    L = (double *)calloc(n * n, sizeof(double));
    for (int i = 0; i < n; i++) {
        if (i > 0)
            L[i * n + i - 1] = coeff;

        if (i < n-1)
            L[i * n + i + 1] = coeff;

        L[i * n + i] = -2.0 * coeff;
    }

    return L;
}

void laplace_2d_bench() {
    for (int n = 2; n < 30; n++) {
        struct timespec start, end;

        double *d = malloc(n * n * sizeof(double));

        double *A = create_matrix_laplace_2d(n, n, 1.0, 1.0);

        clock_gettime(CLOCK_MONOTONIC, &start);
        qr_eigs_full(A, n, -1, 1e-12, 1000, d);

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("{\"n\": %d, \"time\": %.9f}\n", n, elapsed);

        free(A);
        free(d);
    }
}

void laplace_1d_bench() {
    for (int n = 3; n < 1000; n++) {
        double *d = malloc(n * sizeof(double));
        double *e = malloc(n * sizeof(double));

        double *A = create_matrix_laplace_1d(n, 2.0 / n);

        print_mat(A, n, n, "A");

        int iter = qr_eigs_full(A, n, -1, 1e-12, 1000, d);

        for (int i = 0; i < n; i++) {
            printf("%f\n", d[i]);
        }

        printf("---\n");

        printf("{\"n\": %d, \"iter\": %d}\n", n, iter);

        break;

        free(A);
        free(e);
        free(d);
    }
}

int main(int argc, char **argv) {
    assert(argc == 2);

    int n;
    double *A = load_matrix(argv[1], &n);
    assert(A != NULL);

    double *d = malloc(n * sizeof(double));
    assert(d != NULL);


    int iter = qr_eigs_full(A, n, -1, 1e-12, 1000, d);

    for (int i = 0; i < n; i++) {
        printf("%.*g\n", DBL_DECIMAL_DIG, d[i]);
    }

    free(d);
    free(A);
}