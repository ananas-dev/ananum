#include "devoir_1.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "debug.h"
#include <stdio.h>
#include <endian.h>
#include <assert.h>
#include <stdint.h>

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

void eigenvalues_qr(double *d, double *e, int n, double eps, int max_iter) {
    int m = n;
    int total_iter = 0;
    
    while (m > 1 && total_iter < max_iter) {
        int m_new = step_qr_tridiag(d, e, m, eps);
        
        if (m_new == m) {
            total_iter++;
        } else {
            m = m_new;
            total_iter = 0;
        }
    }
    
    if (m == 1 || total_iter < max_iter) {
        for (int i = 0; i < n; i++) {
            printf("%lf\n", d[i]);
        }
    } else {
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

int main(int argc, char **argv) {
    assert(argc == 2);

    int n;
    double *A = load_matrix(argv[1], &n);
    assert(A != NULL);

    double *d = malloc(n * sizeof(double));
    double *e = malloc(n * sizeof(double));

    tridiagonalize_full(A, n, -1, d, e);
    eigenvalues_qr(d, e, n, 1e-12, 1000);

    free(e);
    free(d);
    free(A);
}