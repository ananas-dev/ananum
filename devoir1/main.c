#include "devoir_1.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "debug.h"
#include <stdio.h> //

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


int main() {  //tests écrits par chat gpt
    int n = 3;
    double eps = 1e-20;
    int max_iter = 20;

    // Matrice symétrique de test
    // double A[] = {
    //     6.0,  0.0,  5.0,  0.0,  9.0,
    //     0.0,  5.0,  5.0,  0.0,  7.0,
    //     5.0,  5.0,  9.0,  0.0,  2.0,
    //     0.0,  0.0,  0.0,  6.0,  2.0,
    //     9.0,  7.0,  2.0,  2.0,  8.0
    // };

    double A[] = {
        2.0,  0.0,  0.0,
        2.0,  5.0,  0.0, 
        0.0,  0.0,  9.0 
    };

    double *d = malloc(sizeof(double) * n);
    double *e = malloc(sizeof(double) * n);

    printf("Matrice originale :\n");
    print_mat(A, n, n, "A");

    // Tridiagonalisation de la matrice
    tridiagonalize_full(A, n, 0, d, e);
    
    printf("\nMatrice tridiagonalisée :\n");
    print_mat(A, n, n, "H");
    
    printf("\nValeurs diagonales et super-diagonales extraites :\n");
    print_array(d, n, "d");
    print_array(e, n, "e");
    
    // Test d'une étape unique de QR
    printf("\nTest d'une étape unique de QR :\n");
    printf("==============================\n");
    printf("Avant step_qr_tridiag :\n");
    print_array(d, n, "d");
    print_array(e, n, "e");
    
    int result = step_qr_tridiag(d, e, n, eps);
    
    printf("\nAprès step_qr_tridiag :\n");
    print_array(d, n, "d");
    print_array(e, n, "e");
    printf("Résultat : %d\n", result);
    
    if (result < n) {
        printf("Une valeur propre a été isolée à l'indice %d : %.6f\n", result, d[result]);
    } else {
        printf("Aucune valeur propre n'a été isolée.\n");
    }
    
    // Réinitialisation des tableaux pour le test complet
    tridiagonalize_full(A, n, 0, d, e);
    
    // Test de l'algorithme QR complet
    test_complete_qr(d, e, n, eps, max_iter);
    
    // Libération de la mémoire
    free(e);
    free(d);

    return 0;
}