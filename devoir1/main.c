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
// int test_complete_qr(double *A, int n, int k, double eps, int max_iter, double *d) {

//     double *dA = malloc(sizeof(double) * n);
//     double *eA = malloc(sizeof(double) * n);
//     tridiagonalize_full(A, n, k, dA, eA);


//     printf("\nTest de l'algorithme QR complet :\n");
//     printf("=================================\n");
    
//     // Sauvegarde des tableaux originaux
//     double *d_orig = malloc(n * sizeof(double));
//     double *e_orig = malloc(n * sizeof(double));
//     memcpy(d_orig, dA, n * sizeof(double));
//     memcpy(e_orig, eA, n * sizeof(double));

//     int m = n;
//     int total_iter = 0;
    
//     while (m > 1 && total_iter < max_iter) {
//         int m_new = step_qr_tridiag(dA, eA, m, eps);
        
//         if (m_new == m) {
//             // Pas de convergence lors de cette étape
//             total_iter++;
//         } else {
//             // Une valeur propre a été isolée
//             printf("Valeur propre isolée à l'indice %d : %.6f\n", m_new, d[m_new]);
//             m = m_new;
//             total_iter++;
//         }
//     }

//     memcpy(dA, d_orig, n * sizeof(double));
//     memcpy(eA, e_orig, n * sizeof(double));
    
//     free(d_orig);
//     free(e_orig);

//     return total_iter;
// }


int main() {  //tests écrits par chat gpt

    // Matrice symétrique de test
    // double A[] = {
    //     6.0,  0.0,  5.0,  0.0,  9.0,
    //     0.0,  5.0,  5.0,  0.0,  7.0,
    //     5.0,  5.0,  9.0,  0.0,  2.0,
    //     0.0,  0.0,  0.0,  6.0,  2.0,
    //     9.0,  7.0,  2.0,  2.0,  8.0
    // };

    // int n = 4;
    // double eps = 1e-20;
    // int max_iter = 100;

    // double A[] = {
    //     2.0,  2.0,  0.0, 0.0,
    //     2.0,  8.0,  0.0, 0.0,
    //     0.0,  0.0,  9.0, 6.0,
    //     0.0,  0.0,  6.0, 6.0
    // };


    int n = 3;
    double eps = 1e-20;
    int max_iter = 1000;

    double A[] = {
        2.0,  0.0,  0.0,
        0.0,  8.0,  2.0,
        0.0,  2.0,  9.0
    };



    // int n = 3;
    // double eps = 1e-20;
    // int max_iter = 100;  le code marche pour des matrices diagonales

    // double A[] = {
    //     2.0,  0.0,  0.0,
    //     0.0,  8.0,  0.0,
    //     0.0,  0.0,  9.0
    // };


    double *d = malloc(sizeof(double) * n);
    double *e = malloc(sizeof(double) * n);

    // printf("Matrice originale :\n");
    // print_mat(A, n, n, "A");

    // // Tridiagonalisation de la matrice
    // tridiagonalize_full(A, n, 0, d, e);
    
    // printf("\nMatrice tridiagonalisée :\n"); 
    // print_mat(A, n, n, "H");
    
    //int qr_eigs_full(double *A, int n, int k, double eps, int max_iter, double *d){
    double *A_copy = malloc(sizeof(A));
    memcpy(A_copy, A, sizeof(A));

    
    tridiagonalize_full(A_copy, n, 0, d, e);
    printf("\nMatrice tridiagonalisée :\n"); 
    print_mat(A_copy, n, n, "H");    

    int iter = qr_eigs_full(A_copy, n, 0, eps, max_iter,d);
    for(int i = 0; i<n; i++){
        printf("valeur propre isolée numéro %d : %f\n", i, d[i]);
    }    
    printf("nombre max d'itération : %d\n", iter);
    free(d);
    free(A_copy);

    return 0;
}


