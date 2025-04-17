#include "devoir_2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



double dot_product(int n, const double *x, const double *y) {
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

//y = y + alpha * x (axpy: alpha * x + y)
void axpy(int n, double alpha, const double *x, double *y) {
    for (int i = 0; i < n; ++i) {
       y[i] += alpha * x[i];
   }
} //on peut utiliser cblas si on veut

//x = alpha * x
void scal(int n, double alpha, double *x) {
    for (int i = 0; i < n; ++i) {
       x[i] *= alpha;
   }
}

// Copie le vecteur src dans dest
void copy_vector(int n, const double *src, double *dest) {
    for (int i = 0; i < n; ++i) {
       dest[i] = src[i];
   }
}

void Matvec(
    int n, int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *v,
    double *Av)
{

    for(int i = 0; i<n; i++){
        for(int j = rows_idx[i]; j < rows_idx[i+1]; j++){
            double val_A = A[j];    
            int col_idx = cols[j]; 
            Av[i] += val_A * v[col_idx];
        }
    }
}

void solve(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *L,
    const double *b,
    double *x)
{

    

}

int CG(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *b,
    double *x,
    double eps)
{
    double *r = (double*)malloc(n * sizeof(double));
    double *p = (double*)malloc(n * sizeof(double));
    double *Ap = (double*)malloc(n * sizeof(double));

    if (!r || !p || !Ap) {
        fprintf(stderr, "Erreur d'allocation mémoire dans CG\n");
        // Libérer ce qui a pu être alloué
        free(r);
        free(p);
        free(Ap);
        return -1; // Code d'erreur
    }

    // x_0 est l'estimation initiale passée en argument.
    // Calculer r_0 = b - Ax_0

    memset(r, 0, n * sizeof(double)); // Initialiser r à 0 avant Matvec si Matvec accumule
    Matvec(n, nnz, rows_idx, cols, A, x, r); // r contient Ax_0
    for(int i=0; i<n; ++i) {
        r[i] = b[i] - r[i]; // r = b - Ax_0
    }

    
    return 0;
}

void ILU(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    double *L)
{
    memcpy(L, A, nnz * sizeof(double));

    for (int k = 0; k < n; k++) {
        double L_kk = 0.0;
        int ptr_kk = -1;
        for (int ptr = rows_idx[k]; ptr < rows_idx[k+1]; ptr++) {
            if (cols[ptr] == k) {
                L_kk = L[ptr];
                ptr_kk = ptr;
                break;
            }
        }

        if (ptr_kk == -1 || fabs(L_kk) < 1e-12) continue;

        for (int j = k + 1; j < n; j++) {
            int ptr_jk = -1;
            for (int ptr = rows_idx[j]; ptr < rows_idx[j+1]; ptr++) {
                if (cols[ptr] == k) {
                    ptr_jk = ptr;
                    break;
                }
            }

            if (ptr_jk == -1) continue;

            L[ptr_jk] /= L_kk;
            double L_jk = L[ptr_jk];

            for (int ptr_i = rows_idx[k]; ptr_i < rows_idx[k+1]; ptr_i++) {
                int i = cols[ptr_i];
                if (i <= k) continue;

                int ptr_ji = -1;
                for (int ptr = rows_idx[j]; ptr < rows_idx[j+1]; ptr++) {
                    if (cols[ptr] == i) {
                        ptr_ji = ptr;
                        break;
                    }
                }

                if (ptr_ji == -1) continue;

                double L_ki = 0.0;
                for (int ptr = rows_idx[k]; ptr < rows_idx[k+1]; ptr++) {
                    if (cols[ptr] == i) {
                        L_ki = L[ptr];
                        break;
                    }
                }

                L[ptr_ji] -= L_jk * L_ki;
            }
        }
    }
}

int PCG(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *b,
    double *x,
    double eps)
{
    return 0;
}

int csr_sym()
{
    return 0; //  Both parts
    // return 1; // Lower part only
}


