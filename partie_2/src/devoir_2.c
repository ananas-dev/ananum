#include "devoir_2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>


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

int CG(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *b,
    double *x, // x est a la fois l'estimation initiale ET la solution finale
    double eps)
{
    // double *r = (double*)malloc(n * sizeof(double));
    // double *p = (double*)malloc(n * sizeof(double));
    // double *Ap = (double*)malloc(n * sizeof(double));
    double *Ap = (double*)calloc(n, sizeof(double));
    double *r = (double*)calloc(n, sizeof(double));
    double *p = (double*)calloc(n, sizeof(double));

    if (!r || !p || !Ap) {
        fprintf(stderr, "Erreur d'allocation mémoire dans CG\n");
        free(r); 
        free(p);
        free(Ap);
        return -1; 
    }

    // x_0 est l'estimation initiale (contenue dans x à l'entrée).
    // on pourrait partir de x_0 = 0, il faudrait faire memset(x, 0, n * sizeof(double)) 

    // r_0 = b - Ax_0
    // calculer Ax_0, résultat dans r temporairement pour évieter de faire trop de malloc inutile
    Matvec(n, nnz, rows_idx, cols, A, x, r); // r contient Ax_0
    // calculer b - Ax_0 -> le résultat final va dans r
    for(int i=0; i<n; i++) {
        r[i] = b[i] - r[i]; // r = b - Ax_0
    }

    // p_0 = r_0
    copy_vector(n, r, p);

    // initialisation
    double r_sq_old = dot_product(n, r, r);
    double initial_r_norm = sqrt(r_sq_old); // Norme initiale pour le critère d'arrêt
    double r_sq_new;
    double alpha, beta;
    double pAp;

    // Si le résidu initial est déjà très petit, on a fini
    if (initial_r_norm < 1e-15) { //chat gpt mais jsp pourquoi on utilise pas eps
        free(r);
        free(p);
        free(Ap);
        return 0; // convergence immédiate
    }

    int k = 0;
    // int max_iter = 10*n; // Limite pour éviter boucle infinie
    int max_iter = 2 * n;

    while (k < max_iter) {
        // Ap_k = A * p_k
        // Il faut remettre Ap à zéro car Matvec accumule et ne set rien à 0
        memset(Ap, 0, n * sizeof(double)); 
        Matvec(n, nnz, rows_idx, cols, A, p, Ap); 

        // p_k^T * Ap_k
        pAp = dot_product(n, p, Ap);

        // alpha_k = r_k^T * r_k / (p_k^T * Ap_k)
        alpha = r_sq_old / pAp; // r_sq_old contient r_k^T * r_k de l'itération précédente

        // x_{k+1} = x_k + alpha_k * p_k
        axpy(n, alpha, p, x); // x = x + alpha * p

        // r_{k+1} = r_k - alpha_k * Ap_k
        axpy(n, -alpha, Ap, r); // r = r - alpha * Ap

        // r_{k+1}^T * r_{k+1}
        r_sq_new = dot_product(n, r, r);

        // critère d'arrêt
        double current_r_norm = sqrt(r_sq_new);

        if (current_r_norm / initial_r_norm < eps) {
            k++; // On compte cette dernière itération
            break; // Convergence atteinte
        }

        // beta_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
        beta = r_sq_new / r_sq_old;

        //  p_{k+1} = r_{k+1} + beta_k * p_k
        for(int i=0; i<n; ++i) {
            p[i] = r[i] + beta * p[i];
        }


        r_sq_old = r_sq_new;
        k++;
    }

     if (k == max_iter) {
         fprintf(stderr, "CG: Convergence non atteinte après %d itérations.\n", max_iter);
    }


    free(r);
    free(p);
    free(Ap);

    return k; 
}

int CSR_LU_solve(int n, int nnz, const int *rows_idx, const int *cols, const double *LU, double *b) {
    int i, j, k;
    double sum;
    
    // Forward substitution
    for (i = 0; i < n; i++) {
        sum = b[i];
        for (j = rows_idx[i]; j < rows_idx[i+1]; j++) {
            if (cols[j] < i) {
                sum -= LU[j] * b[cols[j]];
            } else {
                break;
            }
        }
        b[i] = sum;
    }
    
    // Backward substitution
    for (i = n - 1; i >= 0; i--) {
        sum = b[i];
        int diag_pos = -1;
        
        for (j = rows_idx[i]; j < rows_idx[i+1]; j++) {
            if (cols[j] == i) {
                diag_pos = j;
                break;
            }
        }
        
        // Singular matrix
        if (diag_pos == -1 || fabs(LU[diag_pos]) < 1e-14) {
            fprintf(stderr, "Zero pivot at row %d\n", i);
            return 1;
        }
        
        for (j = diag_pos + 1; j < rows_idx[i+1]; j++) {
            if (cols[j] > i) {
                sum -= LU[j] * b[cols[j]];
            }
        }
        
        b[i] = sum / LU[diag_pos];
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
    clock_t start, end;
    double elapsed;
    start = clock(); 
    
    
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
    end = clock();
    elapsed = ((double)end - start) / CLOCKS_PER_SEC; /* Conversion en seconde  */
 
 
    printf("ILU prend %.2fsecondes\n", elapsed);
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
    int iter = 0;

    double *M = malloc(nnz * sizeof(double));
    double* r = calloc(n, sizeof(double));
    double* z = calloc(n, sizeof(double));
    double* d = calloc(n, sizeof(double));
    double* Ad = calloc(n, sizeof(double));
    

    // Dispable ILU for now

    ILU(n, nnz, rows_idx, cols, A, M);

    // r_0 = b - Ax_0
    Matvec(n, nnz, rows_idx, cols, A, x, r);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - r[i];
    }

    // Solve Mz_0 = r_0
    memcpy(z, r, n * sizeof(double));
    assert(CSR_LU_solve(n, nnz, rows_idx, cols, M, z) == 0);

    // d_0 = z_0
    memcpy(d, z, n * sizeof(double));

    double last_dot_rz = dot_product(n, r, z);
    double r_0_norm = sqrt(dot_product(n, r, r));

    for(;;) {
        // Compute Ad
        memset(Ad, 0, sizeof(double) * n);
        Matvec(n, nnz, rows_idx, cols, A, d, Ad);

        // alpha_k = (r^Tz)/(d^TAd)
        double dot_d_Ad = dot_product(n, d, Ad);
        double alpha;
        alpha = last_dot_rz / dot_d_Ad;

        // x_k = x_(k-1) + alpha_k * d_(k-1);
        axpy(n, alpha, d, x);

        // r_k = r_(k-1) - alpha_k * Ad_(k-1)
        axpy(n, -alpha, Ad, r);

        if (sqrt(dot_product(n, r, r)) / r_0_norm < eps) {
            iter++;
            break;
        }

        // Solve Mz_k = r_k
        memcpy(z, r, n * sizeof(double));
        assert(CSR_LU_solve(n, nnz, rows_idx, cols, M, z) == 0);

        // beta_k = (r^T_k z_k)/(r^T_(k-1)z_(k-1))
        double dot_rz = dot_product(n, r, z);
        double beta;
        beta = dot_rz / last_dot_rz;
        last_dot_rz = dot_rz;

        // d_k = z + beta_k * d_(k-1);
        for (int i = 0; i < n; i++) {
            d[i] = z[i] + beta * d[i];
        }

        iter++;
    }

pcg_end:

    free(Ad);
    free(d);
    free(z);
    free(r);
    free(M);

    return iter;
}

int csr_sym()
{
    return 0; //  Both parts
    // return 1; // Lower part only
}


