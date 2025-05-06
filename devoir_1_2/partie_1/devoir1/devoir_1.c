#include "devoir_1.h"
#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "debug.h"
#include <string.h>

double sign(double x) {
    return (x >= 0) - (x < 0);
}


void tridiagonalize_full(double *A, int n, int k, double *d, double *e) {
    // since they are the same size
    double *x = d;

    // Temporary matricies with upper bound sizes
    double *Q = malloc(n * n * sizeof(double));
    
    double *A_local = malloc(n * n * sizeof(double));

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

        if (fabs(vk_norm < 1e-12)) {
            continue;
        }

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

// a b
// b c

int step_qr_tridiag(double *d, double *e, int m, double eps){

    // Compute Wilkinson shift (see Trefthen pg. 222)
    double mu;
    double del = (d[m - 2] - d[m - 1]) / 2.0;
    mu = d[m - 1] + del - copysign(sqrt(del * del + e[m - 2] * e[m - 2]), del);

    mu +=1e-12; // to avoid mu = d[m-1] in the case of del = 0


    //on applique le shift
    for (int i = 0; i < m; i++) {
        d[i] -= mu;
    }


    //variable pour la rotation     GOOD
    double a, b, r;
    double d_k, d_k1, e_k1;
    double all_c[m];
    double all_s[m]; 
    e[0] = e[1];
    //rotation gauche 
    for (int k = 0; k < m-1; k++){
        a = d[k];
        b = e[k+1];
        r = sqrt(a*a + b*b);
        double c = a/r ;all_c[k] = c;
        double s = b/r; all_s[k] = s;

        d_k = d[k]; d_k1 = d[k+1]; e_k1 = e[k+1];

        //rotation
        d[k] = d_k * c + s*e_k1;        
        d[k+1] = -s * e[0] + c*d_k1;
        e[k+1] = c*e[0] + s * d_k1;                

        e[0] = c * e[k+2];       //e[0] = e_2 cos(theta)
    }

    //rotation droite  GOOD
    for (int k = 0; k < m-1; k++){
        d[k] = d[k] * all_c[k] +e[k+1]*all_s[k];
        e[k+1] = all_s[k]*d[k+1];
        d[k+1] = all_c[k]*d[k+1];
    }

    //on redécale
    for (int i = 0; i < m; i++) {
        d[i] += mu;
    }


    if (fabs(e[m-1]) <= eps * (fabs(d[m-2]) + fabs(d[m-1]))) {
        // Dernière valeur propre isolée
        return m-1;
    }
    return m; // aucune valeur propre a été isolée :(
}




//marche normalement
int qr_eigs_full(double *A, int n, int k, double eps, int max_iter, double *d){
    double *d_A = malloc(sizeof(double) * n);
    double *e_A = malloc(sizeof(double) * n);
    tridiagonalize_full(A, n, k, d_A, e_A);

    int m = n;
    int total_iter = 0;
    
    while (m > 1 && total_iter < max_iter) {
        int m_new = step_qr_tridiag(d_A, e_A, m, eps);
        
        if (m_new == m) {
            // Pas de convergence lors de cette étape
            total_iter++;
        } else {
            // Une valeur propre a été isolée
            m = m_new;
            total_iter++;
        }
    }
    for(int i = 0; i<n; i++){
        d[i] = d_A[i];
    }
    return total_iter;
}