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
    if (m <= 1)
        return m;  // Rien à faire pour une matrice 1x1
    if (fabs(e[m-1]) <= eps * (fabs(d[m-2]) + fabs(d[m-1]))) {
        // Dernière valeur propre isolée
        return m-1;
    }
    

    //calcul de la valeur propre la plus proche de c 
    double d_n1, d_n, e_n, vp_1, vp_2, mu;

    d_n1 = d[m-2];
    e_n = e[m-2];
    d_n = d[m-1];
    
    //calcul des deux vp
    vp_1 = (d_n1 +d_n + sqrt((d_n1 +d_n)*(d_n1 +d_n) - 4*(d_n * d_n1-e_n*e_n)))/2.0;
    vp_2 = (d_n1 +d_n - sqrt((d_n1 +d_n)*(d_n1 +d_n) - 4*(d_n * d_n1-e_n*e_n)))/2.0;

    mu = fabs(d_n - vp_1) < fabs(d_n - vp_2) ? vp_1 : vp_2;


    //on applique le shift
    for (int i = 0; i < m; i++) {
        d[i] -= mu;
    }


    //variable pour la rotation
    double a, b, r;
    e[0] = e[1];
    //rotation gauche 
    for (int k = 0; k < m-1; k++){
        a = d[k];
        b = e[k+1];
        r = sqrt(a*a + b*b);

        //rotation
        d[k] = (d[k] * a + b*e[k+1])/r;        
        d[k+1] = (-b * e[0] + a*d[k+1])/r;
        e[k+1] = (a*e[0] + b * d[k+1])/r;                
        if(k<m-2){
            e[0] = a * e[k+2]/r;       //e[0] = e_2 cos(theta)
        }
    }

    //rotation droite
    for (int k = 0; k < m-1; k++){
        a = d[k];
        b = e[k+1];
        r = sqrt(a*a + b*b);
        d[k] = (d[k] * a +e[k+1]*b)/r;
        e[k+1] = (b*d[k+1] + a*e[k+1])/r;
        d[k+1] = (a*d[k+1] - b*e[k+1])/r;
    }

    if (fabs(e[m-1]) <= eps * (fabs(d[m-2]) + fabs(d[m-1]))) {
        // Dernière valeur propre isolée
        return m-1;
    }

    return m; // aucune valeur propre a été isolée :(
}
//test