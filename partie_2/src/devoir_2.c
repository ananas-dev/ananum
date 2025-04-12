#include "devoir_2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>



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

void Matvec(int n, int nnz,const int *rows_idx,const int *cols,const double *A,const double *v,double *Av)
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
    Matvec(n, nnz, rows_idx, cols, A, x, r); // r contient Ax_0 = 0 d'ailleurs
    for(int i=0; i<n; i++) {
        r[i] = b[i] - r[i]; // r = b - Ax_0 = b    (moyen on peut changer ça et juste mettre r = b pour l'initialisation)
    }
    copy_vector(n, r, p);  //r_0 = p_0

    int k = 0;
    double condition = 2*eps;
    int iter = 0;
    int max_iter = 10*n; //au pif mais comme ça ça pète pas une zine

    while(condition < eps && iter > max_iter){
        //calcule de Ap_k
        //on fait pour que Ap soit nul parce que matvec ne passe pas par tous Ap
        memset(Ap, 0, n*sizeof(double) * n); //sizeof(Ap)
        matvec(n, nnz, rows_idx, cols, A, p, Ap);

        double alpha = dot_product(n, r, r)/dot_product(n, p, matvec());
        
    }
    
    return 0;
}

void ILU(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    double *L) {}

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


