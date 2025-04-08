#include "devoir_2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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


