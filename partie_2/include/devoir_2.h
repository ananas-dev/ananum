#ifndef DEVOIR_2_H
#define DEVOIR_2_H

void Matvec(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *v,
    double *Av
);

void solve(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *L,
    const double *b,
    double *x
);

int CG(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *b,
    double *x,
    double eps
);

void ILU(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    double *L
);

int PCG(
    int n,
    int nnz,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *b,
    double *x,
    double eps
);

int csr_sym();

#endif
