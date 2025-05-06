#ifndef MODEL_H
#define MODEL_H

#include "models.h"

#include <stddef.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define SQUARE(X) ((X) * (X))

typedef enum Renumbering {
    RENUM_NO,
    RENUM_X,
    RENUM_Y
} Renumbering;

typedef enum ElementType {
    TRI = 2,
    QUAD = 3
} ElementType;

typedef struct SymBandMatrix {
    size_t n, k;  // dimension de la matrice et largeur de bande
    double *data; // 1D array [m*(k+1)] avec les Aij de la partie inférieure
    double **a;   // 1D array [m] de pointeurs vers chaque ligne -> a[i][j]
} SymBandMatrix;
void free_sym_band_matrix(SymBandMatrix *mat);

typedef struct CSRMatrix {
    int n, nnz;
    int *row_ptr;
    int *col_idx;
    double *data;
} CSRMatrix;
void free_csr(CSRMatrix *csr);

typedef struct FE_Model {
    const char *model_name;
    double E;
    double nu;
    double rho;
    double L_ref;
    Model2D m_type;
    ElementType e_type;
    Renumbering renum;
    size_t node_band;
    size_t n_elem;
    size_t n_local;
    size_t n_node;
    size_t n_bd_edge;
    size_t *e_tags;     // 1-based indexing
    size_t *elem_nodes; // 1-based indexing
    size_t *bd_edges;   // 1-based indexing (n1, n2, e, l)
    size_t *idx_map;    // 0-based indexing
    double *coords;
    SymBandMatrix *M;
    SymBandMatrix *K;
    SymBandMatrix *M_scalar;
    void (*mesh_model)(double, int);
    void (*set_bk_source)(double, const double[2], double[2]);
    void (*set_bd_disp)(int, char, const double[2], double[1]);
    void (*set_bd_force)(int, char, const double[2], double[2]);
} FE_Model;

typedef struct {
    size_t e1, e2;
    size_t n1, n2;
} Edge;

FE_Model *create_FE_Model(const char *name, ElementType etp, Renumbering renum);
void free_FE_Model(FE_Model *model);
void renumber_nodes(FE_Model *model);

#endif