#include "utils.h"
#include "model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ZERO_RADIUS 1e-14

SymBandMatrix *allocate_sym_band_matrix(const size_t n, const size_t k) {
    // To make things simple, we allocate k+1 entries for each row
    // (even though we need less than that for the k first and last rows).
    // We want to have a[i][i] == data[k + i*(k+1)]

    SymBandMatrix *mat = (SymBandMatrix *)malloc(sizeof(SymBandMatrix));
    mat->n = n;
    mat->k = k;
    mat->data = (double *)calloc(n * (k + 1), sizeof(double));
    mat->a = (double **)malloc(n * sizeof(double *));
    // This is the tricky part :-)
    // We want a[i][i] == data[(k+1)*i + k]
    // which means a[i] + i == data + (k+1)*i + k
    // and therefore a[i] == data + k + k*i
    for (size_t i = 0; i < n; i++) {
        mat->a[i] = mat->data + k + k * i;
    }

    /*
    x : matrix off diagonal element
    = : matrix diagonal element
    + : memory allocated but not used
    Example for k = 3:
    + + + =
      + + x =
        + x x =
          x x x =
            x x x =
              x x x =
                x x x =
                  x x x =
                    x x x =
    */
    return mat;
}

void print_sym_band(SymBandMatrix *M) {
    if (M->n > 31)
        return;
    printf("\nSymmetric band matrix\n");
    size_t j_min;
    double eps = 1.e-12;
    size_t k = M->k;
    for (size_t i = 0; i < M->n; i++) {
        j_min = (i < k) ? 0 : i - k;
        for (size_t j = 0; j < j_min; j++) {
            printf("%10s ", "");
        }
        for (size_t j = j_min; j <= i; j++) {
            if (fabs(M->a[i][j]) < eps)
                printf("%10s ", ".");
            else
                printf("%10.2le ", M->a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void write_sym_band(SymBandMatrix *M, double *rhs, const char *filename) {
    int save_rhs = (rhs != NULL);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    fprintf(f, "# SYM_BAND %zu %zu %d\n", M->n, M->k, save_rhs);
    size_t k = M->k;
    for (size_t i = 0; i < M->n; i++) {
        for (size_t j = 0; j < k + 1; j++) {
            fprintf(f, "%22.15le ", M->data[i * (k + 1) + j]);
        }
        if (save_rhs)
            fprintf(f, "%22.15le", rhs[i]);
        fprintf(f, "\n");
    }
    fclose(f);
}

void free_sym_band_matrix(SymBandMatrix *mat) {
    free(mat->a);
    free(mat->data);
    free(mat);
}

void print_vector_row(size_t n, double *vec) {
    if (n > 31)
        return;
    for (size_t i = 0; i < n; i++) {
        printf("%10.2le ", vec[i]);
    }
    printf("\n");
}

void print_vector(size_t n, double *vec) {
    for (size_t i = 0; i < n; i++) {
        printf("%4zu : %22.15le\n", i, vec[i]);
    }
    printf("\n");
}

CSRMatrix *band_to_csr(const SymBandMatrix *band) {
    CSRMatrix *csr = malloc(sizeof(CSRMatrix));
    double thresh = 1e-15;
    double val;
    size_t j_bound, count;
    size_t k = band->k;
    csr->n = band->n;
    csr->nnz = 0;

    for (size_t i = 0; i < band->n; i++) {
        j_bound = (i < k) ? 0 : i - k;
        count = 0;
        for (size_t j = j_bound; j < i; j++) {
            if (thresh < fabs(band->a[i][j]))
                count++;
        }
        csr->nnz += 2 * count + 1;
    }

    size_t row_idx = 0;
    size_t col_idx = 0;
    csr->row_ptr = malloc((band->n + 1) * sizeof(size_t));
    csr->col_idx = malloc(csr->nnz * sizeof(size_t));
    csr->data = malloc(csr->nnz * sizeof(double));
    csr->row_ptr[0] = 0;

    for (size_t i = 0; i < band->n; i++) {
        csr->row_ptr[row_idx + 1] = csr->row_ptr[row_idx];
        j_bound = (i < k) ? 0 : i - k;
        for (size_t j = j_bound; j <= i; j++) {
            val = band->a[i][j];
            if (thresh < fabs(val)) {
                csr->row_ptr[row_idx + 1]++;
                csr->col_idx[col_idx] = j;
                csr->data[col_idx] = val;
                col_idx++;
            }
        }
        j_bound = (i + k + 1 < band->n) ? i + k + 1 : band->n;
        for (size_t j = i+1; j < j_bound; j++) {
            val = band->a[j][i];
            if (thresh < fabs(val)) {
                csr->row_ptr[row_idx + 1]++;
                csr->col_idx[col_idx] = j;
                csr->data[col_idx] = val;
                col_idx++;
            }
        }
        row_idx++;
    }

    return csr;
}

CSRMatrix *band_to_sym_csr(const SymBandMatrix *band) {
    CSRMatrix *csr = malloc(sizeof(CSRMatrix));
    double thresh = 1e-15;
    double val;
    size_t j_min;
    size_t k = band->k;
    csr->n = band->n;
    csr->nnz = 0;

    for (size_t i = 0; i < band->n; i++) {
        j_min = (i < k) ? 0 : i - k;
        for (size_t j = j_min; j <= i; j++)
            if (thresh < fabs(band->a[i][j]))
                csr->nnz++;
    }

    size_t row_idx = 0;
    size_t col_idx = 0;
    csr->row_ptr = malloc((band->n + 1) * sizeof(size_t));
    csr->col_idx = malloc(csr->nnz * sizeof(size_t));
    csr->data = malloc(csr->nnz * sizeof(double));
    csr->row_ptr[0] = 0;

    for (size_t i = 0; i < band->n; i++) {
        csr->row_ptr[row_idx + 1] = csr->row_ptr[row_idx];
        j_min = (i < k) ? 0 : i - k;
        for (size_t j = j_min; j <= i; j++) {
            val = band->a[i][j];
            if (thresh < fabs(val)) {
                csr->row_ptr[row_idx + 1]++;
                csr->col_idx[col_idx] = j;
                csr->data[col_idx] = val;
                col_idx++;
            }
        }
        row_idx++;
    }

    return csr;
}

void free_csr(CSRMatrix *csr) {
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->data);
    free(csr);
}

/**
 * @brief LDL' in-place d'une matrice symétrique bande
 */
void sym_band_LDL(double *A, size_t n, size_t b) {
    double akk, coef;
    size_t pivot; // Index of the pivot diagonal element
    size_t idx_i; // Index of the element ... rows below the pivot
    size_t idx_j; // Index of the element ... columns to the right of the pivot,
                  // but actually below due to symmetry
    size_t i_max; // Number of remaining rows below the pivot
    size_t lda;   // Stride of the band storage

    lda = b + 1;
    for (size_t k = 0; k < n; k++) {
        pivot = k * lda + b;
        akk = A[pivot];
        i_max = (k + b) < n ? b : n - k - 1;
        for (size_t i = 1; i <= i_max; i++) {
            idx_j = pivot + (lda - 1);
            idx_i = pivot + (lda - 1) * i;
            coef = A[idx_i] / akk;
            for (size_t j = 1; j <= i; j++) {
                A[idx_i + j] -= coef * A[idx_j];
                idx_j += lda - 1;
            }
        }
        for (size_t i = 1; i <= i_max; i++) {
            idx_i = pivot + (lda - 1) * i;
            A[idx_i] /= akk;
        }
    }
}

void solve_sym_band(double *L, size_t n, size_t b, double *x) {
    // Solve L D L' x = x0
    size_t bound, lda = b + 1;
    
    // Solve L * z = x0
    // cblas_dtbsv(BRM, BLW, BNT, BUN, n, b, L, b + 1, x, 1);
    for (size_t i = 0; i < n; i++) { 
        bound = (i < b) ? b - i : 0;
        for (size_t j = bound; j < b; j++) {
            x[i] -= L[i * lda + j] * x[j + i - b];
        }
    }
    
    // Solve D * y = z
    for (size_t i = 0; i < n; i++) {
        x[i] /= L[i * (b + 1) + b];
    }
    
    // Solve L' * x = y
    // cblas_dtbsv(BRM, BLW, BTR, BUN, n, b, L, b + 1, x, 1);
    for (size_t ii = 0; ii < n; ii++) {
        size_t i = n - 1 - ii;
        bound = (i < b) ? b - i : 0;
        for (size_t j = bound; j < b; j++) {
            x[j + i - b] -= L[i * lda + j] * x[i];
        }
    }
}

void compute_shape_functions(
    ElementType e_type,
    size_t *nq,
    double w[4],
    double xi[4][2],
    double phi[4][4],
    double dph[4][4][2],
    const int boundary
) {
    double a = sqrt(1. / 3.), b;
    double x, y;
    // clang-format off
    // Set quadrature points
    if (boundary) {
        nq[0] = 2;
        b = (e_type == TRI) ? 0.5 : 1.0;
        w[0] = b; w[1] = b;
        x = (e_type == TRI) ? 0.5 : 0.;
        y = (e_type == TRI) ? 0. : -1.;
        b = (e_type == TRI) ? 0.5 : 1.;
        xi[0][0] = x - b * a; xi[0][1] = y;
        xi[1][0] = x + b * a; xi[1][1] = y;
    } else if (e_type == TRI) {
        nq[0] = 3;
        a = 1. / 6.;
        b = 2. / 3.;
        w[0] = a; w[1] = a; w[2] = a; w[3] = 0.;
        xi[0][0] = a ; xi[0][1] = a;
        xi[1][0] = b ; xi[1][1] = a;
        xi[2][0] = a ; xi[2][1] = b;
        xi[3][0] = 0.; xi[3][1] = 0.;
    } else if (e_type == QUAD) {
        // nq[0] = 1;
        // w[0] = 4.;
        // xi[0][0] = 0.; xi[0][1] = 0.;
        nq[0] = 4;
        w[0] = 1.; w[1] = 1.; w[2] = 1.; w[3] = 1.;
        xi[0][0] = -a; xi[0][1] = -a;
        xi[1][0] = +a; xi[1][1] = -a;
        xi[2][0] = +a; xi[2][1] = +a;
        xi[3][0] = -a; xi[3][1] = +a;
    }
    // Set shape functions and derivatives
    if (e_type == TRI) {
        for (int i = 0; i < *nq; i++) {
            phi[i][0] = 1. - xi[i][0] - xi[i][1];
            phi[i][1] = xi[i][0];
            phi[i][2] = xi[i][1];
            phi[i][3] = 0.;
            dph[i][0][0] = -1.; dph[i][0][1] = -1.;
            dph[i][1][0] = +1.; dph[i][1][1] = +0.;
            dph[i][2][0] = +0.; dph[i][2][1] = +1.;
            dph[i][3][0] = +0.; dph[i][3][1] = +0.;
        }
    } else if (e_type == QUAD) {
        for (int i = 0; i < *nq; i++) {
            x = xi[i][0];
            y = xi[i][1];
            phi[i][0] = 0.25 * (1. - x) * (1. - y);
            phi[i][1] = 0.25 * (1. + x) * (1. - y);
            phi[i][2] = 0.25 * (1. + x) * (1. + y);
            phi[i][3] = 0.25 * (1. - x) * (1. + y);
            dph[i][0][0] = -0.25 * (1. - y); dph[i][0][1] = -0.25 * (1. - x);
            dph[i][1][0] = +0.25 * (1. - y); dph[i][1][1] = -0.25 * (1. + x);
            dph[i][2][0] = +0.25 * (1. + y); dph[i][2][1] = +0.25 * (1. + x);
            dph[i][3][0] = -0.25 * (1. + y); dph[i][3][1] = +0.25 * (1. - x);
        }
    } else {
        printf("Unknown element type: %d\n", e_type);
        exit(EXIT_FAILURE);
    }
    // clang-format on
}

void set_hooke_matrix(double E, double nu, int m_type, double h[4][4]) {
    // Non typical Hooke matrix ordering (allows flexibility for axisymmetric)
    // relates (s_xx, s_yy, s_xy) to (e_xx, e_yy, e_xy) for plane stress/strain
    // relates (s_rr, s_zz, s_rz, s_tt) to (e_rr, e_zz, e_rz, e_tt) for axisym.
    double a;
    memset(&h[0][0], 0, 16 * sizeof(double));
    if (m_type == PLANE_STRESS) {
        a = E / (1. - nu * nu);
        h[0][0] = h[1][1] = a;          // volumetric diagonal
        h[0][1] = h[1][0] = a * nu;     // volumetric off-diagonal
        h[2][2] = E / (2. * (1. + nu)); // shear modulus
    } else if (m_type == PLANE_STRAIN) {
        a = E / ((1. + nu) * (1. - 2. * nu));
        h[0][0] = h[1][1] = a * (1. - nu);
        h[0][1] = h[1][0] = a * nu;
        h[2][2] = E / (2. * (1. + nu));
    } else if (m_type == AXISYMMETRIC) {
        a = E / ((1. + nu) * (1. - 2. * nu));
        h[0][0] = h[1][1] = h[3][3] = a * (1. - nu);
        h[0][1] = h[1][0] = a * nu;
        h[2][2] = E / (2. * (1. + nu));
        h[0][3] = h[1][3] = h[3][0] = h[3][1] = nu * a;
    } else {
        printf("Unknown model type: %d\n", m_type);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Set the strain basis matrix B
 * @param n_loc Number of local nodes
 * @param phi Shape functions (nloc)
 * @param dph Shape functions derivatives (nloc, 2)
 * @param B Strain basis matrix (nloc, 4)
 * @note Unset entries should be set to zero by the caller
 */
void set_strain_basis(
    const int n_loc,
    const double phi[4],
    const double dphi[4][2],
    const double inv_r,
    double B[8][4]
) {
    for (int i = 0; i < n_loc; i++) {
        B[2 * i + 0][0] = B[2 * i + 1][2] = dphi[i][0];
        B[2 * i + 0][2] = B[2 * i + 1][1] = dphi[i][1];
        B[2 * i + 0][3] = phi[i] * inv_r;
    }
}

void set_local_stiffness_matrix(
    const int n_loc,
    const double phi[4],
    const double dphi[4][2],
    const double H[4][4],
    const double r,
    const double inv_r,
    double K[8][8]
) {
    const int n_dof = 2 * n_loc;
    double BT[8][4] = {0};
    set_strain_basis(n_loc, phi, dphi, inv_r, BT);

    // Compute [eps(phi_i) H eps(phi_j)]
    double BT_H[8][4];
    for (int i = 0; i < n_dof; i++) {
        for (int j = 0; j < 4; j++) {
            BT_H[i][j] = 0.;
            for (int k = 0; k < 4; k++) {
                BT_H[i][j] += r * BT[i][k] * H[k][j];
            }
        }
    }
    for (int i = 0; i < n_dof; i++) {
        for (int j = 0; j < n_dof; j++) {
            K[i][j] = 0.;
            for (int k = 0; k < 4; k++) {
                K[i][j] += BT_H[i][k] * BT[j][k];
            }
        }
    }
}

void set_local_mass_matrix(
    const int n_loc, const double phi[4], const double r, double M[8][8]
) {
    for (int i = 0; i < n_loc; i++) {
        for (int j = 0; j < n_loc; j++) {
            M[2 * i + 0][2 * j + 0] = r * phi[i] * phi[j]; // u - u
            M[2 * i + 1][2 * j + 1] = r * phi[i] * phi[j]; // v - v
            M[2 * i + 0][2 * j + 1] = 0.;                  // u - v
            M[2 * i + 1][2 * j + 0] = 0.;                  // v - u
        }
    }
}

void set_geometry(
    const int n_loc,
    const double x_node[4][2],
    const double phi[4],
    const double dph[4][2],
    double x_ptr[2],
    double det[1],
    double dphi[4][2]
) {
    double dxidx[2][2] = {{0., 0.}, {0., 0.}};
    x_ptr[0] = x_ptr[1] = 0.;
    for (int j = 0; j < n_loc; j++) {
        x_ptr[0] += x_node[j][0] * phi[j];
        x_ptr[1] += x_node[j][1] * phi[j];
        dxidx[0][0] += x_node[j][0] * dph[j][0];
        dxidx[0][1] += x_node[j][0] * dph[j][1];
        dxidx[1][0] += x_node[j][1] * dph[j][0];
        dxidx[1][1] += x_node[j][1] * dph[j][1];
    }
    det[0] = dxidx[0][0] * dxidx[1][1] - dxidx[0][1] * dxidx[1][0];
    for (int j = 0; j < n_loc; j++) {
        dphi[j][0] = +dxidx[1][1] * dph[j][0] - dxidx[1][0] * dph[j][1];
        dphi[j][1] = -dxidx[0][1] * dph[j][0] + dxidx[0][0] * dph[j][1];
    }
}

/**
 * @brief Compute the mass and stiffness matrices of a FE mesh
 */
void assemble_system(FE_Model *model) {
    int n_elem = model->n_elem;
    int nl = model->n_local;
    const size_t *elem_nodes = model->elem_nodes;
    int n_node = model->n_node;
    const double *coords = model->coords;
    const size_t *idx_map = model->idx_map;
    double nu = model->nu;

    int max_diff = model->node_band;
    model->M = allocate_sym_band_matrix(2 * n_node, 2 * max_diff + 1);
    model->K = allocate_sym_band_matrix(2 * n_node, 2 * max_diff + 1);
    SymBandMatrix *M = model->M;
    SymBandMatrix *K = model->K;

    size_t nq;
    size_t num[4] = {0};
    double w[4], xi[4][2], phi[4][4], dph[4][4][2];
    double x_node[4][2], dphi[4][2], x_loc[2], det, scaleM, scaleK;
    double hooke[4][4], K_loc[8][8], M_loc[8][8];
    double r, ir;

    set_hooke_matrix(1., nu, model->m_type, hooke);
    compute_shape_functions(model->e_type, &nq, w, xi, phi, dph, 0);

    for (size_t i = 0; i < n_elem; i++) {
        const size_t *local_nodes = &elem_nodes[nl * i];
        SET_ELEM_INFO(nl, local_nodes, idx_map, coords, x_node, num);
        for (size_t q = 0; q < nq; q++) {
            set_geometry(nl, x_node, phi[q], dph[q], x_loc, &det, dphi);
            r = (model->m_type == AXISYMMETRIC) ? x_loc[0] : 1.;
            ir = (model->m_type == AXISYMMETRIC) ? det / x_loc[0] : 0.;
            set_local_stiffness_matrix(nl, phi[q], dphi, hooke, r, ir, K_loc);
            set_local_mass_matrix(nl, phi[q], r, M_loc);
            scaleK = w[q] / det;
            scaleM = w[q] * det;
            // Global matrices are numbered [u1, v1, u2, v2, u3, v3, u4, v4]
            for (size_t j = 0; j < 2 * nl; j++) {
                size_t row = 2 * num[j / 2] + (j % 2);
                for (size_t k = 0; k < 2 * nl; k++) {
                    size_t col = 2 * num[k / 2] + (k % 2);
                    if (col <= row) { // only fill lower part
                        K->a[row][col] += K_loc[j][k] * scaleK;
                        M->a[row][col] += M_loc[j][k] * scaleM;
                    }
                }
            }
        }
    }
    return;
}

/**
 * @brief Add the bulk source term to the rhs
 * @param model Finite element model
 * @param rhs Right-hand side of the system
 * @param bulk_source Function pointer to the source term evaluation
 */
void add_bulk_source(FE_Model *model, double *rhs) {
    size_t nq;
    size_t num[4] = {0};
    double w[4], xi[4][2], phi[4][4], dph[4][4][2];
    double x_node[4][2], dphi[4][2], x_loc[2], det, r;
    double f[2];

    int n_elem = model->n_elem;
    int nl = model->n_local;
    const size_t *elem_nodes = model->elem_nodes;
    const double *coords = model->coords;
    const size_t *idx_map = model->idx_map;

    compute_shape_functions(model->e_type, &nq, w, xi, phi, dph, 0);

    for (size_t i = 0; i < n_elem; i++) {
        const size_t *local_nodes = &elem_nodes[nl * i];
        SET_ELEM_INFO(nl, local_nodes, idx_map, coords, x_node, num);
        for (size_t q = 0; q < nq; q++) {
            set_geometry(nl, x_node, phi[q], dph[q], x_loc, &det, dphi);
            r = (model->m_type == AXISYMMETRIC) ? x_loc[0] : 1.;
            model->set_bk_source(model->rho, x_loc, f);
            f[0] *= model->L_ref / model->E;
            f[1] *= model->L_ref / model->E;
            for (size_t j = 0; j < nl; j++) {
                rhs[2 * num[j] + 0] += w[q] * det * phi[q][j] * f[0] * r;
                rhs[2 * num[j] + 1] += w[q] * det * phi[q][j] * f[1] * r;
            }
        }
    }
}

double set_direction(
    const char kind, const double xy1[2], const double xy2[2], double *dir
) {
    double dx = xy2[0] - xy1[0];
    double dy = xy2[1] - xy1[1];
    double det = hypot(dx, dy);
    if (kind == 'x') {
        dir[0] = 1.0;
        dir[1] = 0.0;
    } else if (kind == 'y') {
        dir[0] = 0.0;
        dir[1] = 1.0;
    } else if (kind == 'n') {
        dir[0] = +dy / det;
        dir[1] = -dx / det;
    } else if (kind == 't') {
        dir[0] = dx / det;
        dir[1] = dy / det;
    } else { // cannot happen, but avoids compiler warning
        dir[0] = 0.0;
        dir[1] = 0.0;
    }
    return det;
}

/**
 * @brief Add the Robin boundary condition (Neumann is a special case)
 * @param model Finite element model
 * @param entity Entity number
 * @param size Size of edges (2 * n_edges)
 * @param edges Nodes on edges of the boundary (2 * n_edges)
 * @param kind x, y, n, or t for x, y, normal, or tangent
 * @param rhs System rhs to be modified
 * @note The Robin bc : n*sigma*kind = -k*(u - u_ref) + f = -alpha u + beta
 */
void apply_force(
    const FE_Model *model,
    const size_t entity,
    const size_t size,
    const size_t *edges,
    const char kind,
    double *rhs
) {
    size_t j1, j2, num1, num2, num_min, num_max, nq;
    double r, ref_l, det, val, dir[2], xy[2], f[2], tuu, tuv, tvv;
    double w[4], xi[4][2], phi[4][4], dph[4][4][2];
    double L = model->L_ref;
    double *x = model->coords;
    SymBandMatrix *K = model->K;

    compute_shape_functions(model->e_type, &nq, w, xi, phi, dph, 1);
    ref_l = 0.;
    for (int q = 0; q < nq; q++)
        ref_l += w[q];

    for (int e = 0; 2 * e < size; e += 1) {
        j1 = edges[2 * e + 0] - 1;
        j2 = edges[2 * e + 1] - 1;
        det = set_direction(kind, &x[2 * j1 + 0], &x[2 * j2 + 0], dir) / ref_l;
        num1 = 2 * model->idx_map[j1];
        num2 = 2 * model->idx_map[j2];
        num_min = MIN(num1, num2);
        num_max = MAX(num1, num2);
        for (int q = 0; q < nq; q++) {
            xy[0] = (x[2 * j1 + 0] * phi[q][0] + x[2 * j2 + 0] * phi[q][1]) * L;
            xy[1] = (x[2 * j1 + 1] * phi[q][0] + x[2 * j2 + 1] * phi[q][1]) * L;
            model->set_bd_force(entity, kind, xy, f);
            f[0] *= model->L_ref / model->E; // alpha (dimensionless)
            f[1] *= 1. / model->E;           // beta (dimensionless)
            r = (model->m_type == AXISYMMETRIC) ? xy[0] / L : 1.;
            // Boundary stiffness matrix
            tuu = f[0] * dir[0] * dir[0];
            tuv = f[0] * dir[0] * dir[1];
            tvv = f[0] * dir[1] * dir[1];
            val = w[q] * phi[q][0] * phi[q][0] * r * det; // j1 j1
            K->a[num1 + 0][num1 + 0] += val * tuu;
            K->a[num1 + 1][num1 + 0] += val * tuv;
            K->a[num1 + 1][num1 + 1] += val * tvv;
            val = w[q] * phi[q][1] * phi[q][1] * r * det; // j2 j2
            K->a[num2 + 0][num2 + 0] += val * tuu;
            K->a[num2 + 1][num2 + 0] += val * tuv;
            K->a[num2 + 1][num2 + 1] += val * tvv;
            val = w[q] * phi[q][1] * phi[q][0] * r * det; // j2 j1
            K->a[num_max + 0][num_min + 0] += val * tuu;
            K->a[num_max + 0][num_min + 1] += val * tuv;
            K->a[num_max + 1][num_min + 0] += val * tuv;
            K->a[num_max + 1][num_min + 1] += val * tvv;
            // Boundary force vector
            val = w[q] * r * det * f[1]; // rhs
            rhs[num1 + 0] += phi[q][0] * val * dir[0];
            rhs[num1 + 1] += phi[q][0] * val * dir[1];
            rhs[num2 + 0] += phi[q][1] * val * dir[0];
            rhs[num2 + 1] += phi[q][1] * val * dir[1];
        }
    }
}

void compute_node_normals(
    const size_t n_edge, const double *x, double *nx, double *ny, int periodic
) {
    double dx, dy;
    for (int e = 0; e < n_edge; e += 1) { // edge normals
        dx = x[2 * (e + 1) + 0] - x[2 * (e + 0) + 0];
        dy = x[2 * (e + 1) + 1] - x[2 * (e + 0) + 1];
        nx[e] = +dy;
        ny[e] = -dx;
    }
    nx[n_edge] = nx[n_edge - 1];
    ny[n_edge] = ny[n_edge - 1];
    for (int e = n_edge - 1; 0 < e; e--) {
        nx[e] = 0.5 * (nx[e - 1] + nx[e]);
        ny[e] = 0.5 * (ny[e - 1] + ny[e]);
    }
    if (periodic) {
        nx[0] = 0.5 * (nx[0] + nx[n_edge]);
        ny[0] = 0.5 * (ny[0] + ny[n_edge]);
    }
    for (int e = 0; e <= n_edge; e += 1) {
        double norm = hypot(nx[e], ny[e]);
        nx[e] /= norm;
        ny[e] /= norm;
    }
}

void project_system(
    size_t i, SymBandMatrix *K, double *rhs, double nx, double ny
) {
    size_t jxy, i_bound;
    size_t k_node = K->k / 2; // k = 2 * k_node + 1
    size_t ix = 2 * i + 0;
    size_t iy = 2 * i + 1;
    double aix, aiy, axy;
    // Rotate rows
    i_bound = (i < k_node) ? 0 : 2 * (i - k_node);
    for (jxy = i_bound; jxy < ix; jxy++) {
        aix = K->a[ix][jxy];
        aiy = K->a[iy][jxy];
        K->a[ix][jxy] = nx * aix + ny * aiy;
        K->a[iy][jxy] = ny * aix - nx * aiy;
    }
    i_bound = MIN(2 * (i + k_node + 1), K->n);
    for (jxy = iy + 1; jxy < i_bound; jxy++) {
        aix = K->a[jxy][ix];
        aiy = K->a[jxy][iy];
        K->a[jxy][ix] = nx * aix + ny * aiy;
        K->a[jxy][iy] = ny * aix - nx * aiy;
    }
    // Central 2x2 block
    aix = K->a[ix][ix];
    aiy = K->a[iy][ix];
    axy = K->a[iy][iy];
    K->a[ix][ix] = nx * nx * aix + 2 * nx * ny * aiy + ny * ny * axy;
    K->a[iy][iy] = ny * ny * aix - 2 * nx * ny * aiy + nx * nx * axy;
    K->a[iy][ix] = nx * ny * aix + (ny * ny - nx * nx) * aiy - nx * ny * axy;
    // Rhs
    aix = rhs[ix];
    aiy = rhs[iy];
    rhs[ix] = nx * aix + ny * aiy;
    rhs[iy] = ny * aix - nx * aiy;
}

/**
 * @brief Apply Dirichlet boundary conditions
 * @param model Finite element model
 * @param entity Entity number
 * @param size Size of edges (2 * n_edges)
 * @param edges Nodes on edges of the boundary (2 * n_edges)
 * @param kind x, y, n, or t for x, y, normal, or tangent
 * @param rhs System rhs to be modified
 */
void apply_dirichlet(
    const FE_Model *model,
    const size_t entity,
    const size_t size,
    size_t *edges,
    const char kind,
    double *rhs
) {
    double val;
    size_t node, num, numxy, i, i_bound;
    const size_t *i_map = model->idx_map;
    SymBandMatrix *K = model->K;
    size_t n_bd_node = size / 2 + 1;
    int periodic = edges[0] == edges[size - 1];

    double *nx = malloc(n_bd_node * sizeof(double));
    double *ny = malloc(n_bd_node * sizeof(double));
    double *xy = malloc(2 * n_bd_node * sizeof(double));

    for (int j = 0; j < n_bd_node; j++) {
        edges[j] = edges[MIN(2 * j, size - 1)];
        node = edges[j] - 1;
        xy[2 * j + 0] = model->coords[2 * node + 0];
        xy[2 * j + 1] = model->coords[2 * node + 1];
    }
    compute_node_normals(n_bd_node - 1, xy, nx, ny, periodic);
    n_bd_node = periodic ? n_bd_node - 1 : n_bd_node;
    for (int j = 0; j < 2 * n_bd_node; j++) 
        xy[j] *= model->L_ref; // scale for set_bd_disp

    for (int j = 0; j < n_bd_node; j++) {
        num = i_map[edges[j] - 1];
        model->set_bd_disp(entity, kind, &xy[2 * j], &val);
        val /= model->L_ref;
        if (kind == 'n' || kind == 't')
            project_system(num, K, rhs, nx[j], ny[j]);
        numxy = (kind == 'x' || kind == 'n') ? 2 * num + 0 : 2 * num + 1;
        // Zero out row associated to dof "numxy"
        i_bound = (numxy < K->k) ? 0 : numxy - K->k;
        for (i = i_bound; i < numxy; i++) {
            rhs[i] -= K->a[numxy][i] * val;
            K->a[numxy][i] = 0.;
        }
        // Zero out col associated to dof "numxy"
        i_bound = MIN(numxy + K->k + 1, K->n);
        for (i = numxy + 1; i < i_bound; i++) {
            rhs[i] -= K->a[i][numxy] * val;
            K->a[i][numxy] = 0.;
        }
        K->a[numxy][numxy] = 1.;
        rhs[numxy] = val;
        if (kind == 'n' || kind == 't')
            project_system(num, K, rhs, nx[j], ny[j]);
    }
    free(nx);
    free(ny);
    free(xy);
}

void compute_local_stress(
    const int n_loc,
    const double phi[4],
    const double dphi[4][2],
    const double u[8],
    const double h[4][4],
    const double ir,
    double stress[4]
) {
    double BT[8][4] = {0};
    double strain[4] = {0};
    set_strain_basis(n_loc, phi, dphi, ir, BT);
    for (int j = 0; j < 2 * n_loc; j++) {
        strain[0] += BT[j][0] * u[j];
        strain[1] += BT[j][1] * u[j];
        strain[2] += BT[j][2] * u[j];
        strain[3] += BT[j][3] * u[j];
    }
    for (int i = 0; i < 4; i++) {
        stress[i] = 0.;
        for (int j = 0; j < 4; j++) {
            stress[i] += h[i][j] * strain[j];
        }
    }
}

void cartesian_to_polar(size_t n_node, double *sigma, double *x) {
    double s_xx, s_yy, s_xy, r, c, s, c2, s2;
    for (size_t i = 0; i < n_node; i++) {
        s_xx = sigma[9 * i + 0];
        s_yy = sigma[9 * i + 4];
        s_xy = sigma[9 * i + 1];
        r = hypot(x[2 * i + 0], x[2 * i + 1]);
        c = x[2 * i + 0] / r;
        s = x[2 * i + 1] / r;
        c2 = c * c;
        s2 = s * s;
        sigma[9 * i + 0] = s_xx * c2 + s_yy * s2 + 2. * s_xy * c * s;
        sigma[9 * i + 4] = s_xx * s2 + s_yy * c2 - 2. * s_xy * c * s;
        sigma[9 * i + 1] = (s_yy - s_xx) * c * s + s_xy * (c2 - s2);
        sigma[9 * i + 3] = sigma[9 * i + 1];
    }
}

/**
 * @brief Compute the mass matrix for the stress least squares problem
 * @param model Finite element model
 * @param M Mass matrix with band storage (n_node x (k + 1))
 */
void assemble_mass_lsq(FE_Model *model, SymBandMatrix *M) {
    int n_elem = model->n_elem;
    int nl = model->n_local;
    const size_t *elem_nodes = model->elem_nodes;
    const double *coords = model->coords;
    const size_t *idx_map = model->idx_map;

    size_t nq;
    size_t num[4] = {0};
    double w[4], xi[4][2], phi[4][4], dph[4][4][2], r;
    double x_node[4][2], x_loc[2], dphi[4][2], det, scaleM;

    compute_shape_functions(model->e_type, &nq, w, xi, phi, dph, 0);

    for (size_t i = 0; i < n_elem; i++) {
        const size_t *local_nodes = &elem_nodes[nl * i];
        SET_ELEM_INFO(nl, local_nodes, idx_map, coords, x_node, num);
        for (size_t q = 0; q < nq; q++) {
            set_geometry(nl, x_node, phi[q], dph[q], x_loc, &det, dphi);
            r = (model->m_type == AXISYMMETRIC) ? x_loc[0] : 1.;
            scaleM = w[q] * det * r;
            for (size_t j = 0; j < nl; j++) {
                size_t row = num[j];
                for (size_t k = 0; k < nl; k++) {
                    size_t col = num[k];
                    if (col <= row) { // only fill lower part
                        M->a[row][col] += phi[q][j] * phi[q][k] * scaleM;
                    }
                }
            }
        }
    }
    return;
}

void set_lsq_rhs(FE_Model *model, const double *u, double *rhs) {
    int n_elem = model->n_elem;
    int nl = model->n_local;
    const size_t *elem_nodes = model->elem_nodes;
    int n_node = model->n_node;
    const double *coords = model->coords;
    const size_t *idx_map = model->idx_map;

    size_t nq;
    size_t num[4] = {0};
    double w[4], xi[4][2], phi[4][4], dph[4][4][2], r, ir, h[4][4], sig[4];
    double x_node[4][2], u_node[8], x_loc[2], dphi[4][2], det, scale;

    compute_shape_functions(model->e_type, &nq, w, xi, phi, dph, 0);
    set_hooke_matrix(1., model->nu, model->m_type, h);

    for (size_t i = 0; i < n_elem; i++) {
        const size_t *e_nodes = &elem_nodes[nl * i];
        SET_ELEM_INFO_U(nl, e_nodes, idx_map, coords, u, x_node, u_node, num);
        for (size_t q = 0; q < nq; q++) {
            set_geometry(nl, x_node, phi[q], dph[q], x_loc, &det, dphi);
            ir = (model->m_type == AXISYMMETRIC) ? det / x_loc[0] : 0.;
            compute_local_stress(nl, phi[q], dphi, u_node, h, ir, sig);
            r = (model->m_type == AXISYMMETRIC) ? x_loc[0] : 1.;
            scale = w[q] * r;
            for (size_t j = 0; j < nl; j++) {
                size_t row = num[j];
                rhs[0 * n_node + row] += sig[0] * phi[q][j] * scale; // s_11
                rhs[1 * n_node + row] += sig[1] * phi[q][j] * scale; // s_22
                rhs[2 * n_node + row] += sig[2] * phi[q][j] * scale; // s_12
                rhs[3 * n_node + row] += sig[3] * phi[q][j] * scale; // s_33
            }
        }
    }
    return;
}


/**
 * @brief Compute the nodal stress for the stress least squares problem
 * @param model Finite element model
 * @param u Solution displacement field (2 * n_node)
 * @param stresses Contains the nodal stresses on exit (n_node x 9)
 */
void compute_nodal_stress_lsq(FE_Model *mdl, const double *u, double *stress) {
    SymBandMatrix *M = mdl->M_scalar;
    size_t n_node = mdl->n_node;
    const size_t *idx_map = mdl->idx_map;
    const double nu = mdl->nu;

    if (!M) {
        M = allocate_sym_band_matrix(mdl->n_node, mdl->node_band);
        assemble_mass_lsq(mdl, M);
        sym_band_LDL(M->data, M->n, M->k);
    } else {
        M = mdl->M_scalar;
    }

    double *rhs = calloc(4 * n_node, sizeof(double));

    // Compute the element stresses
    set_lsq_rhs(mdl, u, rhs);

    solve_sym_band(M->data, M->n, M->k, rhs + 0 * n_node);
    for (size_t i = 0; i < n_node; i++)
        stress[i * 9 + 0] = rhs[0 * n_node + idx_map[i]];
    solve_sym_band(M->data, M->n, M->k, rhs + 1 * n_node);
    for (size_t i = 0; i < n_node; i++)
        stress[i * 9 + 4] = rhs[1 * n_node + idx_map[i]];
    solve_sym_band(M->data, M->n, M->k, rhs + 2 * n_node);
    for (size_t i = 0; i < n_node; i++)
        stress[i * 9 + 1] = stress[i * 9 + 3] = rhs[2 * n_node + idx_map[i]];

    if (mdl->m_type == AXISYMMETRIC) {
        solve_sym_band(M->data, M->n, M->k, rhs + 3 * n_node);
        for (size_t i = 0; i < n_node; i++)
            stress[i * 9 + 8] = rhs[3 * n_node + idx_map[i]];
    } else if (mdl->m_type == PLANE_STRAIN) {
        for (size_t i = 0; i < n_node; i++)
            stress[i * 9 + 8] = nu * (stress[i * 9 + 0] + stress[i * 9 + 4]);
    } else if (mdl->m_type == PLANE_STRESS) {
    } else {
        printf("Unknown model type: %d\n", mdl->m_type);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < 9*n_node; i++)
        stress[i] *= mdl->E;
        
    free(rhs);
}

// s_node = 1/area int_patch s_elem dA
//        = 1/area sum_elem int_elem s_elem dA
//        = 1/area sum_elem sum_q w_q s_elem(xi_q) det(J)
void compute_nodal_stress_avg(FE_Model *mdl, const double *u, double *stress) {
    size_t nq, num[4];
    double w[4], xi[4][2], phi[4][4], dph[4][4][2];
    double det, dphi[4][2], x_loc[2], sig[4];
    double u_node[8], x_node[4][2], h[4][4], scale, ir;

    int n_elem = mdl->n_elem;
    int nl = mdl->n_local;
    const size_t *elem_nodes = mdl->elem_nodes;
    const double *coords = mdl->coords;
    const size_t *idx_map = mdl->idx_map;
    double nu = mdl->nu;

    set_hooke_matrix(1., nu, mdl->m_type, h);
    compute_shape_functions(mdl->e_type, &nq, w, xi, phi, dph, 0);
    double *den = calloc(mdl->n_node, sizeof(double));

    for (size_t i = 0; i < n_elem; i++) {
        const size_t *e_nodes = &elem_nodes[nl * i];
        SET_ELEM_INFO_U(nl, e_nodes, idx_map, coords, u, x_node, u_node, num);
        for (int q = 0; q < nq; q++) {
            set_geometry(nl, x_node, phi[q], dph[q], x_loc, &det, dphi);
            ir = (mdl->m_type == AXISYMMETRIC) ? det / x_loc[0] : 0.;
            compute_local_stress(nl, phi[q], dphi, u_node, h, ir, sig);
            for (size_t j = 0; j < nl; j++) {
                node_idx = e_nodes[j] - 1;
                den[node_idx] += w[q] * phi[q][j] * det;
                scale = w[q] * phi[q][j];
                // 1/det in eps/sig and integral det cancel out
                stress[9 * node_idx + 0] += scale * sig[0];
                stress[9 * node_idx + 1] += scale * sig[2];
                stress[9 * node_idx + 3] += scale * sig[2];
                stress[9 * node_idx + 4] += scale * sig[1];
                stress[9 * node_idx + 8] += scale * sig[3];
            }
        }
    }
    for (size_t j = 0; j < mdl->n_node; j++) {
        stress[9 * j + 0] *= mdl->E / den[j];
        stress[9 * j + 1] *= mdl->E / den[j];
        stress[9 * j + 3] *= mdl->E / den[j];
        stress[9 * j + 4] *= mdl->E / den[j];
        stress[9 * j + 8] *= mdl->E / den[j];
    }
    if (mdl->m_type == PLANE_STRAIN) {
        for (size_t j = 0; j < mdl->n_node; j++) {
            stress[9 * j + 8] = nu * (stress[9 * j + 0] + stress[9 * j + 4]);
        }
    }
    free(den);
}

/**
 * @brief Compute the forces on the boundary edges
 * @param model Finite element model
 * @param stress Stress tensor (n_node x (3*3))
 * @param data Array that stores (x, y, z, fx, fy, fz) for each edge
 * @param n_steps Number of steps
 * @param s Step (eigenmode)
 */
void compute_edge_forces(
    FE_Model *model, const double *stress, double *data, int n_steps, int s
) {
    size_t n1, n2;
    size_t idx_x, idx_f;
    size_t nnb = model->n_bd_edge;
    double norm, nx, ny, s11, s12, s22;

    for (size_t i = 0; i < nnb; i++) {
        idx_x = i * (3 + 3 * n_steps);
        idx_f = i * (3 + 3 * n_steps) + 3 + 3 * s;
        n1 = model->bd_edges[4 * i + 0] - 1;
        n2 = model->bd_edges[4 * i + 1] - 1;
        nx = model->coords[2 * n1 + 0] + model->coords[2 * n2 + 0];
        ny = model->coords[2 * n1 + 1] + model->coords[2 * n2 + 1];
        data[idx_x + 0] = nx / 2. * model->L_ref;
        data[idx_x + 1] = ny / 2. * model->L_ref;
        if (model->m_type == AXISYMMETRIC &&
            fabs(data[idx_x + 0]) < ZERO_RADIUS) {
            data[idx_x + 0] = 0.;
            continue;
        }
        nx = model->coords[2 * n2 + 1] - model->coords[2 * n1 + 1];
        ny = model->coords[2 * n1 + 0] - model->coords[2 * n2 + 0];
        norm = hypot(nx, ny);
        nx /= norm;
        ny /= norm;
        s11 = (stress[9 * n1 + 0] + stress[9 * n2 + 0]) / 2.;
        s12 = (stress[9 * n1 + 1] + stress[9 * n2 + 1]) / 2.;
        s22 = (stress[9 * n1 + 4] + stress[9 * n2 + 4]) / 2.;
        data[idx_f + 0] = s11 * nx + s12 * ny;
        data[idx_f + 1] = s12 * nx + s22 * ny;
        data[idx_f + 2] = 0.;
    }
}

double von_mises(double t[9]) {
    double v1 = SQUARE(t[0] - t[4]) + SQUARE(t[4] - t[8]) + SQUARE(t[8] - t[0]);
    double v2 = SQUARE(t[1]) + SQUARE(t[2]) + SQUARE(t[5]);
    return sqrt(0.5 * v1 + 3. * v2);
}

void tensor_eigws(double t[9], Model2D m_type, double eigws[2]) {
    // s_11  s_12  0
    // s_12  s_22  0
    //  0     0   s_33
    double tmp1, tmp2, l1, l2, l3;
    tmp1 = (t[0] + t[4]) / 2.;
    tmp2 = t[0] * t[4] - t[1] * t[1];
    l1 = tmp1 + sqrt(tmp1 * tmp1 - tmp2);
    l2 = tmp1 - sqrt(tmp1 * tmp1 - tmp2);
    l3 = t[8];
    eigws[0] = fmin(fmin(l1, l2), l3);
    eigws[1] = fmax(fmax(l1, l2), l3);
}

int compare_double(const void *a, const void *b) {
    return (*(double *)a > *(double *)b) - (*(double *)a < *(double *)b);
}

double compute_percentile(size_t n_node, const double *u_sol, double prc) {
    int idx;
    double norm;
    double *norms = malloc(n_node * sizeof(double));
    for (size_t i = 0; i < n_node; i++) {
        norm = hypot(u_sol[2 * i + 0], u_sol[2 * i + 1]);
        norms[i] = norm;
    }
    qsort(norms, n_node, sizeof(double), compare_double);
    idx = MAX(0, MIN(n_node - 1, (int)(prc * n_node)));
    return norms[idx];
}

int sort_edge(const void *a, const void *b) {
    Edge *edge_a = (Edge *)a;
    Edge *edge_b = (Edge *)b;
    size_t a_min = MIN(edge_a->n1, edge_a->n2);
    size_t b_min = MIN(edge_b->n1, edge_b->n2);
    if (a_min != b_min) {
        return a_min - b_min;
    } else {
        size_t a_max = MAX(edge_a->n1, edge_a->n2);
        size_t b_max = MAX(edge_b->n1, edge_b->n2);
        return a_max - b_max;
    }
}

void edge_compress(Edge *edges, size_t n_edges, size_t *ne, size_t *n_bd_e) {
    size_t n = 0;
    size_t n_bd = 0;
    for (size_t i = 0; i < n_edges; i++, n++) {
        size_t min1 = MIN(edges[i + 0].n1, edges[i + 0].n2);
        size_t max1 = MAX(edges[i + 0].n1, edges[i + 0].n2);
        size_t min2 = MIN(edges[i + 1].n1, edges[i + 1].n2);
        size_t max2 = MAX(edges[i + 1].n1, edges[i + 1].n2);
        // No problem at i = n_edges - 1 because of ghost edge
        edges[n] = edges[i];
        if (i < n_edges - 1 && min1 == min2 && max1 == max2) {
            edges[n].e2 = edges[i + 1].e1;
            i++;
        } else { // Boundary edge
            n_bd++;
        }
    }
    *ne = n;
    *n_bd_e = n_bd;
}

/**
 * @brief Compute the edge list of a mesh
 * @param n_loc Number of nodes per element
 * @param n_elem Number of elements
 * @param elems Element nodes (1-based indexing)
 * @param n_bd_edge_ptr Number of edges on the boundary
 * @param n_edge_ptr Number of unique edges
 * @param edges_ptr List of unique edges
 */
void create_edges(
    size_t n_loc,
    size_t n_elem,
    size_t *elems,
    size_t *n_bd_edge_ptr,
    size_t *n_edge_ptr,
    Edge **edges_ptr
) {
    size_t s = 0; // keep 1-based indexing
    Edge *edges = malloc((n_loc * n_elem + 1) * sizeof(Edge)); // 1 ghost edge
    for (size_t i = 0; i < n_elem; i++) {
        for (size_t j = 0; j < n_loc; j++) {
            edges[n_loc * i + j].e1 = i;
            edges[n_loc * i + j].e2 = n_elem; // Mark as boundary edge
            edges[n_loc * i + j].n1 = elems[n_loc * i + j] - s;
            edges[n_loc * i + j].n2 = elems[n_loc * i + (j + 1) % n_loc] - s;
        }
    }

    // Sort edges
    qsort(edges, n_loc * n_elem, sizeof(Edge), sort_edge);

    // Compress edges
    size_t n_edges, n_bd;
    edge_compress(edges, n_loc * n_elem, &n_edges, &n_bd);
    edges = realloc(edges, n_edges * sizeof(Edge));

    *n_bd_edge_ptr = n_bd;
    *n_edge_ptr = n_edges;
    *edges_ptr = edges;
}

int get_local_edge(size_t n1, size_t n2, size_t n_loc, size_t *nodes) {
    for (size_t i = 0; i < n_loc; i++) {
        if (nodes[i] == n1 && nodes[(i + 1) % n_loc] == n2) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Find all boundary nodes in a mesh
 * @param model Finite element model
 * @note The boundary nodes are stored in model->bd_nodes
 */
void find_boundary_nodes(FE_Model *model) {

    size_t n_bd_edge, n_edge;
    Edge *edges;
    size_t n_loc = model->n_local;
    size_t n_elem = model->n_elem;
    size_t *elems = model->elem_nodes;
    create_edges(n_loc, n_elem, elems, &n_bd_edge, &n_edge, &edges);

    // Allocate boundary edges
    const int n_info = 4;
    model->n_bd_edge = n_bd_edge;
    model->bd_edges = malloc(n_info * n_bd_edge * sizeof(size_t));
    for (size_t i = 0, j = 0; i < n_edge; i++) {
        if (edges[i].e2 == n_elem) {
            model->bd_edges[n_info * j + 0] = edges[i].n1;
            model->bd_edges[n_info * j + 1] = edges[i].n2;
            model->bd_edges[n_info * j + 2] = edges[i].e1;
            model->bd_edges[n_info * j + 3] = get_local_edge(
                edges[i].n1, edges[i].n2, n_loc, &elems[n_loc * edges[i].e1]
            );
            j++;
        }
    }

    free(edges);
}

