#ifndef ELASTICITY_H
#define ELASTICITY_H

#include "model.h"
#include <stddef.h>

SymBandMatrix *allocate_sym_band_matrix(const size_t n, const size_t k);
void print_sym_band(SymBandMatrix *M);
void write_sym_band(SymBandMatrix *M, double *rhs, const char *filename);
void print_vector_row(size_t n, double *vec);
void print_vector(size_t n, double *vec);

CSRMatrix *band_to_sym_csr(const SymBandMatrix *band);
CSRMatrix *band_to_csr(const SymBandMatrix *band);

void sym_band_LDL(double *A, size_t n, size_t b);
void solve_sym_band(double *L, size_t n, size_t b, double *x);


#define SET_ELEM_INFO(nl, e_nodes, idx_map, coords, x_node, num)               \
    size_t node_idx;                                                           \
    for (size_t j = 0; j < (nl); j++) {                                        \
        node_idx = (e_nodes)[j] - 1;                                           \
        (x_node)[j][0] = (coords)[2 * node_idx + 0];                           \
        (x_node)[j][1] = (coords)[2 * node_idx + 1];                           \
        (num)[j] = (idx_map)[node_idx];                                        \
    }

#define SET_ELEM_INFO_U(nl, e_nodes, idx_map, coords, u, x_node, u_node, num)  \
    size_t node_idx;                                                           \
    for (size_t j = 0; j < (nl); j++) {                                        \
        node_idx = (e_nodes)[j] - 1;                                           \
        (x_node)[j][0] = (coords)[2 * node_idx + 0];                           \
        (x_node)[j][1] = (coords)[2 * node_idx + 1];                           \
        (num)[j] = (idx_map)[node_idx];                                        \
        (u_node)[2 * j + 0] = (u)[2 * (num)[j] + 0];                           \
        (u_node)[2 * j + 1] = (u)[2 * (num)[j] + 1];                           \
    }

void create_edges(
    size_t n_loc,
    size_t n_elem,
    size_t *elems,
    size_t *n_bd_edge_ptr,
    size_t *n_edge_ptr,
    Edge **edges_ptr
);
void find_boundary_nodes(FE_Model *model);


void compute_shape_functions(
    ElementType e_type,
    size_t *nq,
    double w[4],
    double xi[4][2],
    double phi[4][4],
    double dph[4][4][2],
    const int boundary
);
void set_geometry(
    const int n_loc,
    const double x_node[4][2],
    const double phi[4],
    const double dph[4][2],
    double x_ptr[2],
    double det[1],
    double dphi[4][2]
);
void set_strain_basis(
    const int n_loc,
    const double phi[4],
    const double dph[4][2],
    const double inv_r,
    double B[8][4]
);
void set_hooke_matrix(double E, double nu, int m_type, double h[4][4]);

void assemble_system(FE_Model *model);
void add_bulk_source(FE_Model *model, double *rhs);
void enforce_bd_conditions(FE_Model *model, double *rhs);
void apply_dirichlet(
    const FE_Model *model,
    const size_t entity,
    const size_t size,
    size_t *edges,
    const char kind,
    double *rhs
);
void apply_force(
    const FE_Model *model,
    const size_t entity,
    const size_t size,
    const size_t *edges,
    const char kind,
    double *rhs
);

void set_lsq_rhs(FE_Model *model, const double *u, double *rhs);
void compute_nodal_stress_lsq(FE_Model *mdl, const double *u, double *stress);
void compute_nodal_stress_avg(FE_Model *mdl, const double *u, double *stress);
void assemble_mass_lsq(FE_Model *model, SymBandMatrix *M);
void compute_edge_forces(
    FE_Model *model, const double *stress, double *data, int n_steps, int s
);
double von_mises(double t[9]);
void cartesian_to_polar(size_t n_node, double *sigma, double *x);
void compute_local_stress(
    const int n_loc,
    const double phi[4],
    const double dphi[4][2],
    const double u[8],
    const double h[4][4],
    const double ir,
    double stress[4]
);
void tensor_eigws(double t[9], Model2D m_type, double eigws[2]);
double compute_percentile(size_t n_node, const double *u_sol, double prc);

#endif
