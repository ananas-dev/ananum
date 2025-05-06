#include "utils_gmsh.h"
#include "model.h"
#include "utils.h"
#include <gmshc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNZIP6(a) &(a[0]), &(a[1]), &(a[2]), &(a[3]), &(a[4]), &(a[5])
#define PLOT gmshViewAddHomogeneousModelData
#define UPDATE_MIN(a, b) ((a) = (a) < (b) ? (a) : (b))
#define UPDATE_MAX(a, b) ((a) = (a) > (b) ? (a) : (b))

#define CART_TO_POLAR 0
#define DISPLAY_AVG 0

void load_mesh(FE_Model *model) {
    int ierr;
    int e_type = model->e_type;
    gmshModelMeshRebuildNodeCache(1, &ierr);
    double *coord_bad, *coords;
    size_t *elem_tags, *node_tags, *e_nodes;
    size_t tmp, max_node, n_node, n_elem;

    size_t *old_tags, *new_tags, old_tags_n, new_tags_n;
    gmshModelMeshComputeRenumbering(
        &old_tags, &old_tags_n, &new_tags, &new_tags_n, "RCMK", NULL, 0, &ierr
    );
    gmshModelMeshRenumberNodes(
        old_tags, old_tags_n, new_tags, new_tags_n, &ierr
    );

    gmshModelMeshGetNodes(
        &node_tags, &n_node, &coord_bad, &tmp, NULL, NULL, 2, -1, 1, 0, &ierr
    );
    gmshModelMeshGetElementsByType(
        e_type, &elem_tags, &n_elem, &e_nodes, &tmp, -1, 0, 1, &ierr
    );
    max_node = 0;
    for (size_t i = 0; i < n_node; i++) {
        max_node = MAX(max_node, node_tags[i]);
    }
    if (max_node != n_node) {
        printf("Nodes not contig. : max: %zu  nn: %zu\n", max_node, n_node);
        exit(EXIT_FAILURE);
    }
    coords = (double *)malloc(2 * n_node * sizeof(double));
    for (size_t i = 0; i < n_node; i++) {
        tmp = node_tags[i] - 1; // 0-based indexing
        coords[2 * tmp + 0] = coord_bad[3 * i + 0] / model->L_ref;
        coords[2 * tmp + 1] = coord_bad[3 * i + 1] / model->L_ref;
    }

    model->n_elem = n_elem;
    model->n_local = e_type + 1;
    model->elem_nodes = e_nodes;
    model->n_node = n_node;
    model->coords = coords;
    model->e_tags = elem_tags;

    find_boundary_nodes(model);

    free(coord_bad);
    free(node_tags);
}

/**
 * @brief Enforce boundary conditions
 * @param model Finite element model
 * @param rhs Right-hand side of the system
 */
void enforce_bd_conditions(FE_Model *model, double *rhs) {
    int ierr;
    int *dt_phys, *ent;
    size_t dt_phys_n, entity_n;
    size_t *bd_nodes;
    size_t bd_n; // n_edge; (to be used for improvement)
    char *name, bkind;

    gmshModelGetPhysicalGroups(&dt_phys, &dt_phys_n, 1, &ierr);
    // printf("Physical groups: %zu\n", dt_phys_n);
    for (size_t i = 0; i < dt_phys_n; i += 2) {
        gmshModelGetPhysicalName(dt_phys[i], dt_phys[i + 1], &name, &ierr);
        // printf("  Physical group: %s\n", name);
        bkind = name[strlen(name) - 1];
        if (bkind != 'x' && bkind != 'y' && bkind != 'n' && bkind != 't') {
            exit(EXIT_FAILURE);
        }
        gmshModelGetEntitiesForPhysicalName(name, &ent, &entity_n, &ierr);
        for (size_t e = 0; e < entity_n; e += 2) {
            gmshModelMeshGetElementEdgeNodes(
                1, &bd_nodes, &bd_n, ent[e + 1], 1, 0, 1, &ierr
            );
            // size_t n_edge = bd_n / 2;
            // printf("    Entity %d : %zu nodes\n", ent[e + 1], n_edge + 1);
            if (strncmp(name, "fix", 3) == 0) {
                apply_dirichlet(model, ent[e + 1], bd_n, bd_nodes, bkind, rhs);
            } else if (strncmp(name, "force", 5) == 0) {
                apply_force(model, ent[e + 1], bd_n, bd_nodes, bkind, rhs);
            } else {
                printf("Unknown boundary condition: %s\n", name);
                exit(EXIT_FAILURE);
            }
            free(bd_nodes);
        }
        free(name);
        free(ent);
    }
    free(dt_phys);
}

/**
 * @brief Add the displacement field in Gmsh
 * @param model Finite element model
 * @param sol Solution vector (ux, uy)
 * @param views Gmsh views
 * @param n_steps Number of steps
 * @param s Step (eigenmodes)
 * @param force Array that stores the forces on the edges
 * @param bounds Bounds of the views
 */
void visualize_stress(
    FE_Model *model,
    const double *sol,
    const int *views,
    int n_steps,
    int s,
    double *force,
    double *bds
) {
    int ierr;
    size_t nn = model->n_node;
    size_t *nodes = malloc(nn * sizeof(size_t));
    double eig_stress[2];
    const int inds[4] = {0, 1, 4, 8};
    const int size = 9 * nn;
    double *sig_lsq = calloc(size, sizeof(double));
    compute_nodal_stress_lsq(model, sol, sig_lsq);
    compute_edge_forces(model, sig_lsq, force, n_steps, s);
    // compute_bd_forces(model, sol, force, n_steps, s);
#if DISPLAY_AVG
    double *sig_avg = calloc(size, sizeof(double));
    compute_nodal_stress_avg(model, sol, sig_avg);
#endif
    if (model->m_type != AXISYMMETRIC && CART_TO_POLAR) {
        cartesian_to_polar(nn, sig_lsq, model->coords);
    }
    for (size_t i = 0; i < nn; i++) {
        nodes[i] = i + 1;
        UPDATE_MAX(bds[2 * 2 + 1], von_mises(&sig_lsq[9 * i]));
        tensor_eigws(&sig_lsq[9 * i], model->m_type, eig_stress);
        UPDATE_MIN(bds[2 * 3 + 0], eig_stress[1]);
        UPDATE_MAX(bds[2 * 3 + 1], eig_stress[1]);
        UPDATE_MIN(bds[2 * 4 + 0], eig_stress[0]);
        UPDATE_MAX(bds[2 * 4 + 1], eig_stress[0]);
        for (int j = 0; j < 4; j++) {
            UPDATE_MIN(bds[2 * (5 + j) + 0], -fabs(sig_lsq[9 * i + inds[j]]));
            UPDATE_MAX(bds[2 * (5 + j) + 1], +fabs(sig_lsq[9 * i + inds[j]]));
        }
#if DISPLAY_AVG
        UPDATE_MAX(bds[2 * 9 + 1], von_mises(&sig_avg[9 * i]));
        tensor_eigws(&sig_avg[9 * i], model->m_type, eig_stress);
        UPDATE_MIN(bds[2 * 10 + 0], eig_stress[1]);
        UPDATE_MAX(bds[2 * 10 + 1], eig_stress[1]);
        UPDATE_MIN(bds[2 * 11 + 0], eig_stress[0]);
        UPDATE_MAX(bds[2 * 11 + 1], eig_stress[0]);
        for (int j = 0; j < 4; j++) {
            UPDATE_MIN(bds[2 * (12 + j) + 0], -fabs(sig_avg[9 * i + inds[j]]));
            UPDATE_MAX(bds[2 * (12 + j) + 1], +fabs(sig_avg[9 * i + inds[j]]));
        }
#endif
    }

    char *name;
    char *dtype = "NodeData";
    gmshModelGetCurrent(&name, &ierr);

    PLOT(views[2], s, name, dtype, nodes, nn, sig_lsq, size, 0., 9, -1, &ierr);
    free(sig_lsq);
#if DISPLAY_AVG
    PLOT(views[9], s, name, dtype, nodes, nn, sig_avg, size, 0., 9, -1, &ierr);
    free(sig_avg);
#endif
    free(nodes);
    free(name);
}

/**
 * @brief Add the displacement field in Gmsh
 * @param model Finite element model
 * @param sol Solution vector (ux, uy)
 * @param view Gmsh view_tag
 * @param step Step (eigenmodes)
 * @param bds Bounds of the views
 */
void visualize_disp(
    FE_Model *model, const double *sol, int view, int step, double *bounds
) {
    int ierr;
    size_t num;
    size_t nn = model->n_node;
    size_t *nodes = malloc(nn * sizeof(size_t));
    double *disp = malloc(3 * nn * sizeof(double));
    double norm;

    for (size_t i = 0; i < nn; i++) {
        nodes[i] = i + 1;
        num = model->idx_map[i];
        disp[3 * i + 0] = sol[2 * num + 0] * model->L_ref;
        disp[3 * i + 1] = sol[2 * num + 1] * model->L_ref;
        disp[3 * i + 2] = 0.;
        norm = hypot(disp[3 * i + 0], disp[3 * i + 1]);
        bounds[1] = fmax(bounds[1], norm);
    }
    bounds[0] = fmax(bounds[0], compute_percentile(nn, disp, 0.95));

    char *name;
    char *dtype = "NodeData";
    gmshModelGetCurrent(&name, &ierr);
    PLOT(view, step, name, dtype, nodes, nn, disp, 3 * nn, 0., 3, -1, &ierr);
    gmshViewOptionSetNumber(view, "VectorType", 5, &ierr);
    gmshViewOptionSetNumber(view, "DrawPoints", 0, &ierr);
    free(nodes);
    free(disp);
    free(name);
}

/**
 * @brief Add the boundary forces in Gmsh
 * @param model Finite element model
 * @param data List of size n_views * n_bd_edge * 6
 * @param view List of views
 * @param n_steps Number of steps in gmsh
 * @param bounds List of bounds associated to each view
 * @note Needs all steps to be computed (by visualize_stress)
 */
void visualize_bd_forces(
    FE_Model *model, const double *data, int view, int n_steps, double *bds
) {
    int ierr;
    char *name;
    double norm, max_force;
    size_t nnb = model->n_bd_edge;
    size_t idx;
    int incx = 3 + 3 * n_steps;

    for (size_t s = 0; s < n_steps; s++) {
        max_force = 0.;
        for (size_t i = 0; i < nnb; i++) {
            idx = i * (incx) + (3 + 3 * s);
            norm = hypot(data[idx + 0], data[idx + 1]);
            max_force = fmax(max_force, norm);
        }
        bds[1] = fmax(bds[1], max_force);
    }

    gmshModelGetCurrent(&name, &ierr);
    gmshViewAddListData(view, "VP", nnb, data, nnb * incx, &ierr);
    gmshViewOptionSetNumber(view, "VectorType", 2, &ierr);
    free(name);
}

/**
 * @brief Add the Gmsh views
 * @param views_ptr Pointer to the Gmsh view list
 * @param n_views_ptr Pointer to the number of views
 * @param bounds_ptr Pointer to the bounds of the views
 */
void add_gmsh_views(int **views_ptr, int *n_views_ptr, double **bounds_ptr) {
    int ierr, *prev_views;
    size_t n_prev;
    gmshViewGetTags(&prev_views, &n_prev, &ierr);
    for (int i = 0; i < n_prev; i++) {
        gmshViewRemove(prev_views[i], &ierr);
    }
    free(prev_views);
    int n_views = 2 + 7 + 7 * DISPLAY_AVG;
    int *views = malloc(n_views * sizeof(int));
    double *bounds = malloc(2 * n_views * sizeof(double));
    for (int i = 0; i < n_views; i++) {
        bounds[2 * i + 0] = bounds[2 * i + 1] = 0.;
        views[i] = -1;
    }
    views[0] = gmshViewAdd("forces", -1, &ierr);       // Boundary forces
    views[1] = gmshViewAdd("displacement", -1, &ierr); // Displacement
    views[2] = gmshViewAdd("stress lsq", -1, &ierr);   // Stress Least Squares
#if DISPLAY_AVG
    views[9] = gmshViewAdd("stress avg", -1, &ierr); // Stress Average
#endif

    *views_ptr = views;
    *n_views_ptr = n_views;
    *bounds_ptr = bounds;
    return;
}

void create_tensor_aliases(int *views) {
    int ierr;
    views[3] = gmshViewAddAlias(views[2], 1, -1, &ierr); // Stress max eigw
    gmshViewOptionSetNumber(views[3], "TensorType", 2, &ierr);
    views[4] = gmshViewAddAlias(views[2], 1, -1, &ierr); // Stress max eigw
    gmshViewOptionSetNumber(views[4], "TensorType", 3, &ierr);

    views[5] = gmshViewAddAlias(views[2], 1, -1, &ierr); // Stress xx
    gmshViewOptionSetNumber(views[5], "ForceNumComponents", 1, &ierr);
    gmshViewOptionSetNumber(views[5], "ComponentMap0", 0, &ierr);
    views[6] = gmshViewAddAlias(views[2], 1, -1, &ierr); // Stress xy
    gmshViewOptionSetNumber(views[6], "ForceNumComponents", 1, &ierr);
    gmshViewOptionSetNumber(views[6], "ComponentMap0", 1, &ierr);
    views[7] = gmshViewAddAlias(views[2], 1, -1, &ierr); // Stress yy
    gmshViewOptionSetNumber(views[7], "ForceNumComponents", 1, &ierr);
    gmshViewOptionSetNumber(views[7], "ComponentMap0", 4, &ierr);
    views[8] = gmshViewAddAlias(views[2], 1, -1, &ierr); // Stress zz
    gmshViewOptionSetNumber(views[8], "ForceNumComponents", 1, &ierr);
    gmshViewOptionSetNumber(views[8], "ComponentMap0", 8, &ierr);

#if DISPLAY_AVG
    views[10] = gmshViewAddAlias(views[9], 1, -1, &ierr); // Stress max eigw
    gmshViewOptionSetNumber(views[10], "TensorType", 2, &ierr);
    views[11] = gmshViewAddAlias(views[9], 1, -1, &ierr); // Stress min eigw
    gmshViewOptionSetNumber(views[11], "TensorType", 3, &ierr);

    views[12] = gmshViewAddAlias(views[9], 1, -1, &ierr); // Stress xx
    gmshViewOptionSetNumber(views[12], "ForceNumComponents", 1, &ierr);
    gmshViewOptionSetNumber(views[12], "ComponentMap0", 0, &ierr);
    views[13] = gmshViewAddAlias(views[9], 1, -1, &ierr); // Stress xy
    gmshViewOptionSetNumber(views[13], "ForceNumComponents", 1, &ierr);
    gmshViewOptionSetNumber(views[13], "ComponentMap0", 1, &ierr);
    views[14] = gmshViewAddAlias(views[9], 1, -1, &ierr); // Stress yy
    gmshViewOptionSetNumber(views[14], "ForceNumComponents", 1, &ierr);
    gmshViewOptionSetNumber(views[14], "ComponentMap0", 4, &ierr);
    views[15] = gmshViewAddAlias(views[9], 1, -1, &ierr); // Stress zz
    gmshViewOptionSetNumber(views[15], "ForceNumComponents", 1, &ierr);
    gmshViewOptionSetNumber(views[15], "ComponentMap0", 8, &ierr);
#endif
}

/**
 * @brief Set the bounds of the views
 * @param n_views Number of views
 * @param views Gmsh view list
 * @param bounds Bounds of the views (min1, max1, min2, max2, ...)
 * @param mode 1: Linear system solve, 2: Eigenmodes
 */
void set_view_options(int n_views, int *views, double *bounds) {
    int ierr;

    // Set view field ranges
    double min, max;
    for (int i = 0; i < n_views; i++) {
        min = bounds[2 * i + 0];
        max = bounds[2 * i + 1];
        gmshViewOptionSetNumber(views[i], "RangeType", 2, &ierr);
        gmshViewOptionSetNumber(views[i], "CustomMin", min, &ierr);
        gmshViewOptionSetNumber(views[i], "CustomMax", max, &ierr);
        // gmshViewOptionSetNumber(views[i], "ShowScale", 0, &ierr);
    }
    gmshViewOptionSetNumber(views[1], "RangeType", 2, &ierr);
    gmshViewOptionSetNumber(views[1], "CustomMin", 0., &ierr);

    // Set displacement factor
    double factor, dl, box[6];
    gmshModelGetBoundingBox(-1, -1, UNZIP6(box), &ierr);
    dl = hypot(box[3] - box[0], box[4] - box[1]);
    // factor = bounds[2 * 1 + 1];
    factor = bounds[2 * 1 + 0];
    factor = (factor < 1e-20) ? 1. : (dl / 20.) / factor;
    gmshViewOptionSetNumber(views[1], "DisplacementFactor", factor, &ierr);

    // Hide the mesh
    gmshOptionSetNumber("Mesh.SurfaceEdges", 0, &ierr);

    // Hide the fields
    for (int i = 0; i < n_views; i++)
        gmshViewOptionSetNumber(views[i], "Visible", 0, &ierr);
    gmshViewOptionSetNumber(views[1], "Visible", 1, &ierr);
}

/**
 * @brief Revolve the geometry in Gmsh
 * @param model Finite element model
 */
void revolve_geometry(FE_Model *model) {
    if (model->m_type != AXISYMMETRIC)
        return;
    int ierr;
    int *dim_tags, *out_dim_tags;
    size_t dim_tags_n, out_dim_tags_n;
    gmshModelGetEntities(&dim_tags, &dim_tags_n, 2, &ierr);
    printf("Dim tags: %zu\n", dim_tags_n);
    for (size_t i = 0; i < dim_tags_n; i++) {
        printf("Dim tag: %d\n", dim_tags[i]);
    }

    // gmshModelOccCopy(
    //     dim_tags, dim_tags_n, &out_dim_tags, &out_dim_tags_n, &ierr
    // );
    // gmshModelOccRotate(
    //     out_dim_tags, out_dim_tags_n, 0., 0., 0., 0., 1., 0., 1e-1, &ierr
    // );

    // clang-format off
    gmshModelOccRevolve(
        dim_tags, dim_tags_n, 0., 0., 0., 0., 1., 0., 2.0 * M_PI, 
        &out_dim_tags, &out_dim_tags_n, NULL, 0, NULL, 0, 0, &ierr
    );
    // clang-format on

    gmshModelOccSynchronize(&ierr);
    gmshOptionSetNumber("Geometry.Surfaces", 1, &ierr);
    gmshOptionSetNumber("Geometry.SurfaceType", 1, &ierr);
    free(dim_tags);
    free(out_dim_tags);
}