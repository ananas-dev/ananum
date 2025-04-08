#include "devoir_2.h"
#include "utils.h"
#include "model.h"
#include "utils_gmsh.h"
#include <math.h>
#include <cblas.h>
#include <gmshc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VERBOSE 1
#define PRECISION 10

void display_sol(FE_Model *model, double *sol) {

    int ierr, n_views, *views;
    double *bounds;
    add_gmsh_views(&views, &n_views, &bounds);

    double *data_forces = malloc(6 * model->n_bd_edge * sizeof(double));
    visualize_disp(model, sol, views[1], 0, &bounds[2]);
    visualize_stress(model, sol, views, 1, 0, data_forces, bounds);
    visualize_bd_forces(model, data_forces, views[0], 1, &bounds[0]);

    create_tensor_aliases(views);
    set_view_options(n_views, views, bounds);
    gmshFltkRun(&ierr);
    gmshFltkFinalize(&ierr);
    free(data_forces);
}

void display_info(FE_Model *model, int step, struct timespec ts[4]) {

    char *m_str[3] = {"Plane stress", "Plane strain", "Axisymmetric"};
    char *r_str[4] = {"No", "X", "Y", "RCMK"};

    if (step == 1) {
        printf(
            "\n===========  Linear elasticity simulation - FEM  ===========\n\n"
        );
        printf("%30s = %s\n", "Model", model->model_name);
        printf("%30s = %s\n", "Model type", m_str[model->m_type]);
        printf("%30s = %.3e\n", "Young's Modulus E", model->E);
        printf("%30s = %.3e\n", "Poisson ratio nu", model->nu);
        printf("%30s = %.3e\n\n", "Density rho", model->rho);
    } else if (step == 2) {
        char *e_str = (model->e_type == TRI) ? "Triangle" : "Quadrilateral";
        printf("%30s = %s\n", "Element type", e_str);
        printf("%30s = %zu\n", "Number of elements", model->n_elem);
        printf("%30s = %zu\n", "Number of nodes", model->n_node);
        printf("%30s = %s\n", "Renumbering", r_str[model->renum]);
        printf("%30s = %zu\n\n", "Matrix bandwidth", 2 * model->node_band + 1);
    }
}

int not_main(int argc, char *argv[]) {

    int ierr;
    double mesh_size_ratio;
    if ((argc < 3) || (sscanf(argv[2], "%lf", &mesh_size_ratio)) != 1) {
        printf("Usage: \n./deformation <model> <mesh_size_ratio>\n");
        printf("model: one of the model implemented in models/\n");
        printf("mesh_size_ratio: mesh size factor\n");
        return -1;
    }

    // Simulation parameters
    const ElementType e_type = TRI;
    const Renumbering renum = RENUM_Y;

    FE_Model *model = create_FE_Model(argv[1], e_type, renum);
    display_info(model, 1, NULL);

    gmshInitialize(argc, argv, 0, 0, &ierr);
    gmshOptionSetNumber("General.Verbosity", 2, &ierr);
    model->mesh_model(mesh_size_ratio, e_type);

    load_mesh(model);
    renumber_nodes(model);
    display_info(model, 2, NULL);

    assemble_system(model);

    double *rhs = (double *)calloc(2 * model->n_node, sizeof(double));
    double *sol = (double *)calloc(2 * model->n_node, sizeof(double));
    add_bulk_source(model, rhs);
    enforce_bd_conditions(model, rhs);
    SymBandMatrix *Kbd = model->K;

    // TODO : start
    // CSRMatrix *Ksp = band_to_csr(Kbd); // or band_to_sym_csr(Kbd)
    // double eps = 1.;
    // CG(Ksp->n, Ksp->nnz, Ksp->row_ptr, Ksp->col_idx, Ksp->data, rhs, sol, eps);
    // free_csr(Ksp);
    // TODO : end

    // Slow direct solver without renumbering
    sym_band_LDL(Kbd->data, Kbd->n, Kbd->k);
    solve_sym_band(Kbd->data, Kbd->n, Kbd->k, rhs);
    memcpy(sol, rhs, 2 * model->n_node * sizeof(double));

    display_sol(model, sol);

    // Free stuff
    gmshFinalize(&ierr);
    free(sol);
    free(rhs);
    free_FE_Model(model);
    return 0;
}


