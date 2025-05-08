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

//./deformation <model> <lc> <T> <dt> <initial.txt> <final.txt> <time.txt> <I>
int main(int argc, char *argv[]) {



    //on check si tous les éléments donnés sont bons 
    int ierr;
    double mesh_size_ratio;
    double T_final;
    double dt;
    int node_I;
    FILE *initial_file = fopen(argv[5], "r");
    FILE *final_file = fopen(argv[6], "w");
    FILE *time_file = fopen(argv[7], "w");
    
    if ((argc < 9) || (sscanf(argv[2], "%lf", &mesh_size_ratio)) != 1) { //0
        printf("Erreur: Impossible de convertir <lf> ('%s') en double.\n");
        return -1;
    }
    if(sscanf(argv[3], "%lf", &T_final) != 1){
        printf("Erreur: Impossible de convertir <T_final> ('%s') en double.\n",argv[3]);
        return -1;
    }
    if(sscanf(argv[4], "%lf", &dt) != 1){
        printf("Erreur: Impossible de convertir <dt> ('%s') en double.\n", argv[4]);
        return -1;
    }
    if(sscanf(argv[8], "%d", &node_I) != 1){
        printf("Erreur: Impossible de convertir <I> ('%s') en entier.\n",argv[8]);
        return -1;
    }
    if(initial_file == NULL){
        printf("Erreur : impossible d'ouvrir le fichier %s\n", argv[5]);
        return -1;
    }
    if(final_file == NULL){
        printf("Erreur : impossible d'ouvrir le fichier %s\n", argv[6]);
        return -1;
    }
    if(time_file == NULL){
        printf("Erreur : impossible d'ouvrir le fichier %s\n", argv[7]);
        return -1;
    }

    printf("Arguments et fichiers validés avec succès.\n");
    printf("  Modèle: %s\n", argv[1]);
    printf("  lc: %f\n", mesh_size_ratio);
    printf("  T_final: %f\n", T_final);
    printf("  dt: %f\n", dt);
    printf("  Fichier initial: %s\n", argv[5]);
    printf("  Fichier final: %s\n", argv[6]);
    printf("  Fichier de temps: %s\n", argv[7]);
    printf("  Noeud I: %d\n", node_I);
    
    
    // Simulation parameters
    const ElementType e_type = TRI;
    const Renumbering renum = RENUM_NO;  // let gmsh do the RCMK renumbering

    FE_Model *model = create_FE_Model(argv[1], e_type, renum);
    display_info(model, 1, NULL);

    gmshInitialize(argc, argv, 0, 0, &ierr);
    gmshOptionSetNumber("General.Verbosity", 2, &ierr);
    model->mesh_model(mesh_size_ratio, e_type);

    size_t n_dofs = 2 * model->n_node; 
    double *q0 = (double*)malloc(n_dofs * sizeof(double));
    double *q0_dot = (double*)malloc(n_dofs * sizeof(double));

    if (q0 == NULL || q0_dot == NULL) {
        fprintf(stderr, "Erreur : Échec de l'allocation mémoire pour les vecteurs de conditions initiales.\n");
        if(q0) free(q0);
        if(q0_dot) free(q0_dot);
        return 1;
    }

    for(size_t i = 0; i<n_dofs; i++){
        if(fscanf(initial_file, "%le %le %le %le\n", q0 +i*2, q0 +i*2 +1, q0_dot +i*2, q0_dot +i*2 +1) == EOF){
            printf("EOF detected");
        }
    }

    ////étape 2////
    const double beta_newmark = 0.25;
    const double gamma_newmark = 0.5;

    double *q_n = (double*)malloc(n_dofs * sizeof(double));
    double *p_n = (double*)malloc(n_dofs * sizeof(double));

    double *rhs_newmark_eq3 = (double*)malloc(n_dofs * sizeof(double));
    double *temp_vec_a = (double*)malloc(n_dofs * sizeof(double)); 
    double *temp_vec_b = (double*)malloc(n_dofs * sizeof(double));

    if (!q_n || !p_n || !rhs_newmark_eq3 || !temp_vec_a || !temp_vec_b) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les vecteurs de Newmark.\n");
        // ... (ajouter la libération de toute mémoire déjà allouée et sortie propre) ...
        free(q0); free(q0_dot); // Ceux de l'étape 1
        if(q_n) free(q_n); if(p_n) free(p_n);
        if(rhs_newmark_eq3) free(rhs_newmark_eq3); if(temp_vec_a) free(temp_vec_a); if(temp_vec_b) free(temp_vec_b);
        if(final_file) fclose(final_file);
        if(time_file) fclose(time_file);
        return 1;
    }





    load_mesh(model);
    renumber_nodes(model);
    display_info(model, 2, NULL);
    assemble_system(model);
    double *rhs = (double *)calloc(2 * model->n_node, sizeof(double));
    double *sol = (double *)calloc(2 * model->n_node, sizeof(double));
    add_bulk_source(model, rhs);
    enforce_bd_conditions(model, rhs);
    
    SymBandMatrix *Kbd = model->K;
    SymBandMatrix *Mbd = model->M;
    CSRMatrix *Ksp = band_to_sym_csr(Kbd);
    CSRMatrix *Msp = band_to_sym_csr(Mbd);
    double eps = 1e-8;
    CG(Ksp->n, Ksp->row_ptr, Ksp->col_idx, Ksp->data, rhs, sol, eps);    
    display_sol(model, sol);
    
    // Free stuff
    free_csr(Ksp);
    free_csr(Msp);
    gmshFinalize(&ierr);
    free(sol);
    free(rhs);
    free_FE_Model(model);
    return 0;
}
