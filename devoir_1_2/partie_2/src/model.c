#include "model.h"
#include "models.h"
#include "renumber.h"

#include <gmshc.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FE_Model *create_FE_Model(const char *name, ElementType etp, Renumbering ren) {

    FE_Model *model = (FE_Model *)malloc(sizeof(FE_Model));
    model->model_name = name;

    model->e_type = etp; // element: triangle, quadrilateral
    model->renum = ren;  // renumbering: none, x, y, rcmk
    model->M_scalar = NULL;
    double parameters[4];

    if (strcmp(name, "beam") == 0) {
        model->mesh_model = mesh_beam;
        model->set_bk_source = set_bk_source_beam;
        model->set_bd_force = set_bd_force_beam;
        model->set_bd_disp = set_bd_disp_beam;
        set_physics_beam(parameters, &model->m_type);
    } else if (strcmp(name, "exam") == 0) {
        model->mesh_model = mesh_exam;
        model->set_bk_source = set_bk_source_exam;
        model->set_bd_force = set_bd_force_exam;
        model->set_bd_disp = set_bd_disp_exam;
        set_physics_exam(parameters, &model->m_type);
    } else if (strcmp(name, "section") == 0) {
        model->mesh_model = mesh_section;
        model->set_bk_source = set_bk_source_section;
        model->set_bd_force = set_bd_force_section;
        model->set_bd_disp = set_bd_disp_section;
        set_physics_section(parameters, &model->m_type);
    } else if (strcmp(name, "fork") == 0) {
        model->mesh_model = mesh_fork;
        model->set_bk_source = set_bk_source_fork;
        model->set_bd_force = set_bd_force_fork;
        model->set_bd_disp = set_bd_disp_fork;
        set_physics_fork(parameters, &model->m_type);
    } else {
        printf("Unknown model: %s\n", name);
        exit(EXIT_FAILURE);
    }

    model->E = parameters[0];     // Young's modulus
    model->nu = parameters[1];    // Poisson's ratio
    model->rho = parameters[2];   // Density
    model->L_ref = parameters[3]; // Reference length (just for scaling)

    return model;
}

void free_FE_Model(FE_Model *model) {
    free(model->elem_nodes);
    free(model->coords);
    free(model->idx_map);
    free(model->bd_edges);
    free_sym_band_matrix(model->M);
    free_sym_band_matrix(model->K);
    if (model->M_scalar != NULL) {
        free_sym_band_matrix(model->M_scalar);
    }
    free(model);
}

/**
 * @brief Renumber the nodes of a mesh
 * @param model Finite element model
 */
void renumber_nodes(FE_Model *model) {
    renumber(
        model->n_elem,
        model->n_local,
        model->elem_nodes,
        model->n_bd_edge,
        model->bd_edges,
        model->n_node,
        model->coords,
        &model->idx_map,
        model->renum
    );
    model->node_band = compute_band(
        model->n_elem, model->n_local, model->elem_nodes, model->idx_map
    );
}
