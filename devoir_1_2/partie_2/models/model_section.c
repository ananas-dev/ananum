#include "models.h"
#include "utils_models.h"
#include <gmshc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Beam rotated H Cross-section
// -----------------------------------------------------------------------------

static double scale;
static const double W = 2.;
static const double H = 1.;
static const double A = 0.25;
static const double B = 0.25;

static const double _E = 211e9;
static const double _nu = 0.30;
static const double _rho = 7.85e3;
static const double _L = H;

void set_physics_section(double params[4], Model2D *type) {
    params[0] = _E;
    params[1] = _nu;
    params[2] = _rho;
    params[3] = _L;
    *type = PLANE_STRAIN;
}

void set_bk_source_section(double rho, const double xy[2], double f[2]) {
    double x = xy[0] * _L;
    double y = xy[1] * _L;
    f[0] = 0. * x * y;
    f[1] = -rho * 9.81;
}

void set_bd_disp_section(
    int entity, char kind, const double xy[2], double u[1]
) {
    u[0] = 0.;
}

void set_bd_force_section(
    int entity, char kind, const double xy[2], double f[2]
) {
    if (0) {

    } else {
        f[0] = 0.;
        f[1] = 0.;
    }
}

double size_field_section(
    int dim, int tag, double x, double y, double z, double lc, void *data
) {
    return scale * 0.1;
}

void mesh_section(double mesh_size_factor, int e_type) {
    double r = (0.5 - B) * H;
    double x1 = A / 2. * W + r;

    int ierr;
    int idRect =
        gmshModelOccAddRectangle(0., -H / 2., 0., W / 2., H, -1, 0.0, &ierr);
    int idSlit =
        gmshModelOccAddRectangle(x1, -r, 0., W / 2., 2 * r, -1, 0.0, &ierr);
    int idHole =
        gmshModelOccAddDisk(x1, 0., 0., r, r, -1, NULL, 0, NULL, 0, &ierr);
    int objs[2] = {2, idRect};
    int tool[4] = {2, idHole, 2, idSlit};
    gmshModelOccCut(
        objs, 2, tool, 4, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
    );
    gmshModelOccSynchronize(&ierr);

    int force_x[] = {};
    int force_y[] = {};
    int force_n[] = {4};
    int force_t[] = {4};
    int dirichlet_x[] = {};
    int dirichlet_y[] = {};
    int dirichlet_n[] = {1, 8}; // 5
    int dirichlet_t[] = {};
    gmshModelAddPhysicalGroup(1, force_x, 0, 1, "force_x", &ierr);
    gmshModelAddPhysicalGroup(1, force_y, 0, 2, "force_y", &ierr);
    gmshModelAddPhysicalGroup(1, force_n, 1, 3, "force_n", &ierr);
    gmshModelAddPhysicalGroup(1, force_t, 1, 4, "force_t", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_x, 0, 5, "fix_x", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_y, 0, 6, "fix_y", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_n, 2, 7, "fix_n", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_t, 0, 8, "fix_t", &ierr);

    scale = mesh_size_factor; // set global variable
    gmshModelMeshSetSizeCallback(size_field_section, NULL, &ierr);
    ierr = mesh_tri_quad(e_type);
    // gmshFltkRun(&ierr);
    // exit(0);

    return;
}
