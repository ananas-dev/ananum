#include "models.h"
#include "utils_models.h"
#include <gmshc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// EXAM 2024
// -----------------------------------------------------------------------------

static double scale;
static const double W = 2.0;
static const double H = 1.0;

static const double _E = 1e6;
static const double _nu = 0.40;
static const double _rho = 1e3;
static const double _L = H;

void set_physics_exam(double params[4], Model2D *type) {
    params[0] = _E;
    params[1] = _nu;
    params[2] = _rho;
    params[3] = _L;
    *type = PLANE_STRESS;
}

void set_bk_source_exam(double rho, const double xy[2], double f[2]) {
    double x = xy[0] * _L;
    double y = xy[1] * _L;
    f[0] = 0. * x * y;
    f[1] = 0.;
}

void set_bd_disp_exam(int entity, char kind, const double xy[2], double u[1]) {
    if (entity == 3 && kind == 'y') {
        u[0] = -0.001;
    } else {
        u[0] = 0.;
    }
}

void set_bd_force_exam(int entity, char kind, const double xy[2], double f[2]) {
    f[0] = 0.;
    f[1] = 0.;
}

double size_field_exam(
    int dim, int tag, double x, double y, double z, double lc, void *data
) {
    double width = 2.;
    double height = 1.;
    double hmin = 0.03;
    double hmax = 0.3;
    double dref = 0.5;
    double d = hypot(x - width / 2., y - height / 2.);
    double h = scale * hermite(d, dref, hmin, hmax);
    return h;
}

void mesh_exam(double mesh_size_factor, int e_type) {
    int ierr;
    gmshModelOccAddRectangle(-W / 2., -H / 2., 0., W, H, -1, 0.0, &ierr);
    gmshModelOccSynchronize(&ierr);

    int force_x[] = {};
    int force_y[] = {};
    int force_n[] = {};
    int force_t[] = {};
    int dirichlet_x[] = {4, 3};
    int dirichlet_y[] = {1, 3};
    int dirichlet_n[] = {};
    int dirichlet_t[] = {};
    gmshModelAddPhysicalGroup(1, force_x, 0, 1, "force_x", &ierr);
    gmshModelAddPhysicalGroup(1, force_y, 0, 2, "force_y", &ierr);
    gmshModelAddPhysicalGroup(1, force_n, 0, 3, "force_n", &ierr);
    gmshModelAddPhysicalGroup(1, force_t, 0, 4, "force_t", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_x, 2, 5, "fix_x", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_y, 2, 6, "fix_y", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_n, 0, 7, "fix_n", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_t, 0, 8, "fix_t", &ierr);
    // gmshModelMeshSetTransfiniteCurve(1, 2, "Progression", 1.0, &ierr);
    // gmshModelMeshSetTransfiniteCurve(2, 2, "Progression", 1.0, &ierr);
    // gmshModelMeshSetTransfiniteCurve(3, 2, "Progression", 1.0, &ierr);
    // gmshModelMeshSetTransfiniteCurve(4, 2, "Progression", 1.0, &ierr);
    scale = mesh_size_factor; // set global variable
    gmshModelMeshSetSizeCallback(size_field_exam, NULL, &ierr);
    ierr = mesh_tri_quad(e_type);

    return;
}
