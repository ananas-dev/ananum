#include "models.h"
#include "utils_models.h"
#include <math.h>
#include <gmshc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Classic clamped/free beam
// -----------------------------------------------------------------------------

// Expected end displacement with uniform load
// u = -rho * g * L^4 / (8 * E * I) | I = b * h^3 / 12
// u = - 3/2 (rho * g / b) * L^4 / (E * h^3)

static double scale;
static const double W = 5.;
static const double H = 0.15;
static const double G = 9.81;

static const double _E = 40e9;
static const double _nu = 0.20;
static const double _rho = 2300.;
static const double _L = W;

void set_physics_beam(double params[4], Model2D *type) {
    params[0] = _E;
    params[1] = _nu;
    params[2] = _rho;
    params[3] = _L;
    *type = PLANE_STRESS;
}

void set_bk_source_beam(double rho, const double xy[2], double f[2]) {
    double x = xy[0] * _L;
    double y = xy[1] * _L;
    f[0] = 0. * x * y;
    f[1] = -rho * G;
}

void set_bd_disp_beam(int entity, char kind, const double xy[2], double u[1]) {
    u[0] = 0.;
}

void set_bd_force_beam(int entity, char kind, const double xy[2], double f[2]) {
    f[0] = 0.;
    f[1] = 0.;
}


// void set_bd_values_beam(int entity, int dir, size_t n_pts, double *values) {
//     double x, y;
//     for (size_t j = 0; j < n_pts; j++) {
//         x = values[2 * j + 0];
//         y = values[2 * j + 1];
//         if (entity == 1 && dir == 0) {
//             values[j] = 0. * x + 0. * y;
//         } else if (entity == 1 && dir == 1) {
//             values[j] = 0.;
//         } else if (entity == 2 && dir == 0) {
//             values[j] = 0.;
//         } else if (entity == 2 && dir == 1) {
//             values[j] = 0.;
//         } else if (entity == 3 && dir == 0) {
//             values[j] = -7.85e3 * 9.81 * 0.;
//         } else if (entity == 3 && dir == 1) {
//             values[j] = 0.;
//         } else if (entity == 4 && dir == 0) {
//             values[j] = 0.;
//         } else if (entity == 4 && dir == 1) {
//             values[j] = 0.;
//         } else {
//             values[j] = 0.;
//         }
//     }
// }

double size_field_beam(
    int dim, int tag, double x, double y, double z, double lc, void *data
) {
    double hmin = 0.05;
    double hmax = 0.15;
    double dref = W / 4.;
    double d = x;
    double h = scale * hermite(d, dref, hmin, hmax);
    return h;
}

void mesh_beam(double mesh_size_factor, int e_type) {

    // Requires plane stress
    if (0) {
        // q L⁴/8EI   ---   I = B H³/12   ---    q = ρgS   ---   S = BH
        // --> v = 3/2 (ρg) L⁴ / E H²  [m]
        double u_clamped = -3. / 2. * _rho * G * W * W * W * W / (_E * H * H);
        printf("v_expected : %10.6lf\n", u_clamped);
        // 2pi f = k²/L² sqrt(EI / ρS) = k²/L² sqrt(E H² / 12ρ) [s⁻¹]
        // double ks[4] = {4.730, 7.853, 10.996, 14.137}; // free-free
        double ks[4] = {1.875, 4.694, 7.855, 10.996}; // clamped-free
        for (int i = 0; i < 4; i++) {
            double om = SQUARE(ks[i] / W) * sqrt(_E * H * H / 12. / _rho);
            printf("    freq_%d : %7.3lf Hz\n", i, om / (2. * M_PI));
        }
        printf("\n");
    }

    int ierr;
    gmshModelOccAddRectangle(0., -H / 2., 0., W, H, -1, 0.0, &ierr);
    gmshModelOccSynchronize(&ierr);
    int force_x[] = {};
    int force_y[] = {};
    int force_n[] = {};
    int force_t[] = {};
    int dirichlet_x[] = {4};
    int dirichlet_y[] = {4};
    int dirichlet_n[] = {};
    int dirichlet_t[] = {};
    gmshModelAddPhysicalGroup(1, force_x, 0, 1, "force_x", &ierr);
    gmshModelAddPhysicalGroup(1, force_y, 0, 2, "force_y", &ierr);
    gmshModelAddPhysicalGroup(1, force_n, 0, 3, "force_n", &ierr);
    gmshModelAddPhysicalGroup(1, force_t, 0, 4, "force_t", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_x, 1, 5, "fix_x", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_y, 1, 6, "fix_y", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_n, 0, 7, "fix_n", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_t, 0, 8, "fix_t", &ierr);
    scale = mesh_size_factor; // set global variable
    gmshModelMeshSetSizeCallback(size_field_beam, NULL, &ierr);
    ierr = mesh_tri_quad(e_type);
    // gmshFltkRun(&ierr);

    return;
}
