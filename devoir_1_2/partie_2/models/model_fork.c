#include "models.h"
#include "utils_models.h"
#include <gmshc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RR_FLAG 1

// -----------------------------------------------------------------------------
// Tuning fork
// -----------------------------------------------------------------------------

static double scale;
static const double W = 0.030;
static const double HL = 0.0956;
static const double HR = 0.0956;
static const double HB = 0.040;
static const double DT = 0.008;
static const double DB = 0.007;
static const double RR = 0.0015 * RR_FLAG;

static const double _E = 210e9;
static const double _nu = 0.30;
static const double _rho = 7.85e3;
static const double _L = 1.0;

void set_physics_fork(double params[4], Model2D *type) {
    params[0] = _E;
    params[1] = _nu;
    params[2] = _rho;
    params[3] = _L;
    *type = PLANE_STRESS;
}

void set_bk_source_fork(double rho, const double xy[2], double f[2]) {
    double x = xy[0] * _L;
    double y = xy[1] * _L;
    f[0] = 0. * x * y;
    f[1] = 0.;
}

void set_bd_disp_fork(int entity, char kind, const double xy[2], double u[1]) {
    u[0] = 0.;
}

void set_bd_force_fork(int entity, char kind, const double xy[2], double f[2]) {
#if RR_FLAG
    if (88 == entity && kind == 'n') {
#else
    if (63 == entity && kind == 'n') {
#endif
        f[0] = 0.;
        f[1] = -100e3 * (xy[1] <= 0.75 * HR) * (0.5 * HR <= xy[1]);
#if RR_FLAG
    } else if (81 <= entity && entity <= 85 && kind == 'n') {
#else
    } else if (59 <= entity && entity <= 61 && kind == 'n') {
#endif
        f[0] = +100.e9;
        f[1] = 0.;
    } else {
        f[0] = 0.;
        f[1] = 0.;
    }
}

double size_field_fork(
    int dim, int tag, double x, double y, double z, double lc, void *data
) {
    double r = W / 2.;
    double R = r + DT;
    double dmin = DT / 100.;
    double dmax = DT / 2.;
    double dref = dmax - dmin;
    double d1, d2;
    if (y < 0.) {
        d1 = fmax(hypot(x, y) - r - dmin, 0.);
        if (DB * 0.51 < fabs(x)) {
            d2 = fmax(R - hypot(x, y) - dmin, 0.);
        } else {
            d2 = fmin(hypot(x + DB / 2., y + R), hypot(x - DB / 2., y + R));
            d2 = fmax(d2 - dmin, 0.);
        }
    } else {
        d1 = fmin(hypot(x - (-W / 2.), y), hypot(x - (W / 2.), y));
        d2 = fmin(hypot(x - (-W / 2. - DT), y), hypot(x - (W / 2. + DT), y));
        d1 = fmax(d1 - dmin, 0.);
        d2 = fmax(d2 - dmin, 0.);
    }

    double h1 = hermite(d1, dref, DT / 3., DT / 1.);
    double h2 = hermite(d2, dref, DT / 3., DT / 1.);
    return scale * fmin(h1, h2);
}

void mesh_fork(double mesh_size_factor, int e_type) {
    double r = W / 2.;
    double R = r + DT;
    double hmin = fmin(HL, HR);
    int ierr;

    double delta, tx, ty, tw, th;
    delta = sqrt(SQUARE(R + RR) - SQUARE(DB / 2. + RR)) - R;
    int objs[2], tools[4], main_tools[4];
    int tool, rect, disk, fork;
    objs[0] = tools[0] = tools[2] = 2;

    // Bottom cut tools
    for (int i = 0; i < 2; i++) {
        tx = DB / 2.;
        ty = -R - HB;
        tw = R;
        th = HB - delta;
        tx = (i == 0) ? tx : -tx - tw;
        tool = gmshModelOccAddRectangle(tx, ty, 0., tw, th, -1, 0., &ierr);
        objs[1] = tool;
        if (0. < RR) {
            tx = DB / 2. + RR;
            ty = -R - delta;
            tx = (i == 0) ? tx : -tx;
            disk =
                gmshModelOccAddDisk(tx, ty, 0., RR, RR, -1, 0, 0, 0, 0, &ierr);
            tools[1] = disk;
            gmshModelOccFuse(
                objs, 2, tools, 2, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
            );
        }
        tx = (DB / 2. + RR) * (R) / (R + RR);
        ty = -R - HB;
        tw = R;
        th = HB + R;
        tx = (i == 0) ? tx : -tx - tw;
        rect = gmshModelOccAddRectangle(tx, ty, 0., tw, th, -1, 0., &ierr);
        objs[1] = rect;
        tw = r + DT;
        disk = gmshModelOccAddDisk(0., 0., 0., tw, tw, -1, 0, 0, 0, 0, &ierr);
        tools[1] = disk;
        gmshModelOccCut(
            objs, 2, tools, 2, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
        );
        objs[1] = tool;
        tools[1] = rect;
        gmshModelOccFuse(
            objs, 2, tools, 2, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
        );
        main_tools[2 * i + 0] = 2;
        main_tools[2 * i + 1] = objs[1];
    }
    tx = -DT - W / 2.;
    ty = -r - DB - HB / 2.;
    tw = 2. * DT + W;
    th = HB / 2. + DB + r + hmin / 2.;
    objs[1] = gmshModelOccAddRectangle(tx, ty, 0., tw, th, -1, 0., &ierr);
    gmshModelOccCut(
        objs, 2, main_tools, 4, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
    );

    // Add full bottom branch
    tx = -DB / 2.;
    ty = -R - HB;
    double rr = fmin(RR, DB / 3.);
    rect = gmshModelOccAddRectangle(tx, ty, 0., DB, HB, -1, rr, &ierr);
    tools[1] = rect;
    gmshModelOccFuse(
        objs, 2, tools, 2, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
    );
    fork = objs[1];

    // Cut top branches
    rect = gmshModelOccAddRectangle(-W / 2., 0., 0., W, hmin, -1, 0., &ierr);
    objs[1] = rect;
    disk = gmshModelOccAddDisk(0., 0., 0., r, r, -1, 0, 0, 0, 0, &ierr);
    tools[1] = disk;
    gmshModelOccFuse(
        tools, 2, objs, 2, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
    );
    objs[1] = fork;
    gmshModelOccCut(
        objs, 2, tools, 2, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
    );

    // Add top branches
    tx = -W / 2. - DT;
    tools[1] = gmshModelOccAddRectangle(tx, 0., 0., DT, HL, -1, RR, &ierr);
    tools[3] = gmshModelOccAddRectangle(W / 2., 0., 0., DT, HR, -1, RR, &ierr);
    gmshModelOccFuse(
        objs, 2, tools, 4, NULL, NULL, NULL, NULL, NULL, -1, 1, 1, &ierr
    );

    gmshModelOccSynchronize(&ierr);

    int force_x[] = {};
    int force_y[] = {};
    int force_t[] = {};
    int dirichlet_n[] = {};
    int dirichlet_t[] = {};
    gmshModelAddPhysicalGroup(1, force_x, 0, 1, "force_x", &ierr);
    gmshModelAddPhysicalGroup(1, force_y, 0, 2, "force_y", &ierr);
    gmshModelAddPhysicalGroup(1, force_t, 0, 4, "force_t", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_n, 0, 7, "fix_n", &ierr);
    gmshModelAddPhysicalGroup(1, dirichlet_t, 0, 8, "fix_t", &ierr);

    if (0. < RR) {
        int force_n[] = {92, 88, 83, 82, 84, 81, 85};
        int dirichlet_x[] = {};
        int dirichlet_y[] = {};
        gmshModelAddPhysicalGroup(1, force_n, 7, 3, "force_n", &ierr);
        gmshModelAddPhysicalGroup(1, dirichlet_x, 0, 5, "fix_x", &ierr);
        gmshModelAddPhysicalGroup(1, dirichlet_y, 0, 6, "fix_y", &ierr);
    } else {
        int force_n[] = {63, 66, 59, 61};
        int dirichlet_x[] = {};
        int dirichlet_y[] = {};
        gmshModelAddPhysicalGroup(1, force_n, 4, 3, "force_n", &ierr);
        gmshModelAddPhysicalGroup(1, dirichlet_x, 0, 5, "fix_x", &ierr);
        gmshModelAddPhysicalGroup(1, dirichlet_y, 0, 6, "fix_y", &ierr);
    }

    scale = mesh_size_factor; // set global variable
    gmshOptionSetNumber("Mesh.MeshSizeExtendFromBoundary", 0, &ierr);

    gmshModelMeshSetSizeCallback(size_field_fork, NULL, &ierr);
    ierr = mesh_tri_quad(e_type);

    // gmshFltkRun(&ierr);
    // exit(EXIT_SUCCESS);

    return;
}
