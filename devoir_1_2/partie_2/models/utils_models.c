#include "utils_models.h"
#include <gmshc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _TRI 2
#define _QUAD 3

/**
 * @brief Compute a smooth transition between two mesh sizes
 * @param d Distance from a reference object
 * @param d_tr Transition distance
 * @param h1 Mesh size before transition
 * @param h2 Mesh size after transition
 * @return double Mesh size
 */
double hermite(double d, double d_tr, double h1, double h2) {
    double h;
    if (d < d_tr) {
        h = h1 + (h2 - h1) * (d / d_tr) * (d / d_tr) * (3.0 - 2.0 * d / d_tr);
    } else {
        h = h2;
    }
    return h;
}

int mesh_tri_quad(int e_type) {
    int ierr;
    if (e_type == _TRI) {
        gmshModelMeshGenerate(2, &ierr);
    } else if (e_type == _QUAD) {
        // gmshOptionSetNumber("Mesh.Algorithm", 11, &ierr);
        gmshOptionSetNumber("Mesh.Algorithm", 8, &ierr);
        gmshOptionSetNumber("Mesh.RecombinationAlgorithm", 2, &ierr);
        gmshOptionSetNumber("Mesh.RecombineAll", 1, &ierr);
        gmshModelMeshGenerate(2, &ierr);
    } else {
        printf("Unknown element type: %d\n", e_type);
        exit(EXIT_FAILURE);
    }
    return ierr;
}

void compute_stress_tank(
    double Ri, double Ro, double Pi, double Po, double nu
) {
    double rs[] = {Ri, (Ri + Ro) * 0.5, Ro};
    double sig_rr, sig_tt, sig_zz, r;
    for (int i = 0; i < 3; i++) {
        r = rs[i];
        double r2 = SQUARE(r);
        double Ri2 = SQUARE(Ri);
        double Ro2 = SQUARE(Ro);
        double dP = Pi - Po;
        double den = Ro2 - Ri2;
        sig_rr = (Pi * Ri2 - Po * Ro2 - dP * (Ro2 * Ri2) / r2) / den;
        sig_tt = (Pi * Ri2 - Po * Ro2 + dP * (Ro2 * Ri2) / r2) / den;
        sig_zz = nu * (sig_rr + sig_tt);
        printf("Pressure tank\n");
        printf("  sig_rr(r = %5.3lf) = %12.3le\n", r, sig_rr);
        printf("  sig_tt(r = %5.3lf) = %12.3le\n", r, sig_tt);
        printf("  sig_zz(r = %5.3lf) = %12.3le\n\n", r, sig_zz);
    }
}
