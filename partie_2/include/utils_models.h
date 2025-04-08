#ifndef MODELS_UTILS_H
#define MODELS_UTILS_H

#define SQUARE(X) ((X) * (X))

int mesh_tri_quad(int e_type);
double hermite(double d, double d_tr, double h1, double h2);
void compute_stress_tank(double Ri, double Ro, double Pi, double Po, double nu);

#endif