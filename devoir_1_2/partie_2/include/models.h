#ifndef MODELS_H
#define MODELS_H

#include <stddef.h>
#define M_PI 3.14159265

typedef enum Model2D {
    PLANE_STRESS,
    PLANE_STRAIN,
    AXISYMMETRIC
} Model2D;

void set_physics_beam(double params[4], Model2D *type);
void set_bk_source_beam(double rho, const double xy[2], double f[2]);
void set_bd_disp_beam(int ent, char kind, const double xy[2], double u[1]);
void set_bd_force_beam(int ent, char kind, const double xy[2], double f[2]);
void mesh_beam(double mesh_size_factor, int e_type);

void set_physics_exam(double params[4], Model2D *type);
void set_bk_source_exam(double rho, const double xy[2], double f[2]);
void set_bd_disp_exam(int ent, char kind, const double xy[2], double u[1]);
void set_bd_force_exam(int ent, char kind, const double xy[2], double f[2]);
void mesh_exam(double mesh_size_factor, int e_type);

void set_physics_fork(double params[4], Model2D *type);
void set_bk_source_fork(double rho, const double xy[2], double f[2]);
void set_bd_disp_fork(int ent, char kind, const double xy[2], double u[1]);
void set_bd_force_fork(int ent, char kind, const double xy[2], double f[2]);
void mesh_fork(double mesh_size_factor, int e_type);

void set_physics_hole(double params[4], Model2D *type);
void set_bk_source_hole(double rho, const double xy[2], double f[2]);
void set_bd_disp_hole(int ent, char kind, const double xy[2], double u[1]);
void set_bd_force_hole(int ent, char kind, const double xy[2], double f[2]);
void mesh_hole(double mesh_size_factor, int e_type);

void set_physics_section(double params[4], Model2D *type);
void set_bk_source_section(double rho, const double xy[2], double f[2]);
void set_bd_disp_section(int ent, char kind, const double xy[2], double u[1]);
void set_bd_force_section(int ent, char kind, const double xy[2], double f[2]);
void mesh_section(double mesh_size_factor, int e_type);

void set_physics_tank_cut(double params[4], Model2D *type);
void set_bk_source_tank_cut(double rho, const double xy[2], double f[2]);
void set_bd_disp_tank_cut(int ent, char kind, const double xy[2], double u[1]);
void set_bd_force_tank_cut(int ent, char kind, const double xy[2], double f[2]);
void mesh_tank_cut(double mesh_size_factor, int e_type);

void set_physics_tank_axi(double params[4], Model2D *type);
void set_bk_source_tank_axi(double rho, const double xy[2], double f[2]);
void set_bd_disp_tank_axi(int ent, char kind, const double xy[2], double u[1]);
void set_bd_force_tank_axi(int ent, char kind, const double xy[2], double f[2]);
void mesh_tank_axi(double mesh_size_factor, int e_type);

#endif