#pragma once

/*
Une fonction qui tridiagonalise une matrice symétrique bande par transformations de similitudes.
Vous pouvez choisir le format de stockage et le type de transformations (réflexions de Householder
ou rotations de Givens).
*/
void tridiagonalize_full(double *A, int n, int k, double *d, double *e);

/*
Une fonction qui effectue une étape de l’algorithme QR avec un shift de Wilkinson µ sur une
matrice tridiagonale symétrique.
*/
int step_qr_tridiag(double *d, double *e, int m, double eps);

/*
Une fonction qui calcule l’entièreté du spectre d’une matrice bande symétrique en faisant appel à
vos deux fonctions précedentes.
*/
int qr_eigs_full(double *A, int n, int k, double eps, int max_iter);