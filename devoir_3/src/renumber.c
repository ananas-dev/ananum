#include "renumber.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))


/**
 * @brief Compute the band size of a linear system from a mesh
 * @param ne Number of elements
 * @param nl Number of nodes per element
 * @param elem Element nodes (1-based indexing)
 * @param idx_map Mapping between original and optimized indices
 * @return size_t Maximum neighborhood distance
 */
size_t compute_band(
    const size_t ne, const size_t nl, const size_t *elem, const size_t *idx_map
) {
    size_t idx, imax, imin;
    size_t band = 0;
    size_t s = 1; // 1-based indexing
    idx = 0;
    for (size_t e = 0; e < ne; e++) {
        imin = imax = idx_map[elem[idx + 0] - s];
        for (size_t i = 1; i < nl; i++) {
            imin = MIN(imin, idx_map[elem[idx + i] - s]);
            imax = MAX(imax, idx_map[elem[idx + i] - s]);
        }
        band = MAX(band, imax - imin);
        idx += nl;
    }
    return band;
}


double *cmp_XY;
int compare_node_x(const void *n1, const void *n2) {
    return (cmp_XY[2 * (*(size_t *)n1) + 0] > cmp_XY[2 * (*(size_t *)n2) + 0]) -
           (cmp_XY[2 * (*(size_t *)n1) + 0] < cmp_XY[2 * (*(size_t *)n2) + 0]);
}
int compare_node_y(const void *n1, const void *n2) {
    return (cmp_XY[2 * (*(size_t *)n1) + 1] > cmp_XY[2 * (*(size_t *)n2) + 1]) -
           (cmp_XY[2 * (*(size_t *)n1) + 1] < cmp_XY[2 * (*(size_t *)n2) + 1]);
}

void renumber(
    size_t n_elem,
    size_t n_loc,
    size_t *elems,
    size_t n_bd_node,
    size_t *bd_nodes,
    size_t n_node,
    double *coords,
    size_t **idx_map,
    int kind
) {
    *idx_map = malloc(n_node * sizeof(size_t));
    if (kind == 0) {
        for (size_t i = 0; i < n_node; i++) {
            (*idx_map)[i] = i;
        }
    } else if (kind == 1 || kind == 2) {
        cmp_XY = coords;
        size_t *indices = malloc(n_node * sizeof(size_t));
        for (size_t i = 0; i < n_node; i++) {
            indices[i] = i;
        }
        if (kind == 1) {
            qsort(indices, n_node, sizeof(size_t), compare_node_x);
        } else {
            qsort(indices, n_node, sizeof(size_t), compare_node_y);
        }
        for (size_t i = 0; i < n_node; i++) {
            (*idx_map)[indices[i]] = i;
        }
        free(indices);
    } else {
        fprintf(stderr, "Unknown renumbering kind %d\n", kind);
        exit(1);
    }
}
