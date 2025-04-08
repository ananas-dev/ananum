#ifndef RENUMBER_H
#define RENUMBER_H
#include <stddef.h>

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
);

size_t compute_band(
    const size_t ne, const size_t nl, const size_t *elem, const size_t *idx_map
);

#endif