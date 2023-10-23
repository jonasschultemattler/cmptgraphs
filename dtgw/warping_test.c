#if !defined(__clang__) && defined(__GNUC__) && defined(__GNUC_MINOR__)
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)
/* enable auto-vectorizer */
#pragma GCC optimize("tree-vectorize")
 // float associativity required to vectorize reductions 
#pragma GCC optimize("unsafe-math-optimizations")
/* maybe 5% gain, manual unrolling with more accumulators would be better */
#pragma GCC optimize("unroll-loops")
#endif
#endif

#include <math.h>
#include <stdbool.h>
#include "lap.h"
// #include "lapjv.h"
#include <stdio.h>

#ifdef __cplusplus
/* extern "C" is used for defining C code when compiling with a C++ compiler.
   This is necessary since ctypes needs to call a C function. */
extern "C" {
#endif


static inline double min2d(double a, double b) {
    if (a < b)
        return a;
    else
        return b;
}

static inline double max2d(double a, double b) {
    if (a > b)
        return a;
    else
        return b;
}

static inline double min3d(double a, double b, double c) {
    if (a <= b) {
        if (a <= c)
            return a;
        else
            return c;
    }
    else {
        if (b <= c)
            return b;
        else
            return c;
    }
}

static inline int argmin3d(double a, double b, double c) {
    if (a <= b) {
        if (a <= c)
            return 0;
        else
            return 2;
    }
    else {
        if (b <= c)
            return 1;
        else
            return 2;
    }
}

static inline double city_block_distance3d(const double *u, const double *v, int n) {
    double s = 0.0;
    s += fabs(u[0] - v[0]);
    s += fabs(u[1] - v[1]);
    s += fabs(u[2] - v[2]);
    return s;
}

static inline double city_block_distance(const double *u, const double *v, int n) {
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        s += fabs(u[i] - v[i]);
    }
    return s;
}

// static inline void metric(double *f1, double *f2, int t1, int t2, int columns, double *res) {
//     int i, j;
//     for(i=0; i < t1; i++) {
//         const double *u_i = f1 + i*columns;
//         for(j=0; j < t2; j++) {
//             const double *v_j = f2 + j*columns;
//             res[i*t1 + j] = city_block_distance(u_i, v_j, columns);
//         }
//     }
// }



double lapjv(int n, double *assigncost, int *rowsol, int *colsol) {
    double *rowcost = (double *) malloc(n * sizeof(double));
    double *colcost = (double *) malloc(n * sizeof(double));
    double cost = lap(n, assigncost, rowsol, colsol, rowcost, colcost);
    free(rowcost);
    free(colcost);
    return cost;
}

void update_warping2(const double *f1, const double *f2, int t1, int t2, int n, int columns,
    int* rowsol, int*colsol, int *region, double *warpcost)
{
    int i, j, u, v;
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[i+t1]; j++) {
            warpcost[i*t2 + j] = 0;
        }
    }
    for(u=0; u < n; u++) {
        v = colsol[rowsol[u]];
        // printf("(%d,%d) ", u, v);
        for(i=0; i < t1; i++) {
            const double *u_i = f1 + i*n*columns + u*columns;
            for(j=region[i]; j < region[i+t1]; j++) {
                const double *v_j = f2 + j*n*columns + v*columns;
                warpcost[i*t2 + j] += city_block_distance(u_i, v_j, columns);
            }
        }
    }
}


void update_warping(const double *f1, const double *f2, int t1, int t2, int columns,
    int* matching, int n, int *region, double *warpcost)
{
    int u, v, i, j;
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[i+t1]; j++) {
            warpcost[i*t2 + j] = 0;
        }
    }
    for(u=0; u < n; u++) {
        v = matching[u];
        for(i=0; i < t1; i++) {
            const double *u_i = f1 + i*n*columns + u*columns;
            for(j=region[i]; j < region[i+t1]; j++) {
                const double *v_j = f2 + j*n*columns + v*columns;
                warpcost[i*t2 + j] += city_block_distance(u_i, v_j, columns);
            }
        }
    }
}

void update_matchcost(const double *f1, const double *f2, int n, int columns,
    int *warppath, int l, double *matchcost)
{
    int t, i, j;
    for(i = 0; i < n*n; i++)
        matchcost[i] = 0;
    for(t=0; t < 2*l; t += 2) {
        for(i=0; i < n; i++) {
            const double *u_i = f1 + warppath[t]*n*columns + i*columns;
            for(j=0; j < n; j++) {
                const double *v_j = f2 + warppath[t+1]*n*columns + j*columns;
                matchcost[i*n + j] += city_block_distance(u_i, v_j, columns);
            }
        }
    }
}

void sakoe_chiba_band(int t1, int t2, int window, int* region)
{
    double scale = (double)(t2-1)/(double) (t1-1);
    int h_shift, v_shift;

    if (t2 > t1) {
        h_shift = 0;
        v_shift = max2d(window, scale/2);
    }
    else if (t1 > t2) {
        h_shift = max2d(window, 0.5/scale);
        v_shift = 0;
    }
    else {
        h_shift = 0;
        v_shift = window;
    }

    for(int i = 0; i < t1; i++) {
        region[i] = min2d(max2d(ceil(scale*(i - h_shift) - v_shift), 0), t2);
        region[i+t1] = min2d(max2d(floor(scale*(i + h_shift) + v_shift) + 1, 0), t2);
    }
}

int dynamic_timewarping(double* cost_matrix, int t1, int t2, int* region, int* path, double* res) {
    // cost_matrix t1 x t2
    // region 2 x t1
    // path t1*t2 x 2
    // res t1+1 x t2+1
    int i, j;
    res[(t1+1)*(t2+1)-1] = 0.0;
    for(i=t1-1; i >= 0; i--) {
        for(j=region[i+t1]-1; j >= region[i]; j--) {
            res[i*(t2+1) + j] = cost_matrix[i*t2 + j]
            + min3d(res[(i+1)*(t2+1) + j+1], res[i*(t2+1) + j+1], res[(i+1)*(t2+1) + j]);
        }
    }
    const int x_step[3] = {1, 1, 0};
    const int y_step[3] = {1, 0, 1};
    int direction;
    path[0] = path[1] = 0;
    int l = 2;
    i = j = 0;
    while (i < t1-1 || j < t2-1) {
        direction = argmin3d(res[(i+1)*(t2+1) + j+1], res[(i+1)*(t2+1) + j], res[i*(t2+1) + j+1]);
        i += x_step[direction];
        j += y_step[direction];
        path[l] = i;
        path[l+1] = j;
        l += 2;
    }
    return l/2;
}



int shortest_warp_path(int n, int m, int *path) {
    int i, j, l, a, b;
    i = 0;
    j = 0;
    l = 1;
    path[0] = path[1] = 0;
    while (i < n && j < m) {
        a = (i+1)*m;
        b = (j+1)*n;
        if (a < b + n)
            i++;
        if (b < a + m)
            j++;
        path[2*l] = i;
        path[2*l+1] = j;
        l++;
    }
    return l-1;
}

int init_diagonal_warping(const double *f1, const double *f2, int t1, int t2, int n,
    int columns, int *warppath, double *res)
{
    int l;
    l = shortest_warp_path(t1, t2, warppath);
    update_matchcost(f1, f2, n, columns, warppath, l, res);
    return l;
}

void init_opt_matching(const double *f1, const double *f2, int t1, int t2, int n,
    int columns, int *region, double *res)
{
    int i, j, u, v;
    double * tmp = (double *) malloc(n*n * sizeof(double));
    for(i=0; i< n*n; i++)
        res[i] = 0;
    for(i=0; i < t1; i++) {
        for(u=0; u < n; u++) {
            const double *u_i = f1 + i*n*columns + u*columns;
            for(v=0; v < n; v++) {
                const double *v_j = f2 + region[i]*n*columns + v*columns;
                tmp[u*n + v] = city_block_distance(u_i, v_j, columns);
            }
        }
        for(j=region[i]+1; j < region[t1+i]; j++) {
            for(u=0; u < n; u++) {
                const double *u_i = f1 + i*n*columns + u*columns;
                for(v=0; v < n; v++) {
                    const double *v_j = f2 + j*n*columns + v*columns;
                    tmp[u*n + v] = min2d(tmp[u*n + v], city_block_distance(u_i, v_j, columns));
                }
            }
        }
        for(u=0; u < n; u++) {
            for(v=0; v < n; v++) {
                res[u*n + v] += tmp[u*n + v];
            }
        }
    }
    free(tmp);
}

void init_opt_warping(const double *f1, const double *f2, int t1, int t2, int n,
    int columns, int *region, double *res)
{
    int *rowsol = (int *) malloc(n * sizeof(int));
    int *colsol = (int *) malloc(n * sizeof(int));
    double *rowcost = (double *) malloc(n * sizeof(double));
    double *colcost = (double *) malloc(n * sizeof(double));
    double *tmp = (double *) malloc(n*n * sizeof(double));
    int i, j, u, v;
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[t1+i]; j++) {
            // metric(f1 + i*n, f2 + j*n, n, n, columns, tmp);
            for(u=0; u < n; u++) {
                const double *u_i = f1 + i*n*columns + u*columns;
                for(v=0; v < n; v++) {
                    const double *v_j = f2 + j*n*columns + v*columns;
                    tmp[u*n + v] = city_block_distance(u_i, v_j, columns);
                }
            }
            res[i*t1 + j] = lap(n, tmp, rowsol, colsol, rowcost, colcost);
        }
    }
    free(rowsol);
    free(colsol);
    free(rowcost);
    free(colcost);
    free(tmp);
}

int init_product_warping(const double *f1, const double *f2, int t1, int t2, int n,
    int columns, int *region, int *warppath, double *res)
{
    int i, j, k, l;
    k = 0;
    for(i=0; i < t1; i++) {
        l = region[i+t1] - region[i];
        for(j=0; j < l; j++) {
            warppath[k+2*j] = i;
            warppath[k+2*j+1] = region[i]+j;
        }
        k += 2*l;
    }
    update_matchcost(f1, f2, n, columns, warppath, k/2, res);
    return k/2;
}


void vertex_feature_labels(bool* tlabels, int n, int T, double* feature)
// double feature t x n x 1
// boolean tlabels t x n
{
    int u, t;
    for(u=0; u < n; u++) {
        for(t=0; t < T; t++) {
            feature[t*n + u] = (double) tlabels[t*n + u];
        }
    }
}


void vertex_feature_neighbors_normed(bool* tadj, bool* tlabels, int n, int T, double* feature)
// double feature t x n x 3
// boolean tadj t x n x n
// boolean tlabels t x n
{
    int u, v, t, i, j;
    int deg, max_deg, red, max_red;
    max_deg = max_red = 0;
    for(u=0; u < n; u++) {
        i = 0;
        j = u;
        for(t=0; t < T; t++) {
            feature[j*3] = (double) tlabels[j];
            deg = red = 0;
            // todo optimize
            for(v=0; v < n; v++) {
                deg += tadj[j*n+v];
                red += tadj[j*n+v]*tlabels[i + v];
            }
            if (deg > max_deg)
                max_deg = deg;
            if (red > max_red)
                max_red = red;
            feature[j*3+1] = (double) deg;
            feature[j*3+2] = (double) red;
            i += n;
            j = i + u;
        }
    }
    if (max_deg > 0) {
        for(u=0; u < n; u++)
            for(t=0; t < T; t++)
                feature[t*n*3+u*3+1] /= max_deg;
    }
    if (max_red > 0) {
        for(u=0; u < n; u++)
            for(t=0; t < T; t++)
                feature[t*n*3+u*3+2] /= max_red;
    }
}

void vertex_feature_neighbors(bool* tadj, bool* tlabels, int n, int T, double* feature)
// double feature t x n x 3
// boolean tadj t x n x n
// boolean tlabels t x n
{
    int u, v, t, i, j;
    int deg, red;
    for(u=0; u < n; u++) {
        i = 0;
        j = u;
        for(t=0; t < T; t++) {
            feature[j*3] = (double) tlabels[j];
            deg = red = 0;
            for(v=0; v < n; v++) {
                deg += tadj[j*n+v];
                red += tadj[j*n+v]*tlabels[i + v];
            }
            feature[j*3+1] = (double) deg;
            feature[j*3+2] = (double) red;
            i += n;
            j = i + u;
        }
    }
}


enum algostate {
    MATCHCOST_UPDATED = 1,
    WARPCOST_UPDATED = 2,
    CONVERGED = 3,
    ITERATION_LIMIT_REACHED = 4,
};

enum initialize {
    DIAGONAL_WARPING = 1,
    OPTIMISTIC_WARPING = 2,
    OPTIMISTIC_MATCHING = 3,
    PRODUCT_WARPING = 4
};

bool terminated(int state) {
    return (state == CONVERGED || state == ITERATION_LIMIT_REACHED);
}


double dtgw(const double *f1, const double *f2, int t1, int t2, int n, int columns, int init, int window, int max_iterations)
{
    int *warppath = (int *) malloc(t1*t2 * sizeof(int));
    double *warpcost = (double *) malloc(t1*t2 * sizeof(double));
    double *res = (double *) malloc((t1+1)*(t2+1) * sizeof(double));
    for(int i = 0; i < (t1+1)*(t2+1); i++)
        res[i] = INFINITY;
    double *matchcost = (double *) malloc(n*n * sizeof(double));
    int *rowsol = (int *) malloc(n * sizeof(int));
    int *colsol = (int *) malloc(n * sizeof(int));
    double *rowcost = (double *) malloc(n * sizeof(double));
    double *colcost = (double *) malloc(n * sizeof(double));
    int *region = (int *) malloc(t1*2 * sizeof(int));
    int state, l;
    double (*metric) (const double*, const double*, int) = city_block_distance;

    if(columns == 3)
        metric = city_block_distance3d;

    sakoe_chiba_band(t1, t2, window, region);

    if (init == DIAGONAL_WARPING) {
        l = init_diagonal_warping(f1, f2, t1, t2, n, columns, warppath, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else if (init == OPTIMISTIC_MATCHING) {
        init_opt_matching(f1, f2, t1, t2, n, columns, region, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else if (init == PRODUCT_WARPING) {
        l = init_product_warping(f1, f2, t1, t2, n, columns, region, warppath, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else {
        init_opt_warping(f1, f2, t1, t2, n, columns, region, warpcost);
        state = WARPCOST_UPDATED;
    }

    double cost = INFINITY;
    double new_cost;
    int iterations = 0;
    while (!terminated(state)) {
        // printf("iter: %d state: %d cost %.1f\n", iterations, state, cost);
        if (state == MATCHCOST_UPDATED) {
            new_cost = lap(n, matchcost, rowsol, colsol, rowcost, colcost);
            if(new_cost >= cost)
                state = CONVERGED;
            else {
                update_warping2(f1, f2, t1, t2, n, columns, rowsol, colsol, region, warpcost);
                state = WARPCOST_UPDATED;
            }
        }
        else {
            l = dynamic_timewarping(warpcost, t1, t2, region, warppath, res);
            new_cost = res[0];
            if(new_cost >= cost)
                state = CONVERGED;
            else {
                update_matchcost(f1, f2, n, columns, warppath, l, matchcost);
                state = MATCHCOST_UPDATED;
            }
        }
        cost = new_cost;
        iterations++;
        if(iterations >= max_iterations)
            state = ITERATION_LIMIT_REACHED;
    }
    // printf("iter: %d state: %d cost %.1f\n", iterations, state, cost);

    free(warppath);
    free(warpcost);
    free(matchcost);
    free(rowsol);
    free(colsol);
    free(rowcost);
    free(colcost);
    free(region);
    free(res);

    return cost;
}


#ifdef __cplusplus
}
#endif

