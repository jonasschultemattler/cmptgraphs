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
#include <stdio.h>
#include <stdlib.h>

#include "lapjv.h"

#ifdef __cplusplus
 // extern "C" is used for defining C code when compiling with a C++ compiler.
 //   This is necessary since ctypes needs to call a C function. 
extern "C" {
#endif

static inline float min2d(float a, float b) {
    if (a < b)
        return a;
    else
        return b;
}

static inline float max2d(float a, float b) {
    if (a > b)
        return a;
    else
        return b;
}

static inline float min3d(float a, float b, float c) {
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

static inline int argmin3d(float a, float b, float c) {
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

typedef float metric_function(const float*, const float*, int);


static inline float city_block_distance1d(const float *u, const float *v, int n) {
    return fabs(u[0] - v[0]);
}

static inline float city_block_distance2d(const float *u, const float *v, int n) {
    return fabs(u[0] - v[0]) + fabs(u[1] - v[1]);
}

static inline float city_block_distance3d(const float *u, const float *v, int n) {
    float s = 0.0;
    s += fabs(u[0] - v[0]);
    s += fabs(u[1] - v[1]);
    s += fabs(u[2] - v[2]);
    return s;
}

static inline float city_block_distance(const float *u, const float *v, int n) {
    float s = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        s += fabs(u[i] - v[i]);
    }
    return s;
}

static inline float euclidean_distance(const float *u, const float *v, int n) {
    float s = 0.0;
    float tmp;
    int i;
    for (i = 0; i < n; i++) {
        tmp = u[i] - v[i];
        s += tmp*tmp;
    }
    // return sqrt(s);
    return s;
}


static inline float negative_dot_product(const float *u, const float *v, int n) {
    float s = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        s += u[i]*v[i];
    }
    return -s;
}


enum metric {
    L1 = 1,
    L2 = 2,
    DOT = 3,
};


static metric_function* get_metric(int metric_number, int columns) {
    if(metric_number == L1) {
        if(columns == 1)
            return city_block_distance1d;
        if(columns == 2)
            return city_block_distance2d;
        if(columns == 3)
            return city_block_distance3d;
        return city_block_distance;
    }
    if(metric_number == L2)
        return euclidean_distance;
    if(metric_number == DOT)
        return negative_dot_product;
    return NULL;
}


void update_warping(const float *f1, const float *f2, int t1, int t2, int n, int columns,
    float (*metric)(const float*, const float*, int),
    int* rowsol, int*colsol, int *region, float *warpcost)
{
    int i, j, u, v;
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[i+t1]; j++) {
            warpcost[i*t2 + j] = 0.0;
        }
    }
    for(u=0; u < n; u++) {
        v = rowsol[u];
        // printf("(%d,%d) ", u, v);
        // printf("%d ", rowsol[u]);
        // printf("%d  ", colsol[u]);
        for(i=0; i < t1; i++) {
            const float *u_i = f1 + i*n*columns + u*columns;
            for(j=region[i]; j < region[i+t1]; j++) {
                const float *v_j = f2 + j*n*columns + v*columns;
                warpcost[i*t2 + j] += metric(u_i, v_j, columns);
            }
        }
    }
    // printf("\n");
}

void update_warping2(const float *f1, const float *f2, int t1, int t2, int n, int columns,
    int* matching, int *region, float *warpcost, int m)
{
    int i, j, u, v;
    metric_function* metric = get_metric(m, columns);
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[i+t1]; j++) {
            warpcost[i*t2 + j] = 0.0;
        }
    }
    for(u=0; u < n; u++) {
        v = matching[u];
        for(i=0; i < t1; i++) {
            const float *u_i = f1 + i*n*columns + u*columns;
            for(j=region[i]; j < region[i+t1]; j++) {
                const float *v_j = f2 + j*n*columns + v*columns;
                warpcost[i*t2 + j] += metric(u_i, v_j, columns);
            }
        }
    }
}

void update_warping3(const float *f1, const float *f2, int t1, int t2, int n, int m, int columns,
    int* matching, int *region, float *warpcost, int nmetric)
{
    int i, j, u, v;
    metric_function* metric = get_metric(nmetric, columns);
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[i+t1]; j++) {
            warpcost[i*t2 + j] = 0.0;
        }
    }
    for(u=0; u < n; u++) {
        v = matching[u];
        if(v < m) {
            for(i=0; i < t1; i++) {
                const float *u_i = f1 + i*n*columns + u*columns;
                for(j=region[i]; j < region[i+t1]; j++) {
                    const float *v_j = f2 + j*m*columns + v*columns;
                    warpcost[i*t2 + j] += metric(u_i, v_j, columns);
                }
            }
        }
    }
}


void update_warping4(const float *f1, const float *f2, const float *zeros, int t1, int t2, int n, int m, int columns,
    int* matching, int *region, float *warpcost, int nmetric)
{
    int i, j, u, v;
    float c_d;
    metric_function* metric = get_metric(nmetric, columns);
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[i+t1]; j++) {
            warpcost[i*t2 + j] = 0.0;
        }
    }
    for(u=0; u < n; u++) {
        v = matching[u];
        if(v < m) {
            for(i=0; i < t1; i++) {
                const float *u_i = f1 + i*n*columns + u*columns;
                for(j=region[i]; j < region[i+t1]; j++) {
                    const float *v_j = f2 + j*m*columns + v*columns;
                    warpcost[i*t2 + j] += metric(u_i, v_j, columns);
                }
            }
        }
        else {
            for(i=0; i < t1; i++) {
                const float *u_i = f1 + i*n*columns + u*columns;
                c_d = metric(u_i, zeros, columns);
                for(j=region[i]; j < region[i+t1]; j++) {
                    warpcost[i*t2 + j] += c_d;
                }
            }
        }
    }
}


void update_warping5(const float *f1, const float *f2, const float *zeros, int t1, int t2, int n, int m, int columns,
    int* matching, int *region, float *warpcost, int nmetric)
{
    int i, j, u, v;
    float c_d;
    metric_function* metric = get_metric(nmetric, columns);
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[i+t1]; j++) {
            warpcost[i*t2 + j] = 0.0;
        }
    }
    for(u=0; u < n; u++) {
        v = matching[u];
        if(v < m) {
            for(i=0; i < t1; i++) {
                const float *u_i = f1 + i*n*columns + u*columns;
                for(j=region[i]; j < region[i+t1]; j++) {
                    const float *v_j = f2 + j*m*columns + v*columns;
                    warpcost[i*t2 + j] += metric(u_i, v_j, columns);
                }
            }
        }
        else {
            for(i=0; i < t1; i++) {
                const float *u_i = f1 + i*n*columns + u*columns;
                c_d = metric(u_i, zeros, columns);
                for(j=region[i]; j < region[i+t1]; j++) {
                    warpcost[i*t2 + j] += c_d/(n-m);
                }
            }
        }
    }
}


void update_matchcost(const float *f1, const float *f2, int n, int columns,
    float (*metric)(const float*, const float*, int),
    int *warppath, int l, float *matchcost)
{
    int t, i, j;
    for(i = 0; i < n*n; i++)
        matchcost[i] = 0.0;
    for(t=0; t < 2*l; t += 2) {
        for(i=0; i < n; i++) {
            const float *u_i = f1 + warppath[t]*n*columns + i*columns;
            for(j=0; j < n; j++) {
                const float *v_j = f2 + warppath[t+1]*n*columns + j*columns;
                matchcost[i*n + j] += metric(u_i, v_j, columns);
            }
        }
    }
}


void update_matchcost2(const float *f1, const float *f2, int n, int columns,
    int *warppath, int l, float *matchcost, int m)
{
    int t, i, j;
    metric_function* metric = get_metric(m, columns);
    for(i = 0; i < n*n; i++)
        matchcost[i] = 0.0;
    for(t=0; t < 2*l; t += 2) {
        for(i=0; i < n; i++) {
            const float *u_i = f1 + warppath[t]*n*columns + i*columns;
            for(j=0; j < n; j++) {
                const float *v_j = f2 + warppath[t+1]*n*columns + j*columns;
                matchcost[i*n + j] += metric(u_i, v_j, columns);
            }
        }
    }
}


void update_matchcost3(const float *f1, const float *f2, int n, int m, int columns,
    int *warppath, int l, float *matchcost, int nmetric)
/*
f1 n x c
f2 m x c
n >= m
matchcost n x n
*/
{
    int t, i, j;
    metric_function* metric = get_metric(nmetric, columns);
    // for(i = 0; i < n*n; i++)
    //     matchcost[i] = 0.0;
    for(t=0; t < 2*l; t += 2) {
        for(i=0; i < n; i++) {
            const float *u_i = f1 + warppath[t]*n*columns + i*columns;
            for(j=0; j < m; j++) {
                const float *v_j = f2 + warppath[t+1]*m*columns + j*columns;
                matchcost[i*n + j] += metric(u_i, v_j, columns);
            }
        }
    }
}



void update_matchcost4(const float *f1, const float *f2, const float *zeros, int n, int m, int columns,
    int *warppath, int l, float *matchcost, int nmetric)
/*
f1 n x c
f2 m x c
n >= m
matchcost n x n
*/
{
    int t, i, j;
    float c_d;
    metric_function* metric = get_metric(nmetric, columns);
    // for(i = 0; i < n*n; i++)
    //     matchcost[i] = 0.0;
    for(t=0; t < 2*l; t += 2) {
        for(i=0; i < n; i++) {
            const float *u_i = f1 + warppath[t]*n*columns + i*columns;
            for(j=0; j < m; j++) {
                const float *v_j = f2 + warppath[t+1]*m*columns + j*columns;
                matchcost[i*n + j] += metric(u_i, v_j, columns);
            }
            c_d = metric(u_i, zeros, columns);
            for(j=m; j < n; j++) {
                matchcost[i*n + j] += c_d;
            }
        }
    }
}


void update_matchcost5(const float *f1, const float *f2, const float *zeros, int n, int m, int columns,
    int *warppath, int l, float *matchcost, int nmetric)
/*
f1 n x c
f2 m x c
n > m
matchcost n x n
*/
{
    int t, i, j;
    float c_d;
    metric_function* metric = get_metric(nmetric, columns);
    // for(i = 0; i < n*n; i++)
    //     matchcost[i] = 0.0;
    for(t=0; t < 2*l; t += 2) {
        for(i=0; i < n; i++) {
            const float *u_i = f1 + warppath[t]*n*columns + i*columns;
            for(j=0; j < m; j++) {
                const float *v_j = f2 + warppath[t+1]*m*columns + j*columns;
                matchcost[i*n + j] += metric(u_i, v_j, columns);
            }
            c_d = metric(u_i, zeros, columns);
            for(j=m; j < n; j++) {
                matchcost[i*n + j] += c_d/(n-m);
            }
        }
    }
}


void sakoe_chiba_band(int t1, int t2, int window, int* region)
{
    float scale = (float)(t2-1)/(float) (t1-1);
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

int dynamic_timewarping(float* cost_matrix, int t1, int t2, int* region, int* path, float* res) {
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

int init_diagonal_warping(const float *f1, const float *f2, int t1, int t2, int n, int columns,
    float (*metric)(const float*, const float*, int),
    int *warppath, float *res)
{
    int l = shortest_warp_path(t1, t2, warppath);
    update_matchcost(f1, f2, n, columns, metric, warppath, l, res);
    return l;
}


int init_diagonal_warping2(const float *f1, const float *f2, int t1, int t2, int n, int columns,
    int *warppath, float *res, int m)
{
    int l = shortest_warp_path(t1, t2, warppath);
    update_matchcost2(f1, f2, n, columns, warppath, l, res, m);
    return l;
}


int init_diagonal_warping3(const float *f1, const float *f2, int t1, int t2, int n, int m, int columns,
    int *warppath, float *res, int metric)
{
    int l = shortest_warp_path(t1, t2, warppath);
    update_matchcost3(f1, f2, n, m, columns, warppath, l, res, metric);
    return l;
}

int init_diagonal_warping4(const float *f1, const float *f2, const float *zeros, int t1, int t2, int n, int m, int columns,
    int *warppath, float *res, int metric)
{
    int l = shortest_warp_path(t1, t2, warppath);
    update_matchcost4(f1, f2, zeros, n, m, columns, warppath, l, res, metric);
    return l;
}

int init_diagonal_warping5(const float *f1, const float *f2, const float *zeros, int t1, int t2, int n, int m, int columns,
    int *warppath, float *res, int metric)
{
    int l = shortest_warp_path(t1, t2, warppath);
    update_matchcost5(f1, f2, zeros, n, m, columns, warppath, l, res, metric);
    return l;
}


void init_opt_matching(const float *f1, const float *f2, int t1, int t2, int n, int columns,
    float (*metric)(const float*, const float*, int),
    int *region, float *res)
{
    int i, j, u, v;
    float * tmp = (float *) malloc(n*n * sizeof(float));
    for(i=0; i< n*n; i++)
        res[i] = 0;
    for(i=0; i < t1; i++) {
        for(u=0; u < n; u++) {
            const float *u_i = f1 + i*n*columns + u*columns;
            for(v=0; v < n; v++) {
                const float *v_j = f2 + region[i]*n*columns + v*columns;
                tmp[u*n + v] = metric(u_i, v_j, columns);
            }
        }
        for(j=region[i]; j < region[t1+i]; j++) {
            for(u=0; u < n; u++) {
                const float *u_i = f1 + i*n*columns + u*columns;
                for(v=0; v < n; v++) {
                    const float *v_j = f2 + j*n*columns + v*columns;
                    tmp[u*n + v] = min2d(tmp[u*n + v], metric(u_i, v_j, columns));
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

void init_opt_warping(const float *f1, const float *f2, int t1, int t2, int n, int columns,
    float (*metric)(const float*, const float*, int),
    int *region, float *warpcost, int *rowsol, int *colsol, float *rowcost, float *colcost)
{
    float *tmp = (float *) malloc(n*n * sizeof(float));
    int i, j, u, v;
    for(i = 0; i < t1*t2; i++)
        warpcost[i] = INFINITY;
    for(i=0; i < t1; i++) {
        for(j=region[i]; j < region[t1+i]; j++) {
            for(u=0; u < n; u++) {
                const float *u_i = f1 + i*n*columns + u*columns;
                for(v=0; v < n; v++) {
                    const float *v_j = f2 + j*n*columns + v*columns;
                    tmp[u*n + v] = metric(u_i, v_j, columns);
                }
            }
            // warpcost[i*t2 + j] = lap(n, tmp, rowsol, colsol, rowcost, colcost);
            warpcost[i*t2 + j] = lap<1, int, float>(n, tmp, 0, rowsol, colsol, rowcost, colcost);
        }   
    }
    free(tmp);
}

int init_product_warping(const float *f1, const float *f2, int t1, int t2, int n, int columns,
    float (*metric)(const float*, const float*, int),
    int *region, int *warppath, float *res)
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
    update_matchcost(f1, f2, n, columns, metric, warppath, k/2, res);
    return k/2;
}

void vertex_feature_subtrees0(bool* tadj, bool* tlabels, int n, int T, float* feature) {
    int u, t;
    for(u=0; u < n; u++) {
        for(t=0; t < T; t++) {
            feature[t*n + u] = (float) tlabels[t*n + u];
        }
    }
}
void vertex_feature_labels(bool* tlabels, int n, int T, float* feature)
// float feature t x n x 1
// boolean tlabels t x n
{
    int u, t;
    for(u=0; u < n; u++) {
        for(t=0; t < T; t++) {
            feature[t*n + u] = (float) tlabels[t*n + u];
        }
    }
}

// void vertex_feature_neighbors(bool* tadj, bool* tlabels, int n, int T, float* feature)
// // float feature T x n x 3
// // boolean tadj T x n x n
// // boolean tlabels T x n
// {
//     int u, v, t;
//     int deg, red, max_deg=0, max_red=0;
//     for(t=0; t < T; t++) {
//         for(u=0; u < n; u++) {
//             deg = red = 0;
//             // todo optimize
//             // for(v=u; v < n; v++) {
//             for(v=0; v < n; v++) {
//                 if(tadj[t*n*n+u*n+v]) {
//                     deg++;
//                     if(tlabels[t*n + v])
//                         red++;
//                 }
//             }
//             if (deg > max_deg)
//                 max_deg = deg;
//             if (red > max_red)
//                 max_red = red;
//             feature[t*n*3+u*3] = (float) tlabels[t*n + u];
//             feature[t*n*3+u*3+1] = (float) deg;
//             feature[t*n*3+u*3+2] = (float) red;
//         }
//     }
//     if (max_deg > 0) {
//         for(t=0; t < T; t++)
//             for(u=0; u < n; u++)
//                 feature[t*n*3+u*3+1] = (float) feature[t*n*3+u*3+1]/(max_deg);
//     }
//     if (max_red > 0) {
//         for(t=0; t < T; t++)
//             for(u=0; u < n; u++)
//                 feature[t*n*3+u*3+2] = (float) feature[t*n*3+u*3+2]/(max_red);
//     }
// }

void vertex_feature_neighbors1(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 3
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int u, v, t;
    int deg, red;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            deg = red = 0;
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    deg++;
                    if(tlabels[t*n + v])
                        red++;
                }
            }
            feature[t*n*3+u*3] = (float) tlabels[t*n + u];
            feature[t*n*3+u*3+1] = (float) deg/n;
            feature[t*n*3+u*3+2] = (float) red/n;
        }
    }
}

// void vertex_feature_subtrees1(bool* tadj, bool* tlabels, int n, int T, float* feature)
// // float feature T x n x 3
// // boolean tadj T x n x n
// // boolean tlabels T x n
// {
//     int u, v, t;
//     int black, red, deg;
//     for(t=0; t < T; t++) {
//         for(u=0; u < n; u++) {
//             black = red = deg = 0;
//             for(v=0; v < n; v++) {
//                 if(tadj[t*n*n+u*n+v]) {
//                     if(tlabels[t*n + v])
//                         red++;
//                     else
//                         black++;
//                     deg++;
//                 }
//             }
//             feature[t*n*3+u*3] = (float) tlabels[t*n + u];
//             if(deg > 0) {
//                 feature[t*n*3+u*3+1] = (float) red/(2*deg);
//                 feature[t*n*3+u*3+2] = (float) black/(2*deg);
//             }
//         }
//     }
// }
void vertex_feature_subtrees1(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 3
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int u, v, t;
    int black, red;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            black = red = 0;
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    if(tlabels[t*n + v])
                        red++;
                    else
                        black++;
                }
            }
            feature[t*n*3+u*3] = (float) tlabels[t*n + u];
            feature[t*n*3+u*3+1] = (float) red/n;
            feature[t*n*3+u*3+2] = (float) black/n;
        }
    }
}


void vertex_feature2_subtrees1(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 3
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int u, v, t;
    int black, red;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            black = red = 0;
            if(tlabels[t*n + u]) {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(!tlabels[t*n + v])
                            black++;
                    }
                }
            }
            else {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(tlabels[t*n + v])
                            red++;
                    }
                }
            }
            feature[t*n*3+u*3] = (float) tlabels[t*n + u];
            feature[t*n*3+u*3+1] = (float) red/n;
            feature[t*n*3+u*3+2] = (float) black/n;
        }
    }
}


// void vertex_feature_neighbors2(bool* tadj, bool* tlabels, int n, int T, float* feature)
// // float feature T x n x 5
// // boolean tadj T x n x n
// // boolean tlabels T x n
// {
//     int u, v, t;
//     int deg, red, max_deg=0, max_red=0, max2deg=0, max2red=0;
//     for(t=0; t < T; t++) {
//         for(u=0; u < n; u++) {
//             deg = red = 0;
//             for(v=0; v < n; v++) {
//                 if(tadj[t*n*n+u*n+v]) {
//                     deg++;
//                     if(tlabels[t*n + v])
//                         red++;
//                 }
//             }
//             if (deg > max_deg)
//                 max_deg = deg;
//             if (red > max_red)
//                 max_red = red;
//             feature[t*n*5+u*5] = (float) tlabels[t*n + u];
//             feature[t*n*5+u*5+1] = (float) deg;
//             feature[t*n*5+u*5+2] = (float) red;
//         }
//         for(u=0; u < n; u++) {
//             for(v=0; v < n; v++) {
//                 if(tadj[t*n*n+u*n+v]) {
//                     feature[t*n*5+u*5+3] += feature[t*n*5+v*5+1];
//                     feature[t*n*5+u*5+4] += feature[t*n*5+v*5+2];
//                 }
//             }
//             if (feature[t*n*5+u*5+3] > max2deg)
//                 max2deg = feature[t*n*5+u*5+3];
//             if (feature[t*n*5+u*5+4] > max2red)
//                 max2red = feature[t*n*5+u*5+4];
//         }

//     }
//     if (max_deg > 0) {
//         for(t=0; t < T; t++)
//             for(u=0; u < n; u++)
//                 feature[t*n*5+u*5+1] = (float) feature[t*n*5+u*5+1]/(max_deg);
//     }
//     if (max_red > 0) {
//         for(t=0; t < T; t++)
//             for(u=0; u < n; u++)
//                 feature[t*n*5+u*5+2] = (float) feature[t*n*5+u*5+2]/(max_red);
//     }
//     if (max2deg > 0) {
//         for(t=0; t < T; t++)
//             for(u=0; u < n; u++)
//                 feature[t*n*5+u*5+3] = (float) feature[t*n*5+u*5+3]/(2*max2deg);
//     }
//     if (max2red > 0) {
//         for(t=0; t < T; t++)
//             for(u=0; u < n; u++)
//                 feature[t*n*5+u*5+4] = (float) feature[t*n*5+u*5+4]/(2*max2red);
//     }
// }


void vertex_feature_neighbors2(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 5
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int u, v, w, t;
    int deg, red, deg2, red2;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            deg = red = deg2 = red2 = 0;
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    deg++;
                    if(tlabels[t*n + v])
                        red++;
                    for(w=0; w < n; w++) {
                        if(w != u && tadj[t*n*n+v*n+w]) {
                            deg2++;
                            if(tlabels[t*n + w])
                                red2++;
                        }
                    }
                }
            }
            feature[t*n*5+u*5] = (float) tlabels[t*n + u];
            feature[t*n*5+u*5+1] = (float) deg/n;
            feature[t*n*5+u*5+2] = (float) red/n;
            feature[t*n*5+u*5+3] = (float) deg2/(2*n);
            feature[t*n*5+u*5+4] = (float) red2/(2*n);
        }
    }
}



// void vertex_feature_subtrees2(bool* tadj, bool* tlabels, int n, int T, float* feature)
// // float feature T x n x 5
// // boolean tadj T x n x n
// // boolean tlabels T x n
// {
//     int u, v, w, t;
//     int black, red, deg, black2,red2, deg2;
//     for(t=0; t < T; t++) {
//         for(u=0; u < n; u++) {
//             black = red = deg = 0;
//             black2 = red2 = deg2 = 0;
//             for(v=0; v < n; v++) {
//                 if(tadj[t*n*n+u*n+v]) {
//                     if(tlabels[t*n + v])
//                         red++;
//                     else
//                         black++;
//                     deg++;
//                     for(w=0; w < n; w++) {
//                         if(w != u && tadj[t*n*n+v*n+w]) {
//                             if(tlabels[t*n + w])
//                                 red2++;
//                             else
//                                 black2++;
//                             deg2++;
//                         }
//                     }
//                 }
//             }
//             feature[t*n*5+u*5] = (float) tlabels[t*n + u];
//             if(deg > 0) {
//                 feature[t*n*5+u*5+1] = (float) red/(2*deg);
//                 feature[t*n*5+u*5+2] = (float) black/(2*deg);
//             }
//             if(deg2 > 0) {
//                 feature[t*n*5+u*5+3] = (float) red2/(4*deg2);
//                 feature[t*n*5+u*5+4] = (float) black2/(4*deg2);
//             }
//         }
//     }
// }

void vertex_feature_subtrees2(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 5
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int u, v, w, t;
    int black, red, black2, red2;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            black = red = 0;
            black2 = red2 = 0;
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    if(tlabels[t*n + v])
                        red++;
                    else
                        black++;
                    for(w=0; w < n; w++) {
                        if(w != u && tadj[t*n*n+v*n+w]) {
                            if(tlabels[t*n + w])
                                red2++;
                            else
                                black2++;
                        }
                    }
                }
            }
            feature[t*n*5+u*5] = (float) tlabels[t*n + u];
            feature[t*n*5+u*5+1] = (float) red/n;
            feature[t*n*5+u*5+2] = (float) black/n;
            feature[t*n*5+u*5+3] = (float) red2/(2*n);
            feature[t*n*5+u*5+4] = (float) black2/(2*n);
        }
    }
}


void vertex_feature2_subtrees2(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 5
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int u, v, w, t;
    int black, red, black2, red2;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            black = red = 0;
            black2 = red2 = 0;
            if(tlabels[t*n + u]) {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(!tlabels[t*n + v])
                            black++;
                        for(w=0; w < n; w++) {
                            if(w != u && tadj[t*n*n+v*n+w] && !tlabels[t*n + w])
                                black2++;
                        }
                    }
                }
            }
            else {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(tlabels[t*n + v])
                            red++;
                        for(w=0; w < n; w++) {
                            if(w != u && tadj[t*n*n+v*n+w] && tlabels[t*n + w])
                                red2++;
                        }
                    }
                }
            }
            feature[t*n*3+u*3] = (float) tlabels[t*n + u];
            feature[t*n*3+u*3+1] = (float) red/n;
            feature[t*n*3+u*3+2] = (float) black/n;
            feature[t*n*5+u*5+3] = (float) red2/(2*n);
            feature[t*n*5+u*5+4] = (float) black2/(2*n);
        }
    }
}



void vertex_feature_walks1(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 4
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int t, v, u;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            if(tlabels[t*n + u]) {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(tlabels[t*n + v])
                            feature[t*n*4+u*4+3] += 1.0;
                        else
                            feature[t*n*4+u*4+2] += 1.0;
                    }
                }
            }
            else {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(tlabels[t*n + v])
                            feature[t*n*4+u*4+1] += 1.0;
                        else
                            feature[t*n*4+u*4] += 1.0;
                    }
                }
            }
        }
    }
}


void vertex_feature_walks2(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 12
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int t, u, v, w;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            if(tlabels[t*n + u]) {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(tlabels[t*n + v]) {
                            feature[t*n*12+u*12+3] += 1.0;
                            for(w=0; w < n; w++) {
                                if(w != u && tadj[t*n*n+v*n+w]) {
                                    if(tlabels[t*n + w])
                                        feature[t*n*12+u*12+11] += 1.0;
                                    else
                                        feature[t*n*12+u*12+10] += 1.0;
                                }
                            }
                        }
                        else {
                            feature[t*n*12+u*12+2] += 1.0;
                            for(w=0; w < n; w++) {
                                if(w != u && tadj[t*n*n+v*n+w]) {
                                    if(tlabels[t*n + w])
                                        feature[t*n*12+u*12+9] += 1.0;
                                    else
                                        feature[t*n*12+u*12+8] += 1.0;
                                }
                            }
                        }
                    }
                }
            }
            else {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(tlabels[t*n + v]) {
                            feature[t*n*12+u*12+1] += 1.0;
                            for(w=0; w < n; w++) {
                                if(w != u && tadj[t*n*n+v*n+w]) {
                                    if(tlabels[t*n + w])
                                        feature[t*n*12+u*12+7] += 1.0;
                                    else
                                        feature[t*n*12+u*12+6] += 1.0;
                                }
                            }
                        }
                        else {
                            feature[t*n*12+u*12] += 1.0;
                            for(w=0; w < n; w++) {
                                if(w != u && tadj[t*n*n+v*n+w]) {
                                    if(tlabels[t*n + w])
                                        feature[t*n*12+u*12+5] += 1.0;
                                    else
                                        feature[t*n*12+u*12+4] += 1.0;
                                }
                            }
                        }
                    }
                }
            }
            feature[t*n*12+u*12+4] = (float) feature[t*n*12+u*12+4]/2;
            feature[t*n*12+u*12+5] = (float) feature[t*n*12+u*12+5]/2;
            feature[t*n*12+u*12+6] = (float) feature[t*n*12+u*12+6]/2;
            feature[t*n*12+u*12+7] = (float) feature[t*n*12+u*12+7]/2;
            feature[t*n*12+u*12+8] = (float) feature[t*n*12+u*12+8]/2;
            feature[t*n*12+u*12+9] = (float) feature[t*n*12+u*12+9]/2;
            feature[t*n*12+u*12+10] = (float) feature[t*n*12+u*12+10]/2;
            feature[t*n*12+u*12+11] = (float) feature[t*n*12+u*12+11]/2;
        }

    }
}


// void vertex_feature_walks2(bool* tadj, bool* tlabels, int n, int T, float* feature)
// // float feature T x n x 8
// // boolean tadj T x n x n
// // boolean tlabels T x n
// {
//     int t, u, v, w;
//     for(t=0; t < T; t++) {
//         for(u=0; u < n; u++) {
//             if(tlabels[t*n + u]) {
//                 for(v=0; v < n; v++) {
//                     if(tadj[t*n*n+u*n+v]) {
//                         if(tlabels[t*n + v]) {
//                             for(w=0; w < n; w++) {
//                                 if(w != u && tadj[t*n*n+v*n+w]) {
//                                     if(tlabels[t*n + w])
//                                         feature[t*n*8+u*8+7] += 1.0; //1 1 1
//                                     else
//                                         feature[t*n*8+u*8+6] += 1.0; //1 1 0
//                                 }
//                             }
//                         }
//                         else {
//                             for(w=0; w < n; w++) {
//                                 if(w != u && tadj[t*n*n+v*n+w]) {
//                                     if(tlabels[t*n + w])
//                                         feature[t*n*8+u*8+5] += 1.0; //1 0 1
//                                     else
//                                         feature[t*n*8+u*8+4] += 1.0; //1 0 0
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//             else {
//                 for(v=0; v < n; v++) {
//                     if(tadj[t*n*n+u*n+v]) {
//                         if(tlabels[t*n + v]) {
//                             for(w=0; w < n; w++) {
//                                 if(w != u && tadj[t*n*n+v*n+w]) {
//                                     if(tlabels[t*n + w])
//                                         feature[t*n*8+u*8+3] += 1.0; // 0 1 1
//                                     else
//                                         feature[t*n*8+u*8+2] += 1.0; // 0 1 0
//                                 }
//                             }
//                         }
//                         else {
//                             for(w=0; w < n; w++) {
//                                 if(w != u && tadj[t*n*n+v*n+w]) {
//                                     if(tlabels[t*n + w])
//                                         feature[t*n*8+u*8+1] += 1.0; // 0 0 1
//                                     else
//                                         feature[t*n*8+u*8] += 1.0; // 0 0 0
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//     }
// }


void vertex_feature_walks3(bool* tadj, bool* tlabels, int n, int T, float* feature)
// float feature T x n x 16
// boolean tadj T x n x n
// boolean tlabels T x n
{
    int t, u, v, w, x;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            if(tlabels[t*n + u]) {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(tlabels[t*n + v]) {
                            for(w=0; w < n; w++) {
                                if(tadj[t*n*n+v*n+w]) {
                                    if(tlabels[t*n + w]) {
                                        for(x=0; x < n; x++) {
                                            if(tadj[t*n*n+w*n+x]) {
                                                if(tlabels[t*n + x])
                                                    feature[t*n*16+u*16+15] += 1.0; //1 1 1 1
                                                else
                                                    feature[t*n*16+u*16+14] += 1.0; //1 1 1 0
                                            }
                                        }
                                    }
                                    else {
                                        for(x=0; x < n; x++) {
                                            if(tadj[t*n*n+w*n+x]) {
                                                if(tlabels[t*n + x])
                                                    feature[t*n*16+u*16+13] += 1.0; //1 1 0 1
                                                else
                                                    feature[t*n*16+u*16+12] += 1.0; //1 1 0 0
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            for(w=0; w < n; w++) {
                                if(tadj[t*n*n+v*n+w]) {
                                    if(tlabels[t*n + w]) {
                                        for(x=0; x < n; x++) {
                                            if(tadj[t*n*n+w*n+x]) {
                                                if(tlabels[t*n + x])
                                                    feature[t*n*16+u*16+11] += 1.0; //1 0 1 1
                                                else
                                                    feature[t*n*16+u*16+10] += 1.0; //1 0 1 0
                                            }
                                        }
                                    }
                                    else {
                                        for(x=0; x < n; x++) {
                                            if(tadj[t*n*n+w*n+x]) {
                                                if(tlabels[t*n + x])
                                                    feature[t*n*16+u*16+9] += 1.0; //1 0 0 1
                                                else
                                                    feature[t*n*16+u*16+8] += 1.0; //1 0 0 0
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else {
                for(v=0; v < n; v++) {
                    if(tadj[t*n*n+u*n+v]) {
                        if(tlabels[t*n + v]) {
                            for(w=0; w < n; w++) {
                                if(tadj[t*n*n+v*n+w]) {
                                    if(tlabels[t*n + w]) {
                                        for(x=0; x < n; x++) {
                                            if(tadj[t*n*n+w*n+x]) {
                                                if(tlabels[t*n + x])
                                                    feature[t*n*16+u*16+7] += 1.0; //0 1 1 1
                                                else
                                                    feature[t*n*16+u*16+6] += 1.0; //0 1 1 0
                                            }
                                        }
                                    }
                                    else {
                                        for(x=0; x < n; x++) {
                                            if(tadj[t*n*n+w*n+x]) {
                                                if(tlabels[t*n + x])
                                                    feature[t*n*16+u*16+5] += 1.0; //0 1 0 1
                                                else
                                                    feature[t*n*16+u*16+4] += 1.0; //0 1 0 0
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            for(w=0; w < n; w++) {
                                if(tadj[t*n*n+v*n+w]) {
                                    if(tlabels[t*n + w]) {
                                        for(x=0; x < n; x++) {
                                            if(tadj[t*n*n+w*n+x]) {
                                                if(tlabels[t*n + x])
                                                    feature[t*n*16+u*16+3] += 1.0; //0 0 1 1
                                                else
                                                    feature[t*n*16+u*16+2] += 1.0; //0 0 1 0
                                            }
                                        }
                                    }
                                    else {
                                        for(x=0; x < n; x++) {
                                            if(tadj[t*n*n+w*n+x]) {
                                                if(tlabels[t*n + x])
                                                    feature[t*n*16+u*16+1] += 1.0; //0 0 0 1
                                                else
                                                    feature[t*n*16+u*16] += 1.0; // 0 0 0
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }
}


void vertex_feature_interaction2(bool* tadj, int n, int T, float* feature)
// float feature T x n x 2 x n
// boolean tadj T x n x n
{
    int t, u, v, w;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    feature[t*n*2*n+u*n*2+v] = 1.0;
                    for(w=0; w < n; w++) {
                        if(tadj[t*n*n+v*n+w]) {
                            feature[t*n*2*n+u*n*2+n+w] = 0.5;
                        }
                    }
                }
            }
        }
    }
}


void vertex_feature_interaction3(bool* tadj, int n, int T, float* feature)
// float feature T x n x 3 x n
// boolean tadj T x n x n
{
    int t, u, v, w, x;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    feature[t*n*3*n+u*n*3+v] = 1.0;
                    for(w=0; w < n; w++) {
                        if(tadj[t*n*n+v*n+w]) {
                            feature[t*n*3*n+u*n*3+n+w] = 0.5;
                            for(x=0; x < n; x++) {
                                if(tadj[t*n*n+w*n+x]) {
                                    feature[t*n*3*n+u*n*3+2*n+x] = 0.25;
                                }
                            }
                        }
                    }
                }
                
            }
        }
    }
}

void vertex_feature_degree2(bool* tadj, int* degs, int n, int T, float* feature)
// float feature T x n x (n + 1)
// boolean tadj T x n x n
// int degs T x n
{
    int t, u, v;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            feature[t*n*(n+1)+u*(n+1)] = (float) degs[t*n+u];
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    feature[t*n*(n+1)+u*(n+1)+1+v] = (float) degs[t*n+v]/2;
                }
            }
        }
    }
}


void vertex_feature_degree3(bool* tadj, int* degs, int n, int T, float* feature)
// float feature T x n x (2 x n + 1)
// boolean tadj T x n x n
// int degs T x n
{
    int t, u, v, w;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            feature[t*n*(2*n+1)+u*(2*n+1)] = (float) degs[t*n+u];
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    feature[t*n*(2*n+1)+u*(2*n+1)+1+v] = (float) degs[t*n+v]/2;
                    for(w=0; w < n; w++) {
                        if(tadj[t*n*n+v*n+w]) {
                            feature[t*n*(2*n+1)+u*(2*n+1)+1+n+w] = (float) degs[t*n+w]/4;
                        }
                    }
                }
            }
        }
    }
}


void vertex_feature_degree4(bool* tadj, int* degs, int n, int T, float* feature)
// float feature T x n x (3 x n + 1)
// boolean tadj T x n x n
// int degs T x n
{
    int t, u, v, w, x;
    for(t=0; t < T; t++) {
        for(u=0; u < n; u++) {
            feature[t*n*(3*n+1)+u*(3*n+1)] = (float) degs[t*n+u];
            for(v=0; v < n; v++) {
                if(tadj[t*n*n+u*n+v]) {
                    feature[t*n*(3*n+1)+u*(3*n+1)+1+v] = (float) degs[t*n+v]/2;
                    for(w=0; w < n; w++) {
                        if(tadj[t*n*n+v*n+w]) {
                            feature[t*n*(3*n+1)+u*(3*n+1)+1+n+w] = (float) degs[t*n+w]/4;
                            for(x=0; x < n; x++) {
                                if(tadj[t*n*n+w*n+x]) {
                                    feature[t*n*(3*n+1)+u*(3*n+1)+1+2*n+x] = (float) degs[t*n+x]/8;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



void vertex_feature_labeled_neighbors(bool* tadj, int* labels, int n, int T, float* feature)
// float feature T x n x n x k
// boolean tadj T x n x n
// int labels n
{
    // int i = 0;
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


static bool terminated(int state) {
    return (state == CONVERGED || state == ITERATION_LIMIT_REACHED);
}


double dtgw(const float *f1, const float *f2, int metric_number, int t1, int t2, int n, int columns,
    int init, int window, int max_iterations)
{
    int *warppath = (int *) malloc(t1*t2 * sizeof(int));
    float *warpcost = (float *) malloc(t1*t2 * sizeof(float));
    float *res = (float *) malloc((t1+1)*(t2+1) * sizeof(float));
    float *matchcost = (float *) malloc(n*n * sizeof(float));
    int *rowsol = (int *) malloc(n * sizeof(int));
    int *colsol = (int *) malloc(n * sizeof(int));
    float *rowcost = (float *) malloc(n * sizeof(float));
    float *colcost = (float *) malloc(n * sizeof(float));
    int *region = (int *) malloc(t1*2 * sizeof(int));
    int state, l;

    metric_function *metric = get_metric(metric_number, columns);
    for(int i = 0; i < (t1+1)*(t2+1); i++)
        res[i] = INFINITY;
    for(int i=0; i < t1*t2; i++)
        warpcost[i] = INFINITY;

    sakoe_chiba_band(t1, t2, window, region);

    if (init == DIAGONAL_WARPING) {
        l = init_diagonal_warping(f1, f2, t1, t2, n, columns, metric, warppath, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else if (init == OPTIMISTIC_MATCHING) {
        init_opt_matching(f1, f2, t1, t2, n, columns, metric, region, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else if (init == PRODUCT_WARPING) {
        l = init_product_warping(f1, f2, t1, t2, n, columns, metric, region, warppath, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else {
        init_opt_warping(f1, f2, t1, t2, n, columns, metric, region, warpcost, rowsol, colsol, rowcost, colcost);
        state = WARPCOST_UPDATED;
    }

    double cost = INFINITY;
    double new_cost;
    int iterations = 0;
    while (!terminated(state)) {
        if (state == MATCHCOST_UPDATED) {
            // new_cost = lap(n, matchcost, rowsol, colsol, rowcost, colcost);
            // new_cost = lap<0, int, float>(n, matchcost, 1, rowsol, colsol, rowcost, colcost);
            new_cost = lap<true, int, float>(n, matchcost, false, rowsol, colsol, rowcost, colcost);
            // printf("lap: %.3f\n", new_cost);
            if(new_cost >= cost)
                state = CONVERGED;
            else {
                update_warping(f1, f2, t1, t2, n, columns, metric, rowsol, colsol, region, warpcost);
                // for(int i=0; i < t1; i++) {
                //     printf("%d: ", i);
                //     for(int j=0; j < t2; j++)
                //         printf("%.2f ", warpcost[i*t1+j]);
                //     printf("\n");
                // }
                state = WARPCOST_UPDATED;
            }
        }
        else {
            l = dynamic_timewarping(warpcost, t1, t2, region, warppath, res);
            new_cost = (double) res[0];
            // printf("tw: %.3f\n", new_cost);
            if(new_cost >= cost)
                state = CONVERGED;
            else {
                update_matchcost(f1, f2, n, columns, metric, warppath, l, matchcost);
                state = MATCHCOST_UPDATED;
            }
        }
        cost = new_cost;
        iterations++;
        if(iterations >= max_iterations)
            state = ITERATION_LIMIT_REACHED;
    }

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

int wp_distance(int *warppath, int l) {
    int d = 0;
    for(int i=0; i < 2*l; i += 2) {
        d = max2d(fabs(warppath[i] - warppath[i+1]), d);
    }
    return d;
}


double dtgw_log(const float *f1, const float *f2, int metric_number, int t1, int t2, int n, int columns,
    int init, int window, int max_iterations, int *log)
{
    int *warppath = (int *) malloc(t1*t2 * sizeof(int));
    float *warpcost = (float *) malloc(t1*t2 * sizeof(float));
    float *res = (float *) malloc((t1+1)*(t2+1) * sizeof(float));
    float *matchcost = (float *) malloc(n*n * sizeof(float));
    int *rowsol = (int *) malloc(n * sizeof(int));
    int *colsol = (int *) malloc(n * sizeof(int));
    float *rowcost = (float *) malloc(n * sizeof(float));
    float *colcost = (float *) malloc(n * sizeof(float));
    int *region = (int *) malloc(t1*2 * sizeof(int));
    int state, l;

    metric_function *metric = get_metric(metric_number, columns);
    for(int i = 0; i < (t1+1)*(t2+1); i++)
        res[i] = INFINITY;
    for(int i=0; i < t1*t2; i++)
        warpcost[i] = INFINITY;

    sakoe_chiba_band(t1, t2, window, region);

    if (init == DIAGONAL_WARPING) {
        l = init_diagonal_warping(f1, f2, t1, t2, n, columns, metric, warppath, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else if (init == OPTIMISTIC_MATCHING) {
        init_opt_matching(f1, f2, t1, t2, n, columns, metric, region, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else if (init == PRODUCT_WARPING) {
        l = init_product_warping(f1, f2, t1, t2, n, columns, metric, region, warppath, matchcost);
        state = MATCHCOST_UPDATED;
    }
    else {
        init_opt_warping(f1, f2, t1, t2, n, columns, metric, region, warpcost, rowsol, colsol, rowcost, colcost);
        state = WARPCOST_UPDATED;
    }
    
    double cost = INFINITY;
    double new_cost;
    int iterations = 0;
    while (!terminated(state)) {
        if (state == MATCHCOST_UPDATED) {
            // new_cost = lap(n, matchcost, rowsol, colsol, rowcost, colcost);
            new_cost = lap<true, int, float>(n, matchcost, false, rowsol, colsol, rowcost, colcost);
            if(new_cost >= cost)
                state = CONVERGED;
            else {
                update_warping(f1, f2, t1, t2, n, columns, metric, rowsol, colsol, region, warpcost);
                state = WARPCOST_UPDATED;
            }
        }
        else {
            l = dynamic_timewarping(warpcost, t1, t2, region, warppath, res);
            new_cost = (double) res[0];
            if(new_cost >= cost)
                state = CONVERGED;
            else {
                update_matchcost(f1, f2, n, columns, metric, warppath, l, matchcost);
                state = MATCHCOST_UPDATED;
            }
        }
        cost = new_cost;
        iterations++;
        if(iterations >= max_iterations)
            state = ITERATION_LIMIT_REACHED;
    }

    log[0] = iterations;
    log[1] = wp_distance(warppath, l);

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


float tgw(const float *f1, const float *f2, int t1, int t2, int n, int columns, int window, int metric_number)
{
    int *warppath = (int *) malloc(t1*t2 * sizeof(int));
    float *warpcost = (float *) malloc(t1*t2 * sizeof(float));
    float *res = (float *) malloc((t1+1)*(t2+1) * sizeof(float));
    for(int i = 0; i < (t1+1)*(t2+1); i++)
        res[i] = INFINITY;
    int *region = (int *) malloc(t1*2 * sizeof(int));
    int *rowsol = (int *) malloc(n * sizeof(int));
    int *colsol = (int *) malloc(n * sizeof(int));

    float (*metric) (const float*, const float*, int);

    if(metric_number == L1)
        metric = city_block_distance;
    if(metric_number == L2)
        metric = euclidean_distance;
    if(metric_number == DOT)
        metric = negative_dot_product;

    sakoe_chiba_band(t1, t2, window, region);

    for(int u=0; u < n; u++) {
        colsol[u] = u;
        rowsol[u] = u;
    }

    update_warping(f1, f2, t1, t2, n, columns, metric, rowsol, colsol, region, warpcost);
    dynamic_timewarping(warpcost, t1, t2, region, warppath, res);

    float ans = res[0];

    free(warppath);
    free(warpcost);
    free(rowsol);
    free(colsol);
    free(region);
    free(res);

    return ans;
}


#ifdef __cplusplus
}
#endif

