// GPU shape search for primitive: every shape type (incl. beziers and -rep), any image size.
//
//   gpu_search.exe target.bin canvas.bin out.bin count mode alpha [seed] [rep]
//
// target.bin / canvas.bin : int32 W, int32 H, then W*H*4 RGBA bytes (canvas = state to continue from)
// out.bin                 : int32 n, then n records of {int type, int alpha, float p[8], int r, g, b, int step}
// mode  : 0=combo 1=triangle 2=rect 3=ellipse 4=circle 5=rotated rect 6=bezier 7=rotated ellipse 8=polygon (convex quads)
// alpha : 1..255, or 0 to let the search pick alpha per shape (like primitive -a 0)
//
// Each step is primitive's Model.Step: 16 groups x 1000 random shapes scored, the best of each group hill-climbed
// (async warps sharing a best, stop after 100 non-improving evals), best group committed. Scoring uses hard-edge
// coverage (no anti-aliasing); the Go side replays the shapes with its own anti-aliased rasteriser and colours.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

constexpr int MAXROWS = 512, KW = 16, GROUPS = 16, PER_GROUP = 1000, MAX_FAILS = 100;
enum { T_TRI = 1, T_RECT = 2, T_ELLIPSE = 3, T_CIRCLE = 4, T_ROTRECT = 5, T_QUAD = 6, T_ROTELL = 7, T_POLY = 8 };

struct Cand { int t, alpha; float p[8]; };
struct Rec { int t, alpha; float p[8]; int r, g, b; int step; };

__device__ __forceinline__ int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ---------------- RNG ----------------
__device__ __forceinline__ uint64_t sm64(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ int rint_(uint64_t& s, int n) { return (int)(((sm64(s) >> 32) * (uint64_t)n) >> 32); }
__device__ __forceinline__ float rf01(uint64_t& s) { return (float)(sm64(s) >> 40) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float rnorm(uint64_t& s) {
    uint64_t r = sm64(s);
    float u1 = ((uint32_t)(r >> 40) + 1) * (1.0f / 16777216.0f);
    float u2 = ((uint32_t)(r >> 8) & 0xffffff) * (1.0f / 16777216.0f);
    return sqrtf(-2.0f * __logf(u1)) * __cosf(6.2831853f * u2);
}

// ---------------- shapes: random start and mutation (ports of the Go shape types) ----------------
__device__ __forceinline__ bool vertex_ok(float ax, float ay, float bx, float by) {
    float d = (ax * bx + ay * by) * rsqrtf((ax * ax + ay * ay) * (bx * bx + by * by));
    return d < 0.9659258f;     // angle > 15 degrees; NaN (zero-length edge) compares false
}
__device__ bool tri_valid(int x1, int y1, int x2, int y2, int x3, int y3) {
    return vertex_ok(x2 - x1, y2 - y1, x3 - x1, y3 - y1) && vertex_ok(x1 - x2, y1 - y2, x3 - x2, y3 - y2) &&
           vertex_ok(x1 - x3, y1 - y3, x2 - x3, y2 - y3);
}
__device__ void tri_mutate(Cand& c, uint64_t& rng, int W, int H) {
    int x1 = (int)c.p[0], y1 = (int)c.p[1], x2 = (int)c.p[2], y2 = (int)c.p[3], x3 = (int)c.p[4], y3 = (int)c.p[5];
    const int m = 16;
    for (;;) {
        switch (rint_(rng, 3)) {
        case 0: x1 = clampi(x1 + (int)(rnorm(rng) * 16), -m, W - 1 + m); y1 = clampi(y1 + (int)(rnorm(rng) * 16), -m, H - 1 + m); break;
        case 1: x2 = clampi(x2 + (int)(rnorm(rng) * 16), -m, W - 1 + m); y2 = clampi(y2 + (int)(rnorm(rng) * 16), -m, H - 1 + m); break;
        case 2: x3 = clampi(x3 + (int)(rnorm(rng) * 16), -m, W - 1 + m); y3 = clampi(y3 + (int)(rnorm(rng) * 16), -m, H - 1 + m); break;
        }
        if (tri_valid(x1, y1, x2, y2, x3, y3)) break;
    }
    c.p[0] = x1; c.p[1] = y1; c.p[2] = x2; c.p[3] = y2; c.p[4] = x3; c.p[5] = y3;
}
// convex check: all turns have the same sign (Go's Polygon.Valid with Convex=true)
__device__ bool poly_convex(const float* p) {
    bool sign = false;
    for (int a = 0; a < 4; a++) {
        int i = a, j = (a + 1) & 3, k = (a + 2) & 3;
        float c = (p[2 * j] - p[2 * i]) * (p[2 * k + 1] - p[2 * j + 1]) - (p[2 * j + 1] - p[2 * i + 1]) * (p[2 * k] - p[2 * j]);
        if (a == 0) sign = c > 0; else if ((c > 0) != sign) return false;
    }
    return true;
}
__device__ void poly_mutate(Cand& c, uint64_t& rng, int W, int H) {
    const float m = 16;
    for (;;) {
        if (rf01(rng) < 0.25f) {
            int i = rint_(rng, 4), j = rint_(rng, 4);
            float tx = c.p[2 * i], ty = c.p[2 * i + 1];
            c.p[2 * i] = c.p[2 * j]; c.p[2 * i + 1] = c.p[2 * j + 1]; c.p[2 * j] = tx; c.p[2 * j + 1] = ty;
        } else {
            int i = rint_(rng, 4);
            c.p[2 * i] = clampf(c.p[2 * i] + rnorm(rng) * 16, -m, (float)(W - 1) + m);
            c.p[2 * i + 1] = clampf(c.p[2 * i + 1] + rnorm(rng) * 16, -m, (float)(H - 1) + m);
        }
        if (poly_convex(c.p)) break;
    }
}
// Go's Quadratic.Valid: the chord is the longest side (integer-truncated lengths)
__device__ bool quad_valid(const float* p) {
    const int dx12 = (int)(p[0] - p[2]), dy12 = (int)(p[1] - p[3]);
    const int dx23 = (int)(p[2] - p[4]), dy23 = (int)(p[3] - p[5]);
    const int dx13 = (int)(p[0] - p[4]), dy13 = (int)(p[1] - p[5]);
    const int d12 = dx12 * dx12 + dy12 * dy12, d23 = dx23 * dx23 + dy23 * dy23, d13 = dx13 * dx13 + dy13 * dy13;
    return d13 > d12 && d13 > d23;
}
__device__ void quad_mutate(Cand& c, uint64_t& rng, int W, int H) {
    const float m = 16;
    for (;;) {
        const int i = rint_(rng, 3);   // Go has a dead 4th case (width) that is never reached
        c.p[2 * i] = clampf(c.p[2 * i] + rnorm(rng) * 16, -m, (float)(W - 1) + m);
        c.p[2 * i + 1] = clampf(c.p[2 * i + 1] + rnorm(rng) * 16, -m, (float)(H - 1) + m);
        if (quad_valid(c.p)) break;
    }
}
__device__ void shape_mutate(Cand& c, uint64_t& rng, int W, int H, bool mutAlpha) {
    switch (c.t) {
    case T_TRI: tri_mutate(c, rng, W, H); break;
    case T_RECT:
        if (rint_(rng, 2) == 0) { c.p[0] = clampi((int)c.p[0] + (int)(rnorm(rng) * 16), 0, W - 1); c.p[1] = clampi((int)c.p[1] + (int)(rnorm(rng) * 16), 0, H - 1); }
        else                    { c.p[2] = clampi((int)c.p[2] + (int)(rnorm(rng) * 16), 0, W - 1); c.p[3] = clampi((int)c.p[3] + (int)(rnorm(rng) * 16), 0, H - 1); }
        break;
    case T_ELLIPSE: case T_CIRCLE: {
        const bool circ = c.t == T_CIRCLE;
        switch (rint_(rng, 3)) {
        case 0: c.p[0] = clampi((int)c.p[0] + (int)(rnorm(rng) * 16), 0, W - 1); c.p[1] = clampi((int)c.p[1] + (int)(rnorm(rng) * 16), 0, H - 1); break;
        case 1: c.p[2] = clampi((int)c.p[2] + (int)(rnorm(rng) * 16), 1, W - 1); if (circ) c.p[3] = c.p[2]; break;
        case 2: c.p[3] = clampi((int)c.p[3] + (int)(rnorm(rng) * 16), 1, H - 1); if (circ) c.p[2] = c.p[3]; break;
        }
        break; }
    case T_ROTRECT:
        switch (rint_(rng, 3)) {
        case 0: c.p[0] = clampi((int)c.p[0] + (int)(rnorm(rng) * 16), 0, W - 1); c.p[1] = clampi((int)c.p[1] + (int)(rnorm(rng) * 16), 0, H - 1); break;
        case 1: c.p[2] = clampi((int)c.p[2] + (int)(rnorm(rng) * 16), 1, W - 1); c.p[3] = clampi((int)c.p[3] + (int)(rnorm(rng) * 16), 1, H - 1); break;
        case 2: c.p[4] = (float)((int)c.p[4] + (int)(rnorm(rng) * 32)); break;
        }
        break;
    case T_ROTELL:
        switch (rint_(rng, 3)) {
        case 0: c.p[0] = clampf(c.p[0] + rnorm(rng) * 16, 0, (float)(W - 1)); c.p[1] = clampf(c.p[1] + rnorm(rng) * 16, 0, (float)(H - 1)); break;
        case 1: c.p[2] = clampf(c.p[2] + rnorm(rng) * 16, 1, (float)(W - 1)); c.p[3] = clampf(c.p[3] + rnorm(rng) * 16, 1, (float)(W - 1)); break;
        case 2: c.p[4] = c.p[4] + rnorm(rng) * 32; break;
        }
        break;
    case T_POLY: poly_mutate(c, rng, W, H); break;
    case T_QUAD: quad_mutate(c, rng, W, H); break;
    }
    if (mutAlpha) c.alpha = clampi(c.alpha + rint_(rng, 21) - 10, 1, 255);
}
__device__ Cand shape_random(int t, int alpha, uint64_t& rng, int W, int H, bool mutAlpha) {
    Cand c; c.t = t; c.alpha = alpha;
    for (int i = 0; i < 8; i++) c.p[i] = 0;
    switch (t) {
    case T_TRI: {
        int x1 = rint_(rng, W), y1 = rint_(rng, H);
        c.p[0] = x1; c.p[1] = y1;
        c.p[2] = x1 + rint_(rng, 31) - 15; c.p[3] = y1 + rint_(rng, 31) - 15;
        c.p[4] = x1 + rint_(rng, 31) - 15; c.p[5] = y1 + rint_(rng, 31) - 15;
        tri_mutate(c, rng, W, H);
        break; }
    case T_RECT: {
        int x1 = rint_(rng, W), y1 = rint_(rng, H);
        c.p[0] = x1; c.p[1] = y1;
        c.p[2] = clampi(x1 + rint_(rng, 32) + 1, 0, W - 1); c.p[3] = clampi(y1 + rint_(rng, 32) + 1, 0, H - 1);
        break; }
    case T_ELLIPSE:
        c.p[0] = rint_(rng, W); c.p[1] = rint_(rng, H); c.p[2] = rint_(rng, 32) + 1; c.p[3] = rint_(rng, 32) + 1;
        break;
    case T_CIRCLE:
        c.p[0] = rint_(rng, W); c.p[1] = rint_(rng, H); c.p[2] = c.p[3] = rint_(rng, 32) + 1;
        break;
    case T_ROTRECT:
        c.p[0] = rint_(rng, W); c.p[1] = rint_(rng, H); c.p[2] = rint_(rng, 32) + 1; c.p[3] = rint_(rng, 32) + 1; c.p[4] = rint_(rng, 360);
        shape_mutate(c, rng, W, H, false);
        break;
    case T_QUAD:
        c.p[0] = rf01(rng) * W; c.p[1] = rf01(rng) * H;
        c.p[2] = c.p[0] + rf01(rng) * 40 - 20; c.p[3] = c.p[1] + rf01(rng) * 40 - 20;
        c.p[4] = c.p[2] + rf01(rng) * 40 - 20; c.p[5] = c.p[3] + rf01(rng) * 40 - 20;
        quad_mutate(c, rng, W, H);
        break;
    case T_ROTELL:
        c.p[0] = rf01(rng) * W; c.p[1] = rf01(rng) * H; c.p[2] = rf01(rng) * 32 + 1; c.p[3] = rf01(rng) * 32 + 1; c.p[4] = rf01(rng) * 360;
        break;
    case T_POLY:
        c.p[0] = rf01(rng) * W; c.p[1] = rf01(rng) * H;
        for (int i = 1; i < 4; i++) { c.p[2 * i] = c.p[0] + rf01(rng) * 40 - 20; c.p[2 * i + 1] = c.p[1] + rf01(rng) * 40 - 20; }
        poly_mutate(c, rng, W, H);
        break;
    }
    if (mutAlpha) c.alpha = 128;
    return c;
}

// ---------------- shapes: hard-edge scanline spans ----------------
// per-warp scratch: scanline spans of the candidate being scored (+ the flattened curve for beziers)
struct WS {
    short2 xr[MAXROWS]; unsigned short cov[MAXROWS]; int ylo, rows;
    float qx[17], qy[17], qex[16], qey[16], qinv[16], qA[16], qB[16];   // bezier as 16 segments
};

struct Aux { int ylo, rows; bool bad; int i[7]; float f[8]; };

__device__ void shape_prep(const Cand& c, int W, int H, Aux& A) {
    int ylo = 0, yhi = -1;
    A.bad = false;
    switch (c.t) {
    case T_TRI: {
        int x1 = (int)c.p[0], y1 = (int)c.p[1], x2 = (int)c.p[2], y2 = (int)c.p[3], x3 = (int)c.p[4], y3 = (int)c.p[5], t;
        if (y1 > y3) { t = x1; x1 = x3; x3 = t; t = y1; y1 = y3; y3 = t; }
        if (y1 > y2) { t = x1; x1 = x2; x2 = t; t = y1; y1 = y2; y2 = t; }
        if (y2 > y3) { t = x2; x2 = x3; x3 = t; t = y2; y2 = y3; y3 = t; }
        A.i[0] = x1; A.i[1] = y1; A.i[2] = x2; A.i[3] = y2; A.i[4] = x3; A.i[5] = y3;
        A.i[6] = (y2 != y3 && y1 != y2) ? x1 + (int)(((double)(y2 - y1) / (double)(y3 - y1)) * (double)(x3 - x1)) : x1;
        ylo = max(y1, 0); yhi = min(y3, H - 1);
        break; }
    case T_RECT: {
        int x1 = (int)c.p[0], y1 = (int)c.p[1], x2 = (int)c.p[2], y2 = (int)c.p[3], t;
        if (x1 > x2) { t = x1; x1 = x2; x2 = t; }
        if (y1 > y2) { t = y1; y1 = y2; y2 = t; }
        A.i[0] = x1; A.i[1] = y1; A.i[2] = x2; A.i[3] = y2;
        ylo = max(y1, 0); yhi = min(y2, H - 1);
        break; }
    case T_ELLIPSE: case T_CIRCLE: {
        int X = (int)c.p[0], Y = (int)c.p[1], Rx = (int)c.p[2], Ry = (int)c.p[3];
        A.i[0] = X; A.i[1] = Y; A.i[2] = Rx; A.i[3] = Ry; A.f[0] = (float)Rx / (float)Ry;
        ylo = max(Y - Ry + 1, 0); yhi = min(Y + Ry - 1, H - 1);
        break; }
    case T_ROTRECT: {
        const float sx = c.p[2], sy = c.p[3], ang = c.p[4] * 0.017453292519943295f;
        const float cs = cosf(ang), sn = sinf(ang);
        const int X = (int)c.p[0], Y = (int)c.p[1];
        const float cx[4] = {-sx / 2, sx / 2, sx / 2, -sx / 2}, cy[4] = {-sy / 2, -sy / 2, sy / 2, sy / 2};
        int miny = 1 << 20, maxy = -(1 << 20);
        for (int k = 0; k < 4; k++) {
            const int vx = (int)(cx[k] * cs - cy[k] * sn) + X, vy = (int)(cx[k] * sn + cy[k] * cs) + Y;
            A.f[2 * k] = vx; A.f[2 * k + 1] = vy; miny = min(miny, vy); maxy = max(maxy, vy);
        }
        ylo = max(miny, 0); yhi = min(maxy, H - 1);
        break; }
    case T_QUAD: {
        for (int k = 0; k < 6; k++) A.f[k] = c.p[k];
        const float y1 = c.p[1], y2 = c.p[3], y3 = c.p[5];
        float ymin = fminf(y1, y3), ymax = fmaxf(y1, y3);
        const float den = y1 - 2 * y2 + y3;
        if (fabsf(den) > 1e-6f) {
            const float t = (y1 - y2) / den;
            if (t > 0 && t < 1) { const float yy = (1 - t) * (1 - t) * y1 + 2 * t * (1 - t) * y2 + t * t * y3; ymin = fminf(ymin, yy); ymax = fmaxf(ymax, yy); }
        }
        ylo = max((int)floorf(ymin - 1.5f), 0); yhi = min((int)floorf(ymax + 0.5f), H - 1);
        break; }
    case T_ROTELL: {
        const float Rx = c.p[2], Ry = c.p[3], ang = c.p[4] * 0.017453292519943295f;
        const float cs = cosf(ang), sn = sinf(ang), ia = 1.0f / (Rx * Rx), ib = 1.0f / (Ry * Ry);
        A.f[0] = c.p[0]; A.f[1] = c.p[1];
        A.f[2] = cs * cs * ia + sn * sn * ib; A.f[3] = 2 * cs * sn * (ia - ib); A.f[4] = sn * sn * ia + cs * cs * ib;
        const float hy = sqrtf(Rx * Rx * sn * sn + Ry * Ry * cs * cs);
        ylo = max((int)floorf(c.p[1] - hy), 0); yhi = min((int)ceilf(c.p[1] + hy), H - 1);
        break; }
    case T_POLY: {
        float miny = 1e30f, maxy = -1e30f;
        for (int k = 0; k < 8; k++) A.f[k] = c.p[k];
        for (int k = 0; k < 4; k++) { miny = fminf(miny, c.p[2 * k + 1]); maxy = fmaxf(maxy, c.p[2 * k + 1]); }
        ylo = max((int)floorf(miny), 0); yhi = min((int)ceilf(maxy), H - 1);
        break; }
    }
    A.ylo = ylo; A.rows = max(yhi - ylo + 1, 0);
    if (A.rows > MAXROWS) { A.rows = 0; A.bad = true; }   // taller than the row buffer: never selected
}

// span [a,b] of scanline y (a > b means empty)
__device__ __forceinline__ void shape_row(const Cand& c, const Aux& A, const WS& w, int y, int W, int& a, int& b, int& cov) {
    a = 1; b = 0; cov = 0xffff;
    switch (c.t) {
    case T_TRI: {
        const int x1 = A.i[0], y1 = A.i[1], x2 = A.i[2], y2 = A.i[3], x3 = A.i[4], y3 = A.i[5], x4 = A.i[6];
        int aa, bb;
        if (y2 == y3) {
            aa = (x1 * (y2 - y1) + (y - y1) * (x2 - x1)) / max(y2 - y1, 1);
            bb = (x1 * (y3 - y1) + (y - y1) * (x3 - x1)) / max(y3 - y1, 1);
        } else if (y1 == y2) {
            if (y == y1) return;                      // Go does not draw the flat top row
            const int u = y3 - y + 1;
            aa = (x3 * (y3 - y1) - u * (x3 - x1)) / (y3 - y1);
            bb = (x3 * (y3 - y2) - u * (x3 - x2)) / (y3 - y2);
        } else if (y <= y2) {
            aa = (x1 * (y2 - y1) + (y - y1) * (x2 - x1)) / (y2 - y1);
            bb = (x1 * (y2 - y1) + (y - y1) * (x4 - x1)) / (y2 - y1);
        } else {
            const int u = y3 - y + 1;
            aa = (x3 * (y3 - y2) - u * (x3 - x2)) / (y3 - y2);
            bb = (x3 * (y3 - y2) - u * (x3 - x4)) / (y3 - y2);
        }
        if (aa > bb) { int s = aa; aa = bb; bb = s; }
        if (aa >= W || bb < 0) return;
        a = clampi(aa, 0, W - 1); b = clampi(bb, 0, W - 1);
        return; }
    case T_RECT:
        a = clampi(A.i[0], 0, W - 1); b = clampi(A.i[2], 0, W - 1);
        return;
    case T_ELLIPSE: case T_CIRCLE: {
        const int dy = abs(y - A.i[1]), Ry = A.i[3];
        const int s = (int)(sqrtf((float)(Ry * Ry - dy * dy)) * A.f[0]);
        a = max(A.i[0] - s, 0); b = min(A.i[0] + s, W - 1);
        return; }
    case T_ROTRECT: {   // Go: dense edge samples, min/max x per integer row
        int mn = 1 << 20, mx = -(1 << 20);
        for (int k = 0; k < 4; k++) {
            const int j = (k + 1) & 3;
            const float xi = A.f[2 * k], yi = A.f[2 * k + 1], xj = A.f[2 * j], yj = A.f[2 * j + 1];
            if ((float)y < fminf(yi, yj) || (float)y > fmaxf(yi, yj)) continue;
            if (yi == yj) { mn = min(mn, (int)fminf(xi, xj)); mx = max(mx, (int)fmaxf(xi, xj)); }
            else { const int x = (int)(xi + ((float)y - yi) * (xj - xi) / (yj - yi)); mn = min(mn, x); mx = max(mx, x); }
        }
        if (mn > mx) return;
        a = max(mn, 0); b = min(mx, W - 1);
        return; }
    case T_ROTELL: {
        const float dy = (float)y + 0.5f - A.f[1];
        const float qb = A.f[3] * dy, qc = A.f[4] * dy * dy - 1.0f, disc = qb * qb - 4 * A.f[2] * qc;
        if (disc < 0) return;
        const float sq = sqrtf(disc), inv = 0.5f / A.f[2];
        const int lo = (int)ceilf(A.f[0] + (-qb - sq) * inv - 0.5f), hi = (int)floorf(A.f[0] + (-qb + sq) * inv - 0.5f);
        if (lo > hi) return;
        a = max(lo, 0); b = min(hi, W - 1);
        return; }
    case T_QUAD: {      // candidate pixels: those whose centre can be within ~1 px of the curve in this row's band.
                        // Exact per-pixel coverage is computed later (quad_cov16); cov=0 marks "per pixel".
        const float lo = (float)y - 0.5f, hi = (float)y + 1.5f;
        float xmin = 1e30f, xmax = -1e30f;
        int mask = 0;
        for (int i = 0; i < 16; i++) {
            const float x0 = w.qx[i], y0 = w.qy[i], x1 = w.qx[i + 1], y1 = w.qy[i + 1];
            if (fmaxf(y0, y1) < lo || fminf(y0, y1) > hi) continue;
            mask |= 1 << i;
            float t0 = 0.0f, t1 = 1.0f;
            if (y1 != y0) {
                t0 = (lo - y0) / (y1 - y0); t1 = (hi - y0) / (y1 - y0);
                if (t0 > t1) { const float t = t0; t0 = t1; t1 = t; }
                t0 = fmaxf(t0, 0.0f); t1 = fminf(t1, 1.0f);
            }
            const float xa = x0 + (x1 - x0) * t0, xb = x0 + (x1 - x0) * t1;
            xmin = fminf(xmin, fminf(xa, xb)); xmax = fmaxf(xmax, fmaxf(xa, xb));
        }
        if (xmin > xmax) return;
        const int aa = max((int)floorf(xmin - 1.5f), 0), bb = min((int)floorf(xmax + 0.5f), W - 1);
        if (aa > bb) return;
        a = aa; b = bb; cov = mask;     // bezier rows keep their segment mask here (coverage is per pixel)
        return; }
    case T_POLY: {      // pixel-centre crossing of the convex quad
        const float yc = (float)y + 0.5f;
        float mn = 1e30f, mx = -1e30f;
        for (int k = 0; k < 4; k++) {
            const int j = (k + 1) & 3;
            const float xi = A.f[2 * k], yi = A.f[2 * k + 1], xj = A.f[2 * j], yj = A.f[2 * j + 1];
            if ((yi <= yc && yc < yj) || (yj <= yc && yc < yi)) {
                const float x = xi + (yc - yi) * (xj - xi) / (yj - yi); mn = fminf(mn, x); mx = fmaxf(mx, x);
            }
        }
        if (mn > mx) return;
        const int lo = (int)ceilf(mn - 0.5f), hi = (int)floorf(mx - 0.5f);
        if (lo > hi) return;
        a = max(lo, 0); b = min(hi, W - 1);
        return; }
    }
}

// ---------------- warp evaluator ----------------

__device__ __forceinline__ long long wsum(long long v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Coverage (0..65535) of pixel (x,y) by a 0.5 px wide stroke along the flattened curve: take the nearest segment, then the
// exact area of a strip of that width and direction at that distance from the pixel centre (trapezoid CDF of the square's
// projection onto the normal). 0 means untouched (Go emits no span there either).
__device__ __forceinline__ uint32_t quad_cov16(const WS& w, int x, int y) {
    const float cx = (float)x + 0.5f, cy = (float)y + 0.5f;
    float best = 1e30f; int bi = 0;
    for (unsigned mask = w.cov[y - w.ylo]; mask; mask &= mask - 1) {
        const int i = __ffs(mask) - 1;
        const float dx = cx - w.qx[i], dy = cy - w.qy[i];
        const float t = fminf(fmaxf((dx * w.qex[i] + dy * w.qey[i]) * w.qinv[i], 0.0f), 1.0f);
        const float px = dx - t * w.qex[i], py = dy - t * w.qey[i], d2 = px * px + py * py;
        if (d2 < best) { best = d2; bi = i; }
    }
    if (best > 0.93f) return 0u;                 // farther than any pixel the stroke can touch (~0.96 px)
    const float d = sqrtf(best), A = w.qA[bi], B = fmaxf(w.qB[bi], 1e-6f), h = 0.25f;
    const float S = 0.5f * (A + B), P = 0.5f * (A - B), l0 = 1.0f / A;
    auto F = [&](float u) -> float {            // integral of the trapezoid density from 0 to u
        if (u >= S) return 0.5f;
        if (u <= P) return u * l0;
        const float v = u - P;
        return P * l0 + l0 * (v - 0.5f * v * v / B);
    };
    const float cov = F(d + h) + (d >= h ? -F(d - h) : F(h - d));
    const int v = (int)(cov * 65535.0f + 0.5f);
    return v < 64 ? 0u : (uint32_t)min(v, 65535);
}

template <int G, typename F, typename S, typename E>
__device__ __forceinline__ void for_pixels(const WS& w, int rows, const uint32_t* __restrict__ tgt,
                                           const uint32_t* __restrict__ cur, int W, F f, S startrow, E endrow) {
    const int lane = threadIdx.x & 31, gi = lane / G, lg = lane % G, ng = 32 / G;
    for (int li = gi; li < rows; li += ng) {
        const short2 s = w.xr[li];
        const uint32_t* trow = tgt + (w.ylo + li) * W;
        const uint32_t* crow = cur + (w.ylo + li) * W;
        startrow(li);
        const int yy = w.ylo + li;
        for (int x = s.x + lg; x <= s.y; x += G) f(trow[x], crow[x], x, yy);
        endrow();   // widen the per-row int32 sums into int64 (keeps big images from overflowing)
    }
}

template <int G, bool QUAD>
__device__ __forceinline__ long long passes(const WS& w, int rows, const uint32_t* __restrict__ tgt,
                                            const uint32_t* __restrict__ cur, int W, int alpha, int col[3]) {
    const int ka = 0x101 * 255 / alpha;
    int rs = 0, gs = 0, bs = 0, cnt = 0;
    long long RS = 0, GS = 0, BS = 0, CN = 0;
    constexpr int NC = 24;
    uint16_t pcache[QUAD ? NC : 1]; int pidx = 0;
    for_pixels<G>(w, rows, tgt, cur, W, [&](uint32_t tt, uint32_t q, int x, int y) {
        if constexpr (QUAD) {
            const uint32_t pm = quad_cov16(w, x, y);
            if (pidx < NC) pcache[pidx] = (uint16_t)pm;
            pidx++;
            if (pm == 0) return;               // Go's computeColor only sees pixels that have a span
        }   // Go's computeColor only sees pixels that have a span
        int tr = tt & 255, tg = (tt >> 8) & 255, tb = (tt >> 16) & 255;
        int cr = q & 255, cg = (q >> 8) & 255, cb = (q >> 16) & 255;
        rs += (tr - cr) * ka + cr * 0x101;
        gs += (tg - cg) * ka + cg * 0x101;
        bs += (tb - cb) * ka + cb * 0x101;
        cnt++;
    }, [&](int) {}, [&] { RS += rs; GS += gs; BS += bs; CN += cnt; rs = gs = bs = cnt = 0; });
    const long long Rs = wsum(RS), Gs = wsum(GS), Bs = wsum(BS), Cnt = wsum(CN);
    if (Cnt == 0) { col[0] = col[1] = col[2] = 0; }
    else {
        col[0] = clampi((int)(Rs / Cnt) >> 8, 0, 255);
        col[1] = clampi((int)(Gs / Cnt) >> 8, 0, 255);
        col[2] = clampi((int)(Bs / Cnt) >> 8, 0, 255);
    }
    const uint32_t m = 0xffff;
    uint32_t sr = (uint32_t)col[0]; sr |= sr << 8; sr = sr * (uint32_t)alpha / 0xff;
    uint32_t sg = (uint32_t)col[1]; sg |= sg << 8; sg = sg * (uint32_t)alpha / 0xff;
    uint32_t sb = (uint32_t)col[2]; sb |= sb << 8; sb = sb * (uint32_t)alpha / 0xff;
    uint32_t sa = (uint32_t)alpha;  sa |= sa << 8;
    uint32_t ma = 0xffff, ba = (m - sa * ma / m) * 0x101;   // per-row coverage
    int d = 0; long long D64 = 0;
    pidx = 0;
    for_pixels<G>(w, rows, tgt, cur, W, [&](uint32_t tt, uint32_t q, int x, int y) {
        uint32_t pma = ma, pba = ba;
        if constexpr (QUAD) {
            pma = pidx < NC ? (uint32_t)pcache[pidx] : quad_cov16(w, x, y);
            pidx++;
            if (pma == 0) return;
            pba = (m - sa * pma / m) * 0x101;
        }
        int tr = tt & 255, tg = (tt >> 8) & 255, tb = (tt >> 16) & 255, ta = tt >> 24;
        uint32_t dr = q & 255, dg = (q >> 8) & 255, db = (q >> 16) & 255, da = q >> 24;
        int ar = (int)(uint8_t)((dr * pba + sr * pma) / m >> 8);
        int ag = (int)(uint8_t)((dg * pba + sg * pma) / m >> 8);
        int ab = (int)(uint8_t)((db * pba + sb * pma) / m >> 8);
        int aa = (int)(uint8_t)((da * pba + sa * pma) / m >> 8);
        int r1 = tr - (int)dr, g1 = tg - (int)dg, b1 = tb - (int)db, a1 = ta - (int)da;
        int r2 = tr - ar, g2 = tg - ag, b2 = tb - ab, a2 = ta - aa;
        d += (r2 * r2 + g2 * g2 + b2 * b2 + a2 * a2) - (r1 * r1 + g1 * g1 + b1 * b1 + a1 * a1);
    }, [&](int li) { ma = w.cov[li]; ba = (m - sa * ma / m) * 0x101; }, [&] { D64 += d; d = 0; });
    return wsum(D64);
}

constexpr long long BAD_DELTA = 1LL << 40;   // above any real delta (<= ~1e12): shapes too tall for the row buffer lose

// squared-error delta (all lanes) and closed-form colour; leaves the row table in w for the commit kernel
__device__ long long eval_warp(WS& w, const Cand& c, const uint32_t* __restrict__ tgt, const uint32_t* __restrict__ cur,
                               int W, int H, int col[3]) {
    const int lane = threadIdx.x & 31;
    Aux A; shape_prep(c, W, H, A);
    if (A.bad) { col[0] = col[1] = col[2] = 0; return BAD_DELTA; }
    const bool quad = c.t == T_QUAD;
    if (quad) {                                    // flatten the quadratic into 16 segments (shared by the warp)
        if (lane < 17) {
            const float t = lane * (1.0f / 16.0f), u = 1.0f - t;
            w.qx[lane] = u * u * c.p[0] + 2 * t * u * c.p[2] + t * t * c.p[4];
            w.qy[lane] = u * u * c.p[1] + 2 * t * u * c.p[3] + t * t * c.p[5];
        }
        __syncwarp();
        if (lane < 16) {
            const float ex = w.qx[lane + 1] - w.qx[lane], ey = w.qy[lane + 1] - w.qy[lane], l2 = ex * ex + ey * ey;
            const float ilen = l2 > 1e-12f ? rsqrtf(l2) : 0.0f;
            w.qex[lane] = ex; w.qey[lane] = ey; w.qinv[lane] = l2 > 1e-12f ? 1.0f / l2 : 0.0f;
            const float ux = fabsf(ex * ilen), uy = fabsf(ey * ilen);
            const float Amax = fmaxf(ux, uy), Bmin = fminf(ux, uy);
            w.qA[lane] = Amax < 1e-6f ? 1.0f : Amax; w.qB[lane] = Amax < 1e-6f ? 0.0f : Bmin;
        }
        __syncwarp();
    }
    long long len = 0;
    for (int k = lane; k < A.rows; k += 32) {
        int a, b, cv; shape_row(c, A, w, A.ylo + k, W, a, b, cv);
        w.xr[k] = make_short2((short)a, (short)b); w.cov[k] = (unsigned short)cv;
        if (a <= b) len += b - a + 1;
    }
    if (lane == 0) { w.ylo = A.ylo; w.rows = A.rows; }
    __syncwarp();
    const int rows = A.rows;
    const long long total = wsum(len);
    int G = 32, best = 1 << 30;
    {
        const int avg = rows > 0 ? (int)((total + rows - 1) / rows) : 1;
        for (int g = 32; g >= 8; g >>= 1) {
            int cost = ((rows * g + 31) / 32) * ((avg + g - 1) / g);
            if (cost < best) { best = cost; G = g; }
        }
    }
    long long d;
    if (G == 32)      d = (quad ? passes<32, true>(w, rows, tgt, cur, W, c.alpha, col) : passes<32, false>(w, rows, tgt, cur, W, c.alpha, col));
    else if (G == 16) d = (quad ? passes<16, true>(w, rows, tgt, cur, W, c.alpha, col) : passes<16, false>(w, rows, tgt, cur, W, c.alpha, col));
    else              d = (quad ? passes<8, true>(w, rows, tgt, cur, W, c.alpha, col) : passes<8, false>(w, rows, tgt, cur, W, c.alpha, col));
    __syncwarp();
    return d;
}

// ---------------- search kernels ----------------
constexpr long long KEY_OFF = 1LL << 41;   // |delta| < 2^41 even for 8K images (shape rows are capped)
__device__ __forceinline__ unsigned long long make_key(long long delta, int idx) {
    return ((unsigned long long)(delta + KEY_OFF) << 20) | (unsigned long long)idx;
}

struct State {                    // lives in device memory; kernels of one step share it
    Cand last;                     // most recently committed shape (start point for -rep climbs)
    int stop, nshapes;             // stop: a -rep climb found no improvement this step
    long long seedD[GROUPS];       // delta of `last` re-scored on the updated canvas
};

struct Dev {
    State* st;
    const uint32_t* tgt; uint32_t* cur;
    Cand* cands; unsigned long long* keys;
    Cand* climbC; long long* climbD;
    Rec* shapes; long long* total;
    int W, H, alpha, mode; bool mutAlpha;
    uint64_t seed;
};

__device__ __forceinline__ int pick_type(const Dev& D, uint64_t& rng) {
    static const int kinds[8] = {T_TRI, T_RECT, T_ELLIPSE, T_CIRCLE, T_ROTRECT, T_QUAD, T_ROTELL, T_POLY};
    return D.mode == 0 ? kinds[rint_(rng, 8)] : D.mode;
}

__global__ void gen_kernel(Dev D, int step) {
    __shared__ WS ws[4];
    __shared__ Cand cs[4];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, b = blockIdx.x * 4 + w;
    if (blockIdx.x == 0 && threadIdx.x == 0) D.st->stop = 0;
    if (lane == 0) {
        uint64_t rng = D.seed ^ ((uint64_t)step << 32) ^ ((uint64_t)b * 0x9E3779B97F4A7C15ull);
        sm64(rng);
        cs[w] = shape_random(pick_type(D, rng), D.mutAlpha ? 128 : D.alpha, rng, D.W, D.H, D.mutAlpha);
        D.cands[b] = cs[w];
    }
    __syncwarp();
    int col[3];
    const long long d = eval_warp(ws[w], cs[w], D.tgt, D.cur, D.W, D.H, col);
    if (lane == 0) atomicMin(&D.keys[b / PER_GROUP], make_key(d, b % PER_GROUP));
}

// Async hill-climb: no per-round barrier. Each warp snapshots the shared best (seqlock), mutates it, scores it and
// commits under a small lock if it improves. Ends after MAX_FAILS consecutive non-improving evals across all warps.
__global__ void climb_kernel(Dev D, int step, bool rep) {
    if (rep && D.st->stop) return;     // uniform across the grid
    extern __shared__ __align__(16) unsigned char dsm[];
    WS* ws = (WS*)dsm;
    __shared__ Cand best, cw[KW];
    __shared__ long long bestD;
    __shared__ volatile int ver;
    __shared__ int fails, lockv;
    const int g = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint64_t rng = D.seed ^ ((uint64_t)step << 40) ^ ((uint64_t)(g * 64 + w + 7777) * 0xD1B54A32D192ED03ull);
    if (threadIdx.x == 0) {
        if (!rep) {
            unsigned long long k = D.keys[g];
            best = D.cands[g * PER_GROUP + (int)(k & 0xfffff)];
            bestD = (long long)(k >> 20) - KEY_OFF;
        }
        ver = 0; fails = 0; lockv = 0;
    }
    if (rep && w == 0) {               // score the previous shape on the updated canvas, then climb from it
        if (lane == 0) cw[0] = D.st->last;
        __syncwarp();
        int col0[3];
        const long long d0 = eval_warp(ws[0], cw[0], D.tgt, D.cur, D.W, D.H, col0);
        if (lane == 0) { best = D.st->last; bestD = d0; D.st->seedD[g] = d0; }
    }
    __syncthreads();
    for (;;) {
        int f = 0;
        if (lane == 0) f = *(volatile int*)&fails;
        f = __shfl_sync(0xffffffffu, f, 0);
        if (f >= MAX_FAILS) break;
        if (lane == 0) {
            Cand t; int v1, v2;
            do {
                v1 = ver;
                const volatile int* b = (const volatile int*)&best;
                t.t = b[0]; t.alpha = b[1];
                for (int i = 0; i < 8; i++) t.p[i] = __int_as_float(b[2 + i]);
                __threadfence_block();
                v2 = ver;
            } while ((v1 & 1) || v1 != v2);
            shape_mutate(t, rng, D.W, D.H, D.mutAlpha);
            cw[w] = t;
        }
        __syncwarp();
        int col[3];
        const long long d = eval_warp(ws[w], cw[w], D.tgt, D.cur, D.W, D.H, col);
        if (lane == 0) {
            bool improved = false;
            if (d < *(volatile long long*)&bestD) {
                while (atomicCAS(&lockv, 0, 1) != 0) {}
                if (d < bestD) {
                    ver = ver + 1; __threadfence_block();
                    best = cw[w]; bestD = d;
                    __threadfence_block(); ver = ver + 1;
                    atomicExch(&fails, 0);
                    improved = true;
                }
                atomicExch(&lockv, 0);
            }
            if (!improved) atomicAdd(&fails, 1);
        }
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x == 0) { D.climbC[g] = best; D.climbD[g] = bestD; }
}

__global__ void commit_kernel(Dev D, int step, bool rep) {
    __shared__ WS ws;
    __shared__ Cand c;
    __shared__ int skip, slot;
    const int lane = threadIdx.x;
    if (lane == 0) {
        skip = 0; slot = 0;
        if (rep && D.st->stop) skip = 1;
        int bg = 0;
        for (int g = 1; g < GROUPS; g++) if (D.climbD[g] < D.climbD[bg]) bg = g;
        if (!skip && rep && D.climbD[bg] >= D.st->seedD[0]) { D.st->stop = 1; skip = 1; }   // Go: no change => stop
        if (!skip) { c = D.climbC[bg]; slot = D.st->nshapes++; }
        if (!rep) for (int g = 0; g < GROUPS; g++) D.keys[g] = ~0ull;
    }
    __syncwarp();
    if (skip) return;
    int col[3];
    const long long d = eval_warp(ws, c, D.tgt, D.cur, D.W, D.H, col);
    const uint32_t m = 0xffff;
    uint32_t sr = (uint32_t)col[0]; sr |= sr << 8; sr = sr * (uint32_t)c.alpha / 0xff;
    uint32_t sg = (uint32_t)col[1]; sg |= sg << 8; sg = sg * (uint32_t)c.alpha / 0xff;
    uint32_t sb = (uint32_t)col[2]; sb |= sb << 8; sb = sb * (uint32_t)c.alpha / 0xff;
    uint32_t sa = (uint32_t)c.alpha; sa |= sa << 8;
    for (int li = 0; li < ws.rows; li++) {
        const short2 s = ws.xr[li];
        const uint32_t ma = ws.cov[li], ba = (m - sa * ma / m) * 0x101;
        uint32_t* row = D.cur + (ws.ylo + li) * D.W;
        for (int x = s.x + lane; x <= s.y; x += 32) {
            uint32_t pma = ma, pba = ba;
            if (c.t == T_QUAD) { pma = quad_cov16(ws, x, ws.ylo + li); if (pma == 0) continue; pba = (m - sa * pma / m) * 0x101; }
            uint32_t q = row[x];
            uint32_t dr = q & 255, dg = (q >> 8) & 255, db = (q >> 16) & 255, da = q >> 24;
            uint32_t ar = (uint8_t)((dr * pba + sr * pma) / m >> 8);
            uint32_t ag = (uint8_t)((dg * pba + sg * pma) / m >> 8);
            uint32_t ab = (uint8_t)((db * pba + sb * pma) / m >> 8);
            uint32_t aa = (uint8_t)((da * pba + sa * pma) / m >> 8);
            row[x] = ar | (ag << 8) | (ab << 16) | (aa << 24);
        }
    }
    if (lane == 0) {
        *D.total += d;
        Rec r; r.t = c.t; r.alpha = c.alpha; for (int i = 0; i < 8; i++) r.p[i] = c.p[i];
        r.r = col[0]; r.g = col[1]; r.b = col[2]; r.step = step;
        D.shapes[slot] = r;
        D.st->last = c;
    }
}

// ---------------- host ----------------
static bool read_img(const char* path, int& W, int& H, std::vector<uint32_t>& px) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int hdr[2];
    if (fread(hdr, 4, 2, f) != 2) { fclose(f); return false; }
    W = hdr[0]; H = hdr[1];
    px.resize((size_t)W * H);
    bool ok = fread(px.data(), 4, px.size(), f) == px.size();
    fclose(f);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: gpu_search target.bin canvas.bin out.bin count mode alpha [seed]\n"); return 1; }
    int W, H, W2, H2; std::vector<uint32_t> tgt, cur;
    if (!read_img(argv[1], W, H, tgt) || !read_img(argv[2], W2, H2, cur) || W != W2 || H != H2) { fprintf(stderr, "bad input files\n"); return 1; }
    const int count = atoi(argv[4]), mode = atoi(argv[5]), alpha = atoi(argv[6]);
    const uint64_t seed = argc > 7 ? strtoull(argv[7], 0, 10) : 1;
    const int nrep = argc > 8 ? atoi(argv[8]) : 0;
    if (!(mode == 0 || mode == 1 || mode == 2 || mode == 3 || mode == 4 || mode == 5 || mode == 6 || mode == 7 || mode == 8)) { fprintf(stderr, "unsupported mode %d\n", mode); return 3; }
    if (alpha < 0 || alpha > 255) { fprintf(stderr, "alpha must be 0..255\n"); return 1; }

    long long total0 = 0;
    for (size_t i = 0; i < tgt.size(); i++)
        for (int s = 0; s < 32; s += 8) { int d = (int)((tgt[i] >> s) & 255) - (int)((cur[i] >> s) & 255); total0 += d * d; }

    Dev D{}; D.W = W; D.H = H; D.alpha = alpha == 0 ? 128 : alpha; D.mutAlpha = alpha == 0; D.mode = mode; D.seed = seed;
    uint32_t *dT, *dQ; CK(cudaMalloc(&dT, tgt.size() * 4)); CK(cudaMalloc(&dQ, cur.size() * 4));
    CK(cudaMalloc(&D.cands, GROUPS * PER_GROUP * sizeof(Cand))); CK(cudaMalloc(&D.keys, GROUPS * 8));
    CK(cudaMalloc(&D.climbC, GROUPS * sizeof(Cand))); CK(cudaMalloc(&D.climbD, GROUPS * 8));
    CK(cudaMalloc(&D.shapes, (size_t)count * (1 + nrep) * sizeof(Rec))); CK(cudaMalloc(&D.total, 8));
    CK(cudaMalloc(&D.st, sizeof(State))); CK(cudaMemset(D.st, 0, sizeof(State)));
    CK(cudaMemcpy(dT, tgt.data(), tgt.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dQ, cur.data(), cur.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.total, &total0, 8, cudaMemcpyHostToDevice));
    std::vector<unsigned long long> k(GROUPS, ~0ull);
    CK(cudaMemcpy(D.keys, k.data(), GROUPS * 8, cudaMemcpyHostToDevice));
    D.tgt = dT; D.cur = dQ;
    const int smem = KW * (int)sizeof(WS);
    CK(cudaFuncSetAttribute(climb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int s = 0; s < count; s++) {
        gen_kernel<<<GROUPS * PER_GROUP / 4, 128>>>(D, s);
        climb_kernel<<<GROUPS, 32 * KW, smem>>>(D, s, false);
        commit_kernel<<<1, 32>>>(D, s, false);
        for (int r = 0; r < nrep; r++) {   // -rep: extra shapes per step, climbing from the last one
            climb_kernel<<<GROUPS, 32 * KW, smem>>>(D, s, true);
            commit_kernel<<<1, 32>>>(D, s, true);
        }
    }
    cudaEventRecord(e1);
    CK(cudaGetLastError()); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1);

    State hst; CK(cudaMemcpy(&hst, D.st, sizeof(State), cudaMemcpyDeviceToHost));
    const int n = hst.nshapes;
    std::vector<Rec> shapes(n); long long total;
    CK(cudaMemcpy(shapes.data(), D.shapes, (size_t)n * sizeof(Rec), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&total, D.total, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cur.data(), dQ, cur.size() * 4, cudaMemcpyDeviceToHost));
    long long full = 0;
    for (size_t i = 0; i < tgt.size(); i++)
        for (int s = 0; s < 32; s += 8) { int d = (int)((tgt[i] >> s) & 255) - (int)((cur[i] >> s) & 255); full += d * d; }
    FILE* o = fopen(argv[3], "wb");
    if (!o) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    fwrite(&n, 4, 1, o); fwrite(shapes.data(), sizeof(Rec), n, o); fclose(o);
    printf("gpu_search: %dx%d mode=%d alpha=%d  %d shapes in %.1f ms (%.2f ms/shape)  score %.6f  bookkeeping %s\n", W, H, mode, alpha,
           n, ms, ms / n, std::sqrt((double)total / (W * H * 4.0)) / 255, full == total ? "exact" : "MISMATCH");
    return full == total ? 0 : 4;
}
