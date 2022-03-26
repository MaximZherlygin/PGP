#include <iostream>
#include <stdio.h>
#include <float.h>
#include <vector>
#include <string>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES // for C++

#include <cmath>

#define CSC(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return(1); \
    } \
} \

typedef unsigned char uchar;

struct vec3 {
    double x;
    double y;
    double z;
};

struct triangle {
    vec3 p1;
    vec3 p2;
    vec3 p3;
    vec3 color;
};

bool replace(std::string &str, const std::string &from, const std::string &to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}


__host__ __device__  vec3 operator+(vec3 v1, vec3 v2) {
    return vec3{v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

__host__ __device__  vec3 operator-(vec3 v1, vec3 v2) {
    return vec3{v1.x - v2.x,
                v1.y - v2.y,
                v1.z - v2.z};
}

__host__ __device__  vec3 operator*(vec3 v, double num) {
    return vec3{v.x * num,
                v.y * num,
                v.z * num};
}

__host__ __device__  double scal_mul(vec3 v1, vec3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__  double len(vec3 v) {
    return sqrt(scal_mul(v, v));
}

__host__ __device__  vec3 norm(vec3 v) {
    double num = len(v);
    return vec3{v.x / num, v.y / num, v.z / num};
}

__host__ __device__ vec3 crossing(vec3 v1, vec3 v2) {
    return {v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x};
}

__host__ __device__ vec3 multiply(vec3 a, vec3 b, vec3 c, vec3 v) {
    return {a.x * v.x + b.x * v.y + c.x * v.z,
            a.y * v.x + b.y * v.y + c.y * v.z,
            a.z * v.x + b.z * v.y + c.z * v.z};
}

vec3 normalise_color(vec3 color) {
    return {color.x * 255.,
            color.y * 255.,
            color.z * 255.};
}

__host__ __device__ uchar4 ray_aux(vec3 pos, vec3 dir, vec3 light_pos,
                                   vec3 light_color, triangle *trigs, int n) {
    int min_value = -1;
    double ts_min;
    for (int i = 0; i < n; ++i) {
        vec3 e1 = trigs[i].p2 - trigs[i].p1;
        vec3 e2 = trigs[i].p3 - trigs[i].p1;
        vec3 p = crossing(dir, e2);
        double div = scal_mul(p, e1);

        if (fabs(div) < 1e-10)
            continue;

        vec3 t = pos - trigs[i].p1;
        double u = scal_mul(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;

        vec3 q = crossing(t, e1);
        double v = scal_mul(q, dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;

        double ts = scal_mul(q, e2) / div;
        if (ts < 0.0)
            continue;

        if (min_value == -1 || ts < ts_min) {
            min_value = i;
            ts_min = ts;
        }
    }

    if (min_value == -1)
        return {0, 0, 0, 0};

    pos = dir * ts_min + pos;
    dir = light_pos - pos;
    double length = len(dir);
    dir = norm(dir);

    for (int i = 0; i < n; i++) {
        vec3 e1 = trigs[i].p2 - trigs[i].p1;
        vec3 e2 = trigs[i].p3 - trigs[i].p1;
        vec3 p = crossing(dir, e2);
        double div = scal_mul(p, e1);

        if (fabs(div) < 1e-10)
            continue;

        vec3 t = pos - trigs[i].p1;
        double u = scal_mul(p, t) / div;

        if (u < 0.0 || u > 1.0)
            continue;

        vec3 q = crossing(t, e1);
        double v = scal_mul(q, dir) / div;

        if (v < 0.0 || v + u > 1.0)
            continue;

        double ts = scal_mul(q, e2) / div;

        if (ts > 0.0 && ts < length && i != min_value) {
            return {0, 0, 0, 0};
        }
    }

    uchar4 color_min;
    color_min.x = trigs[min_value].color.x;
    color_min.y = trigs[min_value].color.y;
    color_min.z = trigs[min_value].color.z;

    color_min.x *= light_color.x;
    color_min.y *= light_color.y;
    color_min.z *= light_color.z;
    color_min.w = 0;
    return color_min;
}

void render_cpu(vec3 p_c, vec3 p_v, int w, int h, double fov, uchar4 *pixels, vec3 light_pos,
                vec3 light_col, triangle *trigs, int n) {
    double dw = (double) 2.0 / (double) (w - 1.0);
    double dh = (double) 2.0 / (double) (h - 1.0);
    double z = 1.0 / tan(fov * M_PI / 360.0);
    vec3 b_z = norm(p_v - p_c);
    vec3 b_x = norm(crossing(b_z, {0.0, 0.0, 1.0}));
    vec3 b_y = norm(crossing(b_x, b_z));
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            vec3 v;
            v.x = (double) -1.0 + dw * (double) i;
            v.y = ((double) -1.0 + dh * (double) j) * (double) h / (double) w;
            v.z = z;
            vec3 dir = multiply(b_x, b_y, b_z, v);
            pixels[(h - 1 - j) * w + i] = ray_aux(p_c, norm(dir), light_pos, light_col, trigs, n);
        }
}

__global__ void render_gpu(vec3 p_c, vec3 p_v, int w, int h, double fov, uchar4 *pixels,
                           vec3 light_pos, vec3 light_col, triangle *trigs, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    double dw = (double) 2.0 / (double) (w - 1.0);
    double dh = (double) 2.0 / (double) (h - 1.0);
    double z = 1.0 / tan(fov * M_PI / 360.0);
    vec3 b_z = norm(p_v - p_c);
    vec3 b_x = norm(crossing(b_z, {0.0, 0.0, 1.0}));
    vec3 b_y = norm(crossing(b_x, b_z));
    for (int i = idx; i < w; i += offsetX)
        for (int j = idy; j < h; j += offsetY) {
            vec3 v;
            v.x = (double) -1.0 + dw * (double) i;
            v.y = ((double) -1.0 + dh * (double) j) * (double) h / (double) w;
            v.z = z;
            vec3 dir = multiply(b_x, b_y, b_z, v);
            pixels[(h - 1 - j) * w + i] = ray_aux(p_c, norm(dir), light_pos, light_col, trigs, n);
        }
}

void ssaa_cpu(uchar4 *pixels, int w, int h, int coeff, uchar4 *ssaa_pixels) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int4 mid_pixel = {0, 0, 0, 0};
            for (int j = 0; j < coeff; j++) {
                for (int i = 0; i < coeff; i++) {
                    int index = y * w * coeff * coeff + x * coeff + j * w * coeff + i;
                    mid_pixel.x += ssaa_pixels[index].x;
                    mid_pixel.y += ssaa_pixels[index].y;
                    mid_pixel.z += ssaa_pixels[index].z;
                    mid_pixel.w += 0;
                }
            }
            pixels[y * w + x].x = (uchar) (int) (mid_pixel.x / (coeff * coeff));
            pixels[y * w + x].y = (uchar) (int) (mid_pixel.y / (coeff * coeff));
            pixels[y * w + x].z = (uchar) (int) (mid_pixel.z / (coeff * coeff));
            pixels[y * w + x].w = 0;
        }
    }
}

__global__ void ssaa_gpu(uchar4 *pixels, int w, int h, int coeff, uchar4 *ssaa_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsetY) {
        for (int x = idx; x < w; x += offsetX) {
            int4 mid = {0, 0, 0, 0};
            for (int j = 0; j < coeff; j++) {
                for (int i = 0; i < coeff; i++) {
                    int index = y * w * coeff * coeff + x * coeff + j * w * coeff + i;
                    mid.x += ssaa_pixels[index].x;
                    mid.y += ssaa_pixels[index].y;
                    mid.z += ssaa_pixels[index].z;
                    mid.w += 0;
                }
            }
            pixels[y * w + x].x = (uchar) (mid.x / (coeff * coeff));
            pixels[y * w + x].y = (uchar) (mid.y / (coeff * coeff));
            pixels[y * w + x].z = (uchar) (mid.z / (coeff * coeff));
            pixels[y * w + x].w = 0;
        }
    }
}

void hexahedron(const vec3 c_coords, const double radius, const vec3 colors, std::vector<triangle>& trigs) { // ++++++

    std::vector <vec3> points {{-1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)},
                               {-1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)},
                               {-1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)},
                               {-1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)},
                               {1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)},
                               {1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)},
                               {1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)},
                               {1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)}};

    for (int i = 0; i < points.size(); i++) {
        points[i].x *= radius;
        points[i].x += c_coords.x;

        points[i].y *= radius;
        points[i].y += c_coords.y;

        points[i].z *= radius;
        points[i].z += c_coords.z;
    }

    trigs.push_back({points[0], points[1], points[3], colors});
    trigs.push_back({points[0], points[2], points[3], colors});
    trigs.push_back({points[1], points[5], points[7], colors});
    trigs.push_back({points[1], points[3], points[7], colors});
    trigs.push_back({points[4], points[5], points[7], colors});
    trigs.push_back({points[4], points[6], points[7], colors});
    trigs.push_back({points[0], points[4], points[6], colors});
    trigs.push_back({points[0], points[2], points[6], colors});
    trigs.push_back({points[0], points[1], points[5], colors});
    trigs.push_back({points[0], points[4], points[5], colors});
    trigs.push_back({points[2], points[3], points[7], colors});
    trigs.push_back({points[2], points[6], points[7], colors});
}

void icosahedron(vec3 c_coords, double radius, const vec3 colors, std::vector<triangle> &trigs) {
    double atrtan_1_2 = 26.565; // arctan(1/2) ~ +-26.57
    double angle = M_PI * atrtan_1_2 / 180;
    double segment_angle = M_PI * 72 / 180;
    double current_angle = 0.0;

    std::vector <vec3> points(12);
    points[0] = {0, radius, 0};
    points[11] = {0, -radius, 0};

    for (int i = 1; i < 6; i++) {
        points[i] = {sin(current_angle) * cos(angle), sin(angle), cos(current_angle) * cos(angle)};
        current_angle += segment_angle;
    }

    current_angle = M_PI * 36 / 180;

    for (int i = 6; i < 11; i++) {
        points[i] = {sin(current_angle) * cos(-angle), sin(-angle), cos(current_angle) * cos(-angle)};
        current_angle += segment_angle;
    }

    for (int i = 0; i < points.size(); i++) {
        points[i].x *= radius;
        points[i].x += c_coords.x;

        points[i].y *= radius;
        points[i].y += c_coords.y;

        points[i].z *= radius;
        points[i].z += c_coords.z;
    }

    trigs.push_back({points[0], points[1], points[2], colors});
    trigs.push_back({points[0], points[2], points[3], colors});
    trigs.push_back({points[0], points[3], points[4], colors});
    trigs.push_back({points[0], points[4], points[5], colors});
    trigs.push_back({points[0], points[5], points[1], colors});

    trigs.push_back({points[1], points[5], points[10], colors});
    trigs.push_back({points[2], points[1], points[6], colors});
    trigs.push_back({points[3], points[2], points[7], colors});
    trigs.push_back({points[4], points[3], points[8], colors});
    trigs.push_back({points[5], points[4], points[9], colors});

    trigs.push_back({points[6], points[7], points[2], colors});
    trigs.push_back({points[7], points[8], points[3], colors});
    trigs.push_back({points[8], points[9], points[4], colors});
    trigs.push_back({points[9], points[10], points[5], colors});
    trigs.push_back({points[10], points[6], points[1], colors});

    trigs.push_back({points[11], points[7], points[6], colors});
    trigs.push_back({points[11], points[8], points[7], colors});
    trigs.push_back({points[11], points[9], points[8], colors});
    trigs.push_back({points[11], points[10], points[9], colors});
    trigs.push_back({points[11], points[6], points[10], colors});
}

void dodecahedron(const vec3 c_coords, const double radius, const vec3 colors, std::vector<triangle>& trigs) { // +++++++
    std::vector<vec3> points {{-(2 / (1 + sqrt(5))) / sqrt(3), 0, ((1 + sqrt(5)) / 2) / sqrt(3)},
                              {(2 / (1 + sqrt(5))) / sqrt(3),  0, ((1 + sqrt(5)) / 2) / sqrt(3)},
                              {-1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)},
                              {1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)},
                              {1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)},
                              {-1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)},
                              {0, -((1 + sqrt(5)) / 2) / sqrt(3), (2 / (1 + sqrt(5))) / sqrt(3)},
                              {0, ((1 + sqrt(5)) / 2) / sqrt(3), (2 / (1 + sqrt(5))) / sqrt(3)},
                              {-((1 + sqrt(5)) / 2) / sqrt(3), -(2 / (1 + sqrt(5))) / sqrt(3), 0},
                              {-((1 + sqrt(5)) / 2) / sqrt(3), (2 / (1 + sqrt(5))) / sqrt(3), 0},
                              {((1 + sqrt(5)) / 2) / sqrt(3), (2 / (1 + sqrt(5))) / sqrt(3), 0},
                              {((1 + sqrt(5)) / 2) / sqrt(3), -(2 / (1 + sqrt(5))) / sqrt(3), 0},
                              {0, -((1 + sqrt(5)) / 2) / sqrt(3), -(2 / (1 + sqrt(5))) / sqrt(3)},
                              {0, ((1 + sqrt(5)) / 2) / sqrt(3), -(2 / (1 + sqrt(5))) / sqrt(3)},
                              {1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)},
                              {1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)},
                              {-1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)},
                              {-1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)},
                              {(2 / (1 + sqrt(5))) / sqrt(3), 0, -((1 + sqrt(5)) / 2) / sqrt(3)},
                              {-(2 / (1 + sqrt(5))) / sqrt(3), 0, -((1 + sqrt(5)) / 2) / sqrt(3)}};

    for (int i = 0; i < points.size(); i++) {
        points[i].x *= radius;
        points[i].x += c_coords.x;

        points[i].y *= radius;
        points[i].y += c_coords.y;

        points[i].y *= radius;
        points[i].y += c_coords.y;
    }

    trigs.push_back({points[4], points[0], points[6], colors});
    trigs.push_back({points[4], points[1], points[0], colors});
    trigs.push_back({points[0], points[5], points[6], colors});

    trigs.push_back({points[7], points[0], points[3], colors});
    trigs.push_back({points[7], points[2], points[0], colors});
    trigs.push_back({points[0], points[1], points[3], colors});

    trigs.push_back({points[10], points[1], points[11], colors});
    trigs.push_back({points[10], points[3], points[1], colors});
    trigs.push_back({points[1], points[4], points[11], colors});

    trigs.push_back({points[8], points[0], points[9], colors});
    trigs.push_back({points[8], points[5], points[0], colors});
    trigs.push_back({points[0], points[2], points[9], colors});

    trigs.push_back({points[12], points[5], points[16], colors});
    trigs.push_back({points[12], points[6], points[5], colors});
    trigs.push_back({points[5], points[8], points[16], colors});

    trigs.push_back({points[4], points[6], points[12], colors});
    trigs.push_back({points[15], points[4], points[12], colors});
    trigs.push_back({points[15], points[11], points[4], colors});

    trigs.push_back({points[2], points[7], points[13], colors});
    trigs.push_back({points[17], points[2], points[13], colors});
    trigs.push_back({points[17], points[9], points[2], colors});

    trigs.push_back({points[3], points[10], points[14], colors});
    trigs.push_back({points[13], points[3], points[14], colors});
    trigs.push_back({points[13], points[7], points[3], colors});

    trigs.push_back({points[8], points[9], points[17], colors});
    trigs.push_back({points[19], points[8], points[17], colors});
    trigs.push_back({points[19], points[16], points[8], colors});

    trigs.push_back({points[11], points[15], points[18], colors});
    trigs.push_back({points[14], points[11], points[18], colors});
    trigs.push_back({points[14], points[10], points[11], colors});

    trigs.push_back({points[12], points[16], points[19], colors});
    trigs.push_back({points[18], points[12], points[19], colors});
    trigs.push_back({points[18], points[15], points[12], colors});

    trigs.push_back({points[13], points[14], points[18], colors});
    trigs.push_back({points[19], points[13], points[18], colors});
    trigs.push_back({points[19], points[17], points[13], colors});

    // std::cout << "Creating dodecahedron done\n";
}

void scene(vec3 a, vec3 b, vec3 c, vec3 d, vec3 color,
           std::vector <triangle> &trigs) {
    // std::cout << "Creating scene\n";
    color = normalise_color(color);
    trigs.push_back(triangle{a, b, c, color});
    trigs.push_back(triangle{c, d, a, color});
}

int cpu_mode(vec3 p_c, vec3 p_v, int w, int ssaa_w, int h, int ssaa_h, double fov, uchar4 *pixels,
             uchar4 *pixels_ssaa, vec3 light_pos, vec3 light_col, triangle *trigs, int n, int ssaa_multiplier) {
    render_cpu(p_c, p_v, ssaa_w, ssaa_h, fov, pixels_ssaa, light_pos, light_col, trigs, n);
    ssaa_cpu(pixels, w, h, ssaa_multiplier, pixels_ssaa);

    return 0;
}

int gpu_mode(vec3 p_c, vec3 p_v, int w, int ssaa_w, int h, int ssaa_h, double fov, uchar4 *pixels,
             uchar4 *pixels_ssaa, vec3 light_pos, vec3 light_col, triangle *trigs, int n, int ssaa_multiplier) {
//    cerr << "Allocate pixels\n";
    // Allocating on gpu
    uchar4 *gpu_pixels;
    CSC(cudaMalloc((uchar4 * *)(&gpu_pixels), w * h * sizeof(uchar4)));
    CSC(cudaMemcpy(gpu_pixels, pixels, w * h * sizeof(uchar4), cudaMemcpyHostToDevice));
//    cerr << "Allocate ssaa pixels\n";
    uchar4 *gpu_pixels_ssaa;
    CSC(cudaMalloc((uchar4 * *)(&gpu_pixels_ssaa), ssaa_w * ssaa_h * sizeof(uchar4)));
    CSC(cudaMemcpy(gpu_pixels_ssaa, pixels_ssaa, ssaa_w * ssaa_h * sizeof(uchar4), cudaMemcpyHostToDevice));
//    cerr << "Allocate trigs\n";
    triangle *gpu_trigs;
    CSC(cudaMalloc((triangle **) (&gpu_trigs), n * sizeof(triangle)));
    CSC(cudaMemcpy(gpu_trigs, trigs, n * sizeof(triangle), cudaMemcpyHostToDevice));
//    cerr << "Start render\n";
    // Rendering
    render_gpu <<< 128, 128 >>>(p_c, p_v, ssaa_w, ssaa_h, fov, gpu_pixels_ssaa, light_pos, light_col, gpu_trigs, n);
    cudaThreadSynchronize();
    CSC(cudaGetLastError());
//    cerr << "Start ssaa\n";
    // Ssaa smoothing algo
    ssaa_gpu <<< 128, 128 >>>(gpu_pixels, w, h, ssaa_multiplier, gpu_pixels_ssaa);
    cudaThreadSynchronize();
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(pixels, gpu_pixels, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));

    // Free memory
    CSC(cudaFree(gpu_pixels));
    CSC(cudaFree(gpu_pixels_ssaa));
    CSC(cudaFree(gpu_trigs));

    return 0;
}

int main(int argc, char *argv[]) {
    std::string arg;
    if (argv[1]) arg = argv[1];

    if (arg == "--default") {
        std::cout << "100" << "\n";
        std::cout << "./out" << "\n";
        std::cout << "640 480 120" << "\n";
        std::cout << "7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0" << "\n";
        std::cout << "2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0" << "\n";
        std::cout << "4.0 4.0 0.0 1.0 0.0 1.0 1.0 0.0 0.0 0.0" << "\n";
        std::cout << "1.0 1.0 0.0 1.0 1.0 0.0 1.5 0.0 0.0 0.0" << "\n";
        std::cout << "-2.5 -2.5 0.0 0.0 1.0 1.0 1.75 0.0 0.0 0.0" << "\n";
        std::cout << "-10.0 -10.0 -1.0 -10.0 10.0 -1.0 10.0 10.0 -1.0 10.0 -10.0 -1.0 temp 0.0 0.9 0.0 0.5" << "\n";
        std::cout << "1" << "\n";
        std::cout << "100 100 100 1.0 1.0 1.0" << "\n";
        std::cout << "1 3" << "\n";
        return 0;
    }

    bool is_gpu = argc == 1 || arg == "--gpu";

    std::string temp;

    // Frames
    int frames_count;
    std::cin >> frames_count;
    std::string path_to_frames;
    std::cin >> path_to_frames;
    int w;
    int h;
    int view_angle;
    std::cin >> w >> h >> view_angle;

    // Camera trajectory
    double r0c;
    double z0c;
    double phi0c;
    std::cin >> r0c >> z0c >> phi0c;
    double Arc;
    double Azc;
    std::cin >> Arc >> Azc;
    double omegaRc;
    double omegaZc;
    double omegaPhiC;
    std::cin >> omegaRc >> omegaZc >> omegaPhiC;
    double pRc;
    double pZc;
    std::cin >> pRc >> pZc;

    double r0v;
    double z0v;
    double phi0v;
    std::cin >> r0v >> z0v >> phi0v;
    double Arv;
    double Azv;
    std::cin >> Arv >> Azv;
    double omegaRv;
    double omegaZv;
    double omegaPhiV;
    std::cin >> omegaRv >> omegaZv >> omegaPhiV;
    double pRv;
    double pZv;
    std::cin >> pRv >> pZv;


    // Figures params with creating
    vec3 c_verticles;
    vec3 col_values;
    double radius;
    std::vector <triangle> trigs;
    std::cin >> c_verticles.x >> c_verticles.y >> c_verticles.z >> col_values.x >> col_values.y >> col_values.z
             >> radius >> temp >> temp >> temp;
    col_values.x *= 255.0;
    col_values.y *= 255.0;
    col_values.z *= 255.0;
    hexahedron(c_verticles, radius, col_values, trigs);
    std::cin >> c_verticles.x >> c_verticles.y >> c_verticles.z >> col_values.x >> col_values.y >> col_values.z
             >> radius >> temp >> temp >> temp;
    col_values.x *= 255.0;
    col_values.y *= 255.0;
    col_values.z *= 255.0;
    icosahedron(c_verticles, radius, col_values, trigs);
    std::cin >> c_verticles.x >> c_verticles.y >> c_verticles.z >> col_values.x >> col_values.y >> col_values.z
             >> radius >> temp >> temp >> temp;
    col_values.x *= 255.0;
    col_values.y *= 255.0;
    col_values.z *= 255.0;
    dodecahedron(c_verticles, radius, col_values, trigs);

    // Scene
    vec3 floor_first_point;
    vec3 floor_second_point;
    vec3 floor_third_point;
    vec3 floor_fourth_point;
    std::cin >> floor_first_point.x >> floor_first_point.y >> floor_first_point.z;
    std::cin >> floor_second_point.x >> floor_second_point.y >> floor_second_point.z;
    std::cin >> floor_third_point.x >> floor_third_point.y >> floor_third_point.z;
    std::cin >> floor_fourth_point.x >> floor_fourth_point.y >> floor_fourth_point.z;
    std::cin >> temp >> col_values.x >> col_values.y >> col_values.z >> temp;
    scene(floor_first_point, floor_second_point, floor_third_point, floor_fourth_point, col_values, trigs);

    // Lights
    int n_lights;
    vec3 light_pos;
    vec3 light_col;
    std::cin >> n_lights;
    std::cin >> light_pos.x >> light_pos.y >> light_pos.z >> light_col.x >> light_col.y >> light_col.z;

    int rec; // Should be 1 (unused)
    int rays_sqrt;
    std::cin >> rec >> rays_sqrt;

    int sum_of_rays;

    triangle *trigs_as_array;
    trigs_as_array = trigs.data();
    uchar4 *pixels = new uchar4[w * h * rays_sqrt * rays_sqrt];
    uchar4 *pixels_ssaa = new uchar4[w * h * rays_sqrt * rays_sqrt];

    std::cout << "Polygons count: " << trigs.size() << "\n";
    std::cout << "Image size: " << w << " " << h << "\n";
    std::cout << "Frames summ: " << frames_count << "\n";

    double total_duration_time = 0;

    for (int i = 0; i < frames_count; i++) {
        auto time_start = std::chrono::steady_clock::now();
        double t = i * 2.0 * M_PI / frames_count;

        // Movement
        double r_c = r0c + Arc * sin(omegaRc * t + pRc);
        double z_c = z0c + Azc * sin(omegaZc * t + pZc);
        double phi_c = phi0c + omegaPhiC * t;

        double r_v = r0v + Arv * sin(omegaRv * t + pRv);
        double z_v = z0v + Azv * sin(omegaZv * t + pZv);
        double phi_v = phi0v + omegaPhiV * t;

        vec3 p_c = {r_c * cos(phi_c),
                    r_c * sin(phi_c),
                    z_c};
        vec3 p_v = {r_v * cos(phi_v),
                    r_v * sin(phi_v),
                    z_v};

        // Total sum of rays (will be the same coz of recursion)
        sum_of_rays = w * h * rays_sqrt * rays_sqrt;
        int p_size = trigs.size();
        int ssaa_h = h * rays_sqrt;

        if (is_gpu)
            gpu_mode(p_c, p_v, w, w * rays_sqrt, h, ssaa_h, (double) view_angle, pixels, pixels_ssaa,
                           light_pos, light_col, trigs_as_array, p_size, rays_sqrt);
        else
            cpu_mode(p_c, p_v, w, w * rays_sqrt, h, ssaa_h, (double) view_angle, pixels, pixels_ssaa,
                           light_pos, light_col, trigs_as_array, p_size, rays_sqrt);

        auto end = std::chrono::steady_clock::now();

        double summ_time = std::chrono::duration_cast<std::chrono::microseconds>(end - time_start).count() / 1000;
        total_duration_time += summ_time;
        std::cout << i + 1 << "/" << frames_count << "\t" << summ_time << "\t" << sum_of_rays << "\n";

        std::string iter = std::to_string(i);
        std::string filename = path_to_frames;
        replace(filename, "%d", iter);
        FILE *fp = fopen(filename.c_str(), "wb");
        fwrite(&w, sizeof(int), 1, fp); // тут падает
        fwrite(&h, sizeof(int), 1, fp);
        fwrite(pixels, sizeof(uchar4), w * h, fp);
        fclose(fp);
    }

    delete[] pixels;
    delete[] pixels_ssaa;

    std::cout << "All time: " << total_duration_time << "\n";
    return 0;
}