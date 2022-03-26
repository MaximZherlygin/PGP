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

using namespace std;

#define CUDA_ERROR(err) { \
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

struct polygon {
    vec3 p1;
    vec3 p2;
    vec3 p3;
    vec3 color;
};

__host__ __device__  vec3 operator + (vec3 v1, vec3 v2) {
    return  vec3{v1.x + v2.x,
                         v1.y + v2.y,
                         v1.z + v2.z};
}

__host__ __device__  vec3 operator - (vec3 v1, vec3 v2) {
    return  vec3{v1.x - v2.x,
                         v1.y - v2.y,
                         v1.z - v2.z};
}

__host__ __device__  vec3 operator * (vec3 v, double num) {
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
    return vec3{v.x / num,
                        v.y / num,
                        v.z / num};
}

__host__ __device__ vec3 crossing(vec3 v1, vec3 v2) {
    return {v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x};
}

__host__ __device__ vec3 multiply(vec3 a, vec3 b, vec3 c, vec3 v) {
    return { a.x * v.x + b.x * v.y + c.x * v.z,
             a.y * v.x + b.y * v.y + c.y * v.z,
             a.z * v.x + b.z * v.y + c.z * v.z };
}

vec3 normalise_color(vec3 color) {
    return {color.x * 255.,
            color.y * 255.,
            color.z * 255.};
}

__host__ __device__ uchar4 ray_aux(vec3 pos, vec3 dir, vec3 light_pos,
                                   vec3 light_color, polygon *polygons, int n) {
    int min_value = -1;
    double ts_min;
    for (int i = 0; i < n; ++i) {
        vec3 e1 = polygons[i].p2 - polygons[i].p1;
        vec3 e2 = polygons[i].p3 - polygons[i].p1;
        vec3 p = crossing(dir, e2);
        double dwiv = scal_mul(p, e1);

        if (fabs(div) < 1e-10)
            continue;

        vec3 t = pos - polygons[i].p1;
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
        vec3 e1 = polygons[i].p2 - polygons[i].p1;
        vec3 e2 = polygons[i].p3 - polygons[i].p1;
        vec3 p = crossing(dir, e2);
        double div = scal_mul(p, e1);

        if (fabs(div) < 1e-10)
            continue;

        vec3 t = pos - polygons[i].p1;
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
    color_min.x = polygons[min_value].color.x;
    color_min.y = polygons[min_value].color.y;
    color_min.z = polygons[min_value].color.z;

    color_min.x *= light_color.x;
    color_min.y *= light_color.y;
    color_min.z *= light_color.z;
    color_min.w = 0;
    return color_min;
}

void render_cpu(vec3 p_c, vec3 p_v, int w, int h, double fov, uchar4* pixels, vec3 light_pos,
                vec3 light_col, polygon* polygons, int n) {
    double dw = (double)2.0 / (double)(w - 1.0);
    double dh = (double)2.0 / (double)(h - 1.0);
    double z = 1.0 / tan(fov * M_PI / 360.0);
    vec3 b_z = norm(p_v - p_c);
    vec3 b_x = norm(crossing(b_z, {0.0, 0.0, 1.0}));
    vec3 b_y = norm(crossing(b_x, b_z));
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            vec3 v;
            v.x = (double)-1.0 + dw * (double)i;
            v.y = ((double)-1.0 + dh * (double)j) * (double)h / (double)w;
            v.z = z;
            vec3 dir = multiply(b_x, b_y, b_z, v);
            pixels[(h - 1 - j) * w + i] = ray_aux(p_c, norm(dir), light_pos, light_col, polygons, n);
        }
}

__global__ void render_gpu(vec3 p_c, vec3 p_v, int w, int h, double fov, uchar4* pixels,
                           vec3 light_pos, vec3 light_col, polygon* polygons, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    double dw = (double)2.0 / (double)(w - 1.0);
    double dh = (double)2.0 / (double)(h - 1.0);
    double z = 1.0 / tan(fov * M_PI / 360.0);
    vec3 b_z = norm(p_v - p_c);
    vec3 b_x = norm(crossing(b_z, {0.0, 0.0, 1.0}));
    vec3 b_y = norm(crossing(b_x, b_z));
    for (int i = idx; i < w; i += offsetX)
        for (int j = idy; j < h; j += offsetY) {
            vec3 v;
            v.x = (double)-1.0 + dw * (double)i;
            v.y = ((double)-1.0 + dh * (double)j) * (double)h / (double)w;
            v.z = z;
            vec3 dir = multiply(b_x, b_y, b_z, v);
            pixels[(h - 1 - j) * w + i] = ray_aux(p_c, norm(dir), light_pos, light_col, polygons, n);
        }
}

void ssaa_cpu(uchar4 *pixels, int w, int h, int coeff, uchar4 *ssaa_pixels) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int4 mid_pixel = { 0, 0, 0, 0 };
            for (int j = 0; j < coeff; j++) {
                for (int i = 0; i < coeff; i++) {
                    int index = y * w * coeff * coeff + x * coeff + j * w * coeff + i;
                    mid_pixel.x += ssaa_pixels[index].x;
                    mid_pixel.y += ssaa_pixels[index].y;
                    mid_pixel.z += ssaa_pixels[index].z;
                    mid_pixel.w += 0;
                }
            }
            pixels[y * w + x].x = (uchar)(int)(mid_pixel.x / (coeff * coeff));
            pixels[y * w + x].y = (uchar)(int)(mid_pixel.y / (coeff * coeff));
            pixels[y * w + x].z = (uchar)(int)(mid_pixel.z / (coeff * coeff));
            pixels[y * w + x].w = mid_pixel.w;
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
            int4 mid = { 0, 0, 0, 0 };
            for (int j = 0; j < coeff; j++) {
                for (int i = 0; i < coeff; i++) {
                    int index = y * w * coeff * coeff + x * coeff + j * w * coeff + i;
                    mid.x += ssaa_pixels[index].x;
                    mid.y += ssaa_pixels[index].y;
                    mid.z += ssaa_pixels[index].z;
                    mid.w += 0;
                }
            }
            pixels[y * w + x].x = (uchar)(mid.x / (coeff * coeff));
            pixels[y * w + x].y = (uchar)(mid.y / (coeff * coeff));
            pixels[y * w + x].z = (uchar)(mid.z / (coeff * coeff));
            pixels[y * w + x].w = mid.w;
        }
    }
}

void hexahedron(vec3 center, double r, vec3 color, vector<polygon> &polygons) { // ++++++
    // cout << "Creating hexahedron\n";

    color = normalise_color(color);

    // Create all vertices
    vector<vec3> vertices(8);

    vec3 point_a {-1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)};
    vec3 point_b {-1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)};
    vec3 point_c {-1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)};
    vec3 point_d {-1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)};
    vec3 point_e {1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)};
    vec3 point_f {1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)};
    vec3 point_g {1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)};
    vec3 point_h {1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)};

    // 6 sides means 12 polygons of triangles
    // Create with shifting
    polygons.push_back({point_a * r + center, point_b * r + center, point_d * r + center, color});
    polygons.push_back({point_a * r + center, point_c * r + center, point_d * r + center, color});
    polygons.push_back({point_b * r + center, point_f * r + center, point_h * r + center, color});
    polygons.push_back({point_b * r + center, point_d * r + center, point_h * r + center, color});
    polygons.push_back({point_e * r + center, point_f * r + center, point_h * r + center, color});
    polygons.push_back({point_e * r + center, point_g * r + center, point_h * r + center, color});
    polygons.push_back({point_a * r + center, point_e * r + center, point_g * r + center, color});
    polygons.push_back({point_a * r + center, point_c * r + center, point_g * r + center, color});
    polygons.push_back({point_a * r + center, point_b * r + center, point_f * r + center, color});
    polygons.push_back({point_a * r + center, point_e * r + center, point_f * r + center, color});
    polygons.push_back({point_c * r + center, point_d * r + center, point_h * r + center, color});
    polygons.push_back({point_c * r + center, point_g * r + center, point_h * r + center, color});
    // cout << "Creating hexahedron done\n";
}

void icosahedron(vec3 center, double radius, vec3 color, vector<polygon> &polygons) {
    double atctan_1_2 = 26.565; // arctan(1/2) ~ +-26.57
    double angle = M_PI * atctan_1_2 / 180;
    double segment_angle = M_PI * 72 / 180;
    double current_angle = 0.0;

    vector<vec3> vertices(12);
    vertices[0] = {0, radius, 0};
    vertices[11] = {0, -radius, 0};

    for (int i = 1; i < 6; i++) {
        vertices[i] = {radius * sin(current_angle) * cos(angle),
                       radius * sin(angle),
                       radius * cos(current_angle) * cos(angle)};
        current_angle += segment_angle;
    }

    current_angle = M_PI * 36 / 180;

    for (int i = 6; i < 11; i++) {
        vertices[i] = {radius * sin(current_angle) * cos(-angle),
                       radius * sin(-angle),
                       radius * cos(current_angle) * cos(-angle)};
        current_angle += segment_angle;
    }

    for (auto &j: vertices) {
        j.x = j.x + center.x;
        j.y = j.y + center.y;
        j.z = j.z + center.z;
    }

    polygons.push_back({vertices[10], vertices[6], vertices[1], color});
    polygons.push_back({vertices[9], vertices[10], vertices[5], color});
    polygons.push_back({vertices[8], vertices[9], vertices[4], color});
    polygons.push_back({vertices[7], vertices[8], vertices[3], color});
    polygons.push_back({vertices[6], vertices[7], vertices[2], color});

    polygons.push_back({vertices[1], vertices[5], vertices[10], color});
    polygons.push_back({vertices[5], vertices[4], vertices[9], color});
    polygons.push_back({vertices[4], vertices[3], vertices[8], color});
    polygons.push_back({vertices[3], vertices[2], vertices[7], color});
    polygons.push_back({vertices[2], vertices[1], vertices[6], color});

    polygons.push_back({vertices[11], vertices[6], vertices[10], color});
    polygons.push_back({vertices[11], vertices[10], vertices[9], color});
    polygons.push_back({vertices[11], vertices[9], vertices[8], color});
    polygons.push_back({vertices[11], vertices[8], vertices[7], color});
    polygons.push_back({vertices[11], vertices[7], vertices[6], color});

    polygons.push_back({vertices[0], vertices[5], vertices[1], color});
    polygons.push_back({vertices[0], vertices[4], vertices[5], color});
    polygons.push_back({vertices[0], vertices[3], vertices[4], color});
    polygons.push_back({vertices[0], vertices[2], vertices[3], color});
    polygons.push_back({vertices[0], vertices[1], vertices[2], color});
}

void dodecahedron(vec3 center, double r, vec3 color, vector<polygon> &polygons) { // +++++++
    // cout << "Creating dodecahedron\n";

    color = normalise_color(color);
    double a = (1 + sqrt(5)) / 2;
    double b = 2 / (1 + sqrt(5));
    // 20 vertices and 12 * 3 polygons (because pentagon == 3 triangles)
    vector<vec3> vertices {{-b, 0, a} ,
                                   { b, 0, a} ,
                                   {-1, 1, 1} ,
                                   { 1, 1, 1} ,
                                   { 1, -1, 1} ,
                                   {-1, -1, 1} ,
                                   { 0, -a, b} ,
                                   { 0, a, b} ,
                                   {-a, -b, 0} ,
                                   {-a, b, 0} ,
                                   { a, b, 0} ,
                                   { a, -b, 0} ,
                                   { 0, -a, -b} ,
                                   { 0, a, -b} ,
                                   { 1, 1, -1} ,
                                   { 1, -1, -1} ,
                                   {-1, -1, -1} ,
                                   {-1, 1, -1} ,
                                   { b, 0, -a} ,
                                   {-b, 0, -a}
    };

    for (auto &j: vertices) {
        j.x /= sqrt(3);
        j.y /= sqrt(3);
        j.z /= sqrt(3);
    }

    // Shifting
    for (auto &j: vertices) {
        j.x = j.x * r + center.x;
        j.y = j.y * r + center.y;
        j.z = j.z * r + center.z;
    }

    // Applying 36 polygons
    polygons.push_back({vertices[4], vertices[0], vertices[6], color});
    polygons.push_back({vertices[0], vertices[5], vertices[6], color});
    polygons.push_back({vertices[0], vertices[4], vertices[1], color});
    polygons.push_back({vertices[0], vertices[3], vertices[7], color});
    polygons.push_back({vertices[2], vertices[0], vertices[7], color});
    polygons.push_back({vertices[0], vertices[1], vertices[3], color});
    polygons.push_back({vertices[10], vertices[1], vertices[11], color});
    polygons.push_back({vertices[3], vertices[1], vertices[10], color});
    polygons.push_back({vertices[1], vertices[4], vertices[11], color});
    polygons.push_back({vertices[5], vertices[0], vertices[8], color});
    polygons.push_back({vertices[0], vertices[2], vertices[9], color});
    polygons.push_back({vertices[8], vertices[0], vertices[9], color});
    polygons.push_back({vertices[5], vertices[8], vertices[16], color});
    polygons.push_back({vertices[6], vertices[5], vertices[12], color});
    polygons.push_back({vertices[12], vertices[5], vertices[16], color});
    polygons.push_back({vertices[4], vertices[12], vertices[15], color});
    polygons.push_back({vertices[4], vertices[6], vertices[12], color});
    polygons.push_back({vertices[11], vertices[4], vertices[15], color});
    polygons.push_back({vertices[2], vertices[13], vertices[17], color});
    polygons.push_back({vertices[2], vertices[7], vertices[13], color});
    polygons.push_back({vertices[9], vertices[2], vertices[17], color});
    polygons.push_back({vertices[13], vertices[3], vertices[14], color});
    polygons.push_back({vertices[7], vertices[3], vertices[13], color});
    polygons.push_back({vertices[3], vertices[10], vertices[14], color});
    polygons.push_back({vertices[8], vertices[17], vertices[19], color});
    polygons.push_back({vertices[16], vertices[8], vertices[19], color});
    polygons.push_back({vertices[8], vertices[9], vertices[17], color});
    polygons.push_back({vertices[14], vertices[11], vertices[18], color});
    polygons.push_back({vertices[11], vertices[15], vertices[18], color});
    polygons.push_back({vertices[10], vertices[11], vertices[14], color});
    polygons.push_back({vertices[12], vertices[19], vertices[18], color});
    polygons.push_back({vertices[15], vertices[12], vertices[18], color});
    polygons.push_back({vertices[12], vertices[16], vertices[19], color});
    polygons.push_back({vertices[19], vertices[13], vertices[18], color});
    polygons.push_back({vertices[17], vertices[13], vertices[19], color});
    polygons.push_back({vertices[13], vertices[14], vertices[18], color});

    // cout << "Creating dodecahedron done\n";
}

void scene(vec3 a, vec3 b, vec3 c, vec3 d, vec3 color,
           vector<polygon> &polygons) {
    // cout << "Creating scene\n";
    color = normalise_color(color);
    polygons.push_back(polygon{a, b, c, color});
    polygons.push_back(polygon{c, d, a, color});
}

int cpu_mode(vec3 p_c, vec3 p_v, int w, int ssaa_w, int h, int ssaa_h, double fov, uchar4* pixels,
             uchar4* pixels_ssaa, vec3 light_pos, vec3 light_col, polygon* polygons, int n, int ssaa_multiplier) {
    render_cpu(p_c, p_v, ssaa_w, ssaa_h, fov, pixels_ssaa, light_pos, light_col, polygons, n);
    ssaa_cpu(pixels, w, h, ssaa_multiplier, pixels_ssaa);

    return 0;
}

int gpu_mode(vec3 p_c, vec3 p_v, int w, int ssaa_w, int h, int ssaa_h, double fov, uchar4* pixels,
             uchar4* pixels_ssaa, vec3 light_pos, vec3 light_col, polygon* polygons, int n, int ssaa_multiplier) {
//    cerr << "Allocate pixels\n";
    // Allocating on gpu
    uchar4* gpu_pixels;
    CUDA_ERROR(cudaMalloc((uchar4**)(&gpu_pixels), w * h * sizeof(uchar4)));
    CUDA_ERROR(cudaMemcpy(gpu_pixels, pixels, w * h * sizeof(uchar4), cudaMemcpyHostToDevice));
//    cerr << "Allocate ssaa pixels\n";
    uchar4* gpu_pixels_ssaa;
    CUDA_ERROR(cudaMalloc((uchar4**)(&gpu_pixels_ssaa), ssaa_w * ssaa_h * sizeof(uchar4)));
    CUDA_ERROR(cudaMemcpy(gpu_pixels_ssaa, pixels_ssaa, ssaa_w * ssaa_h * sizeof(uchar4), cudaMemcpyHostToDevice));
//    cerr << "Allocate polygons\n";
    polygon* gpu_polygons;
    CUDA_ERROR(cudaMalloc((polygon**)(&gpu_polygons), n * sizeof(polygon)));
    CUDA_ERROR(cudaMemcpy(gpu_polygons, polygons, n * sizeof(polygon), cudaMemcpyHostToDevice));
//    cerr << "Start render\n";
    // Rendering
    render_gpu <<< 128, 128 >>> (p_c, p_v, ssaa_w, ssaa_h, fov, gpu_pixels_ssaa, light_pos, light_col, gpu_polygons, n);
    cudaThreadSynchronize();
    CUDA_ERROR(cudaGetLastError());
//    cerr << "Start ssaa\n";
    // Ssaa smoothing algo
    ssaa_gpu <<< 128, 128 >>> (gpu_pixels, w, h, ssaa_multiplier, gpu_pixels_ssaa);
    cudaThreadSynchronize();
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaMemcpy(pixels, gpu_pixels, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_ERROR(cudaFree(gpu_pixels));
    CUDA_ERROR(cudaFree(gpu_pixels_ssaa));
    CUDA_ERROR(cudaFree(gpu_polygons));

    return 0;
}

int main(int argc, char* argv[]) {
    string mode;
    if (argv[1])
        mode = argv[1];
    bool is_gpu = true;

    if (argc > 2) {
        cout << "Incorrect params. Please use '--help' for help\n";
        return 0;
    }

    if (argc == 1 || mode == "--gpu")
        is_gpu = true;

    if (mode == "--cpu")
        is_gpu = false;

    if (mode == "--default") {
        cout << "10\n"
                "./frames_data\n"
                "640 480 120\n"
                "7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0\n"
                "2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n"
                "4.0 4.0 0.0 0.5 0.0 1.0 2.0 0.0 0.0 0.0\n"
                "1.0 1.0 0.0 0.5 0.0 1.0 2.0 0.0 0.0 0.0\n"
                "-2.5 -2.5 0.0 0.5 1.0 1.0 2.0 0.0 0.0 0.0\n"
                "-10.0 -10.0 -1.0 -10.0 10.0 -1.0 10.0 10.0 -1.0 10.0 -10.0 -1.0 temp 0.0 0.9 0.0 0.5\n"
                "1\n"
                "100 100 100 1.0 1.0 1.0\n"
                "1 3\n";
        return 0;
    }

    if (mode == "--help") {
        cout << "<---------------   HELP   --------------->\n"
                "Start program without args will cause computation in gpu mode\n"
                "--cpu     For computation with using cpu\n"
                "--gpu     For computation with using gpu\n"
                "--default Print best configuration for input data\n"
                "--help    For help\n"
                "<---------------END OF HELP--------------->\n";
        return 0;
    }

    int total_frames, width, height, fov;
    string path_to_frames;

    double r_0c, z_0c, phi_0c;
    double A_rc, A_zc;
    double w_rc, w_zc, w_phic;
    double p_rc, p_zc;

    double r_0v, z_0v, phi_0v;
    double A_rv, A_zv;
    double w_rv, w_zv, w_phiv;
    double p_rv, p_zv;

    vec3 center, color;
    double radius;

    string unused;

    vec3 scene_a, scene_b, scene_c, scene_d;
    vec3 light_pos, light_col;

    vector<polygon> polygons;
    polygon *polygons_as_array;
    uchar4 *pixels = nullptr;
    uchar4 *pixels_ssaa = nullptr;


    int n_lights; // Should be 1 (1 light)
    int recursion_step; // Should be 1 (unused)
    int ssaa_multiplier;

    // Frames
    cin >> total_frames;
    cin >> path_to_frames;
    cin >> width >> height >> fov;

    // Camera trajectory
    cin >> r_0c >> z_0c >> phi_0c;
    cin >> A_rc >> A_zc;
    cin >> w_rc >> w_zc >> w_phic;
    cin >> p_rc >> p_zc;

//    cerr << r_0c << " " << z_0c << " " << phi_0c << "\n";
//    cerr << A_rc << " " << A_zc << "\n";
//    cerr << w_rc << " " << w_zc<< " "  << w_phic << "\n";
//    cerr << p_rc << " " << p_zc << "\n";

    cin >> r_0v >> z_0v >> phi_0v;
    cin >> A_rv >> A_zv;
    cin >> w_rv >> w_zv >> w_phiv;
    cin >> p_rv >> p_zv;

//    cerr << r_0v << " " << z_0v << " " << phi_0v << "\n";
//    cerr << A_rv << " " << A_zv << "\n";
//    cerr << w_rv << " " << w_zv << " " << w_phiv << "\n";
//    cerr << p_rv << " " << p_zv << "\n";

    // Figures params with creating
    cin >> center.x >> center.y >> center.z >> color.x >> color.y >> color.z >> radius >> unused >> unused >> unused;
    hexahedron(center, radius, color, polygons);
    cin >> center.x >> center.y >> center.z >> color.x >> color.y >> color.z >> radius >> unused >> unused >> unused;
    icosahedron(center, radius, color, polygons);
    cin >> center.x >> center.y >> center.z >> color.x >> color.y >> color.z >> radius >> unused >> unused >> unused;
    dodecahedron(center, radius, color, polygons);

    // Scene
    cin >> scene_a.x >> scene_a.y >> scene_a.z;
    cin >> scene_b.x >> scene_b.y >> scene_b.z;
    cin >> scene_c.x >> scene_c.y >> scene_c.z;
    cin >> scene_d.x >> scene_d.y >> scene_d.z;

    cin >> unused;
    cin >> color.x >> color.y >> color.z;
    cin >> unused;
    scene(scene_a, scene_b, scene_c, scene_d, color, polygons);

    // Lights
    cin >> n_lights;
    cin >> light_pos.x >> light_pos.y >> light_pos.z;
    cin >> light_col.x >> light_col.y >> light_col.z;

    // Recursion
    cin >> recursion_step;
//    cerr << recursion_step << "\n";

    // SSAA params
    cin >> ssaa_multiplier;
//    cerr << ssaa_multiplier << "\n";

    int ssaa_width = width * ssaa_multiplier;
    int ssaa_height = height * ssaa_multiplier;

    pixels = new uchar4[ssaa_width * ssaa_height];
    pixels_ssaa = new uchar4[ssaa_width * ssaa_height]; // cpu

    polygons_as_array = polygons.data();
    int total_polygons = polygons.size();

//    for (int i = 0; i < polygons.size(); ++i) {
//        cerr << "p1:" << polygons[i].p1.x << " " << polygons[i].p1.y << " " << polygons[i].p1.z << "\n";
//        cerr << "p2:" << polygons[i].p2.x << " " << polygons[i].p2.y << " " << polygons[i].p2.z << "\n";
//        cerr << "p3:" << polygons[i].p3.x << " " << polygons[i].p3.y << " " << polygons[i].p3.z << "\n";
//        cerr << "color:" << polygons[i].color.x << " " << polygons[i].color.y << " " << polygons[i].color.z << "\n";
//    }

    cout << "Start rendering. Total polygons: " << total_polygons << ". Frame size: " << width << "x" << height;
    cout << ". Total frames: " << total_frames << "\n";
    cout << "|\tIteration number\t|\t time in ms\t|\ttotal rays |\n";

    double r_c, z_c, phi_c , r_v, z_v, phi_v;
    vec3 p_c, p_v;
    int sum_of_rays;

    double total_duration_time = 0;
    for (int i = 0; i < total_frames; i++) {
        auto start = chrono::steady_clock::now();
        double time_step = 2.0 * M_PI / total_frames;
        double cur_time = i * time_step;

        // Movement
        r_c = r_0c + A_rc * sin(w_rc * cur_time + p_rc);
        z_c = z_0c + A_zc * sin(w_zc * cur_time + p_zc);
        phi_c = phi_0c + w_phic * cur_time;

        r_v = r_0v + A_rv * sin(w_rv * cur_time + p_rv);
        z_v = z_0v + A_zv * sin(w_zv * cur_time + p_zv);
        phi_v = phi_0v + w_phiv * cur_time;

        p_c = { r_c * cos(phi_c), r_c * sin(phi_c), z_c };
        p_v = { r_v * cos(phi_v), r_v * sin(phi_v), z_v };

        // Total sum of rays (will be the same coz of recursion)
        sum_of_rays = ssaa_width * ssaa_height;

        int res;
        if (is_gpu)
            res = gpu_mode(p_c, p_v, width, ssaa_width, height, ssaa_height, (double)fov, pixels, pixels_ssaa,
                           light_pos, light_col, polygons_as_array, total_polygons, ssaa_multiplier);
        else
            res = cpu_mode(p_c, p_v, width, ssaa_width, height, ssaa_height, (double)fov, pixels, pixels_ssaa,
                           light_pos, light_col, polygons_as_array, total_polygons, ssaa_multiplier);
        if (res)
            cout << "An error occurred. Check output\n";

        auto end = chrono::steady_clock::now();
        cout << "|\tIteration " << i + 1 << " of " << total_frames << "\t|\t";
        double iteration_time = ((double)chrono::duration_cast<chrono::microseconds>(end - start).count()) / 1000.0;
        total_duration_time += iteration_time;
        cout << iteration_time << "ms\t|\t";
        cout << sum_of_rays << "\t\t|\n";

        string frame_name = path_to_frames + "/" + to_string(i) + ".data";
        FILE* f = fopen(frame_name.c_str(), "wb");
//        fwrite(&ssaa_width, sizeof(int), 1, f);
//        fwrite(&ssaa_height, sizeof(int), 1, f);
//        fwrite(pixels_ssaa, sizeof(uchar4), ssaa_width * ssaa_height, f);
        fwrite(&width, sizeof(int), 1, f); // тут падает
        fwrite(&height, sizeof(int), 1, f);
        fwrite(pixels, sizeof(uchar4), width * height, f);
        fclose(f);
    }

    if (pixels)
        delete[] pixels;
    if (pixels_ssaa)
        delete[] pixels_ssaa;

    cout << "Done with total duration: " << total_duration_time << "ms\n";
    return 0;
}