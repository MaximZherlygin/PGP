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
    vec3 a;
    vec3 b;
    vec3 c;
    vec3 color;
};

bool replace(std::string &str, const std::string &from, const std::string &to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

__host__ __device__ vec3 normalize(vec3 hs) {
    vec3 result;
    result.x = hs.x / sqrt(hs * hs);
    result.y = hs.y / sqrt(hs * hs);
    result.z = hs.z / sqrt(hs * hs);
    return result;
}

__host__ __device__ vec3 prod(vec3 lhs, vec3 rhs) {
    vec3 result;
    result.x = lhs.y * rhs.z - lhs.z * rhs.y;
    result.y = lhs.z * rhs.x - lhs.x * rhs.z;
    result.z = lhs.x * rhs.y - lhs.y * rhs.x;
    return result;
}

__host__ __device__ vec3 mult(vec3 first_hs, vec3 second_hs, vec3 third_hs, vec3 multipy_hs) {
    vec3 result;
    result.x = first_hs.x * multipy_hs.x + second_hs.x * multipy_hs.y + third_hs.x * multipy_hs.z;
    result.y = first_hs.y * multipy_hs.x + second_hs.y * multipy_hs.y + third_hs.y * multipy_hs.z;
    result.z = first_hs.z * multipy_hs.x + second_hs.z * multipy_hs.y + third_hs.z * multipy_hs.z;
    return result;
}


__host__ __device__ vec3 operator+(vec3 lhs, vec3 rhs) {
    vec3 result;
    result.x = lhs.x + rhs.x;
    result.y = lhs.y + rhs.y;
    result.z = lhs.z + rhs.z;
    return result;
}

__host__ __device__ vec3 operator-(vec3 lhs, vec3 rhs) {
    vec3 result;
    result.x = lhs.x - rhs.x;
    result.y = lhs.y - rhs.y;
    result.z = lhs.z - rhs.z;
    return result;
}

__host__ __device__ double operator*(vec3 lhs, vec3 rhs) {
    double result = ((lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z));
    return result;
}

__host__ __device__ vec3 operator*(vec3 lhs, double rhs) {
    vec3 result;
    result.x = lhs.x * rhs;
    result.y = lhs.y * rhs;
    result.z = lhs.z * rhs;
    return result;
}

void ssaa_cpu(uchar4 *data, uchar4 *out_data, int w, int h, int k) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int4 mid = {0, 0, 0, 0};

            for (int j = 0; j < k; j++) {
                for (int i = 0; i < k; i++) {
                    int index = k * k * y * w + k * j * w + k * x + i;
                    mid.x += data[index].x;
                    mid.y += data[index].y;
                    mid.z += data[index].z;
                }
            }
            double div = k * k;
            out_data[x + y * w] = make_uchar4(mid.x / div, mid.y / div, mid.z / div, 0);
        }
    }
}

__global__ void ssaa_gpu(uchar4 *data, uchar4 *out_data, int w, int h, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsetY) {
        for (int x = idx; x < w; x += offsetX) {
            int4 mid = {0, 0, 0, 0};
            for (int j = 0; j < k; j++) {
                for (int i = 0; i < k; i++) {
                    int index = k * k * y * w + k * j * w + k * x + i;
                    mid.x += data[index].x;
                    mid.y += data[index].y;
                    mid.z += data[index].z;
                }
            }

            double div = k * k;
            out_data[x + y * w] = make_uchar4(mid.x / div, mid.y / div, mid.z / div, 0);
        }
    }
}

__host__ __device__ uchar4 ray(vec3 pos, vec3 dir, vec3 l_position, vec3 l_color, triangle *trigs, int rays_sqrt) {
    // взято из примера с лекций
    int k_min = -1;
    double ts_min;
    for (int k = 0; k < rays_sqrt; k++) {
        vec3 e1 = trigs[k].b - trigs[k].a;
        vec3 e2 = trigs[k].c - trigs[k].a;
        vec3 p = prod(dir, e2);
        double div = p * e1;
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = pos - trigs[k].a;
        double u = (p * t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = (q * dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = (q * e2) / div;
        if (ts < 0.0)
            continue;
        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }
    if (k_min == -1)
        return {0, 0, 0, 0};
    pos = dir * ts_min + pos;
    dir = l_position - pos;
    
    double size = sqrt(dir * dir);
    
    dir = normalize(dir);
    for (int k = 0; k < rays_sqrt; k++) {
        vec3 e1 = trigs[k].b - trigs[k].a;
        vec3 e2 = trigs[k].c - trigs[k].a;
        vec3 p = prod(dir, e2);
        double div = p * e1;
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = pos - trigs[k].a;
        double u = (p * t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = (q * dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = (q * e2) / div;
        if (ts > 0.0 && ts < size && k != k_min) {
            return {0, 0, 0, 0};
        }
    }

    uchar4 color_min;
    color_min.x = trigs[k_min].color.x;
    color_min.y = trigs[k_min].color.y;
    color_min.z = trigs[k_min].color.z;

    color_min.x *= l_color.x;
    color_min.y *= l_color.y;
    color_min.z *= l_color.z;
    color_min.w = 0;
    return color_min;
}

void cpu_render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, vec3 l_position, vec3 l_color, triangle *trigs, int rays_sqrt) {
    // из примера с лекций
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = normalize(pv - pc);
    vec3 bx = normalize(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = normalize(prod(bx, bz));
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] = ray(pc, normalize(dir), l_position, l_color, trigs, rays_sqrt);
        }
}

__global__ void gpu_render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4* data, vec3 l_position, vec3 l_color, triangle *trigs, int rays_sqrt) {
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int id_y = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    // из примера с лекций
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 b_z = normalize(pv - pc);
    vec3 b_x = normalize(prod(b_z, {0.0, 0.0, 1.0}));
    vec3 b_y = normalize(prod(b_x, b_z));
    for (int i = id_x; i < w; i += offset_x)
        for (int j = id_y; j < h; j += offset_y) {
            vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
            vec3 dir = mult(b_x, b_y, b_z, v);
            data[(h - 1 - j) * w + i] = ray(pc, normalize(dir), l_position, l_color, trigs, rays_sqrt);
        }
}

void cpu_mode(uchar4 *data, uchar4 *s_data, triangle *trigs, vec3 p_c, vec3 p_v, int w, int h, int s_w, int s_h, double angle, vec3 l_position, vec3 l_color, int rays_sqrt, int k) {
    cpu_render(p_c, p_v, s_w, s_h, angle, s_data, l_position, l_color, trigs, rays_sqrt);
    ssaa_cpu(s_data, data, w, h, k);
}

int gpu_mode(uchar4 *data, uchar4 *s_data, triangle *trigs, vec3 p_c, vec3 p_v, int w, int h, int s_w, int s_h, double angle, vec3 l_position, vec3 l_color, int rays_sqrt, int k) {
    uchar4 *gpu_data;
    uchar4 *gpu_s_data;
    triangle *gpu_trigs;

    CSC(cudaMalloc((uchar4**)(&gpu_data), w * h * sizeof(uchar4)));
    CSC(cudaMemcpy(gpu_data, data, w * h * sizeof(uchar4), cudaMemcpyHostToDevice));

    CSC(cudaMalloc((uchar4**)(&gpu_s_data), s_w * s_h * sizeof(uchar4)));
    CSC(cudaMemcpy(gpu_s_data, s_data, s_w * s_h * sizeof(uchar4), cudaMemcpyHostToDevice));

    CSC(cudaMalloc((triangle**) (&gpu_trigs), rays_sqrt * sizeof(triangle)));
    CSC(cudaMemcpy(gpu_trigs, trigs, rays_sqrt * sizeof(triangle), cudaMemcpyHostToDevice));

    gpu_render<<<256, 256>>>(p_c, p_v, s_w, s_h, angle, gpu_s_data, l_position, l_color, gpu_trigs, rays_sqrt);

    cudaThreadSynchronize();
    CSC(cudaGetLastError());

    ssaa_gpu<<<256, 256>>>(gpu_s_data, gpu_data, w, h, k);

    cudaThreadSynchronize();
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(data, gpu_data, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));
    CSC(cudaFree(gpu_data));
    CSC(cudaFree(gpu_trigs));
    CSC(cudaFree(gpu_s_data));
    return 0;
}


void create_scene(vec3 first_point, vec3 second_point, vec3 third_point, vec3 fourth_point, vec3 color, std::vector <triangle> &trigs) {
    trigs.push_back(triangle{third_point, fourth_point, first_point, color});
    trigs.push_back(triangle{first_point, second_point, third_point, color});
}


void create_icosahedron(std::vector<triangle> &trigs, const double& radius, const vec3& c_coords, const vec3& colors) {
    double atrtan_1_2 = 26.565; // arctan(1/2) ~ +-26.57
    double angle = M_PI * atrtan_1_2 / 180;
    double segment_angle = M_PI * 72 / 180;
    double current_angle = 0.0;

    std::vector <vec3> points(12);
    points[0] = {0, radius, 0};
    points[11] = {0, -radius, 0};

    for (int i = 1; i < 6; i++) {
        points[i] = {radius * sin(current_angle) * cos(angle), radius * sin(angle), radius * cos(current_angle) * cos(angle)};
        current_angle += segment_angle;
    }

    current_angle = M_PI * 36 / 180;

    for (int i = 6; i < 11; i++) {
        points[i] = {radius * sin(current_angle) * cos(-angle), radius * sin(-angle), radius * cos(current_angle) * cos(-angle)};
        current_angle += segment_angle;
    }

    for (int i = 0; i < points.size(); i++) {
        points[i].x += c_coords.x;
        points[i].y += c_coords.y;
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



void create_dodecahedron(std::vector<triangle>& trigs, const double& radius, const vec3& c_coords, const vec3& colors) { // +++++++
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

        points[i].z *= radius;
        points[i].z += c_coords.z;
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
}



void create_hexahedron(std::vector<triangle>& trigs, const double& radius, const vec3& c_coords, const vec3& colors) { // ++++++
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


int main(int argc, char *argv[]) {
    std::string arg;
    if (argv[1]) arg = argv[1];

    if (arg == "--default") {
        std::cout << "100" << "\n";
        std::cout << "./out/img_%d.data" << "\n";
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

    int frames_count;
    std::cin >> frames_count;
    std::string path_to_frames;
    std::cin >> path_to_frames;
    int w;
    int h;
    double view_angle;
    std::cin >> w >> h >> view_angle;

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

    vec3 c_verticles;
    vec3 col_values;
    double radius;
    std::vector <triangle> trigs;
    std::cin >> c_verticles.x >> c_verticles.y >> c_verticles.z >> col_values.x >> col_values.y >> col_values.z
             >> radius >> temp >> temp >> temp;
    col_values.x *= 255.0;
    col_values.y *= 255.0;
    col_values.z *= 255.0;
    create_hexahedron(trigs, radius, c_verticles, col_values);
    std::cin >> c_verticles.x >> c_verticles.y >> c_verticles.z >> col_values.x >> col_values.y >> col_values.z
             >> radius >> temp >> temp >> temp;
    col_values.x *= 255.0;
    col_values.y *= 255.0;
    col_values.z *= 255.0;
    create_icosahedron(trigs, radius, c_verticles, col_values);
    std::cin >> c_verticles.x >> c_verticles.y >> c_verticles.z >> col_values.x >> col_values.y >> col_values.z
             >> radius >> temp >> temp >> temp;
    col_values.x *= 255.0;
    col_values.y *= 255.0;
    col_values.z *= 255.0;
    create_dodecahedron(trigs, radius, c_verticles, col_values);

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
    col_values.x *= 255.0;
    col_values.y *= 255.0;
    col_values.z *= 255.0;
    create_scene(floor_first_point, floor_second_point, floor_third_point, floor_fourth_point, col_values, trigs);

    int l_count;
    vec3 l_position;
    vec3 l_color;
    std::cin >> l_count;
    std::cin >> l_position.x >> l_position.y >> l_position.z >> l_color.x >> l_color.y >> l_color.z;

    int recurse_count;
    int rays_sqrt;
    std::cin >> recurse_count >> rays_sqrt;

    int sum_of_rays;

    triangle* trigs_arr = trigs.data();
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

        sum_of_rays = w * h * rays_sqrt * rays_sqrt;
        int p_size = trigs.size();

        if (!is_gpu) {
            cpu_mode(pixels, pixels_ssaa, trigs_arr, p_c, p_v, w, h, w * rays_sqrt, h * rays_sqrt, view_angle, l_position, l_color, p_size, rays_sqrt);
        } else {
            gpu_mode(pixels, pixels_ssaa, trigs_arr, p_c, p_v, w, h, w * rays_sqrt, h * rays_sqrt, view_angle, l_position, l_color, p_size, rays_sqrt);
        }

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