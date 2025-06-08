#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>

#define nx 41
#define ny 41

#define N_THREAD 1024
#define N_BLOCK (ny * nx + N_THREAD - 1) / N_THREAD

using namespace std;

__device__ float u[ny][nx];
__device__ float v[ny][nx];
__device__ float p[ny][nx];
__device__ float b[ny][nx];
__device__ float un[ny][nx];
__device__ float vn[ny][nx];
__device__ float pn[ny][nx];

__device__ double dx = 2. / (nx - 1);
__device__ double dy = 2. / (ny - 1);
__device__ double dt = .01;
__device__ double rho = 1.;
__device__ double nu = .02;

__global__ void init() {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (j >= ny) return;

  u[j][i] = 0;
  v[j][i] = 0;
  p[j][i] = 0;
  b[j][i] = 0;
}

__global__ void compute_b() {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (j >= ny) return;
  
  // Compute b[j][i]
  float dudx = (u[j][i+1] - u[j][i-1]) / (2 * dx);
  float dudy = (u[j][i+1] - u[j][i-1]) / (2 * dy);
  float dvdx = (v[j+1][i] - v[j-1][i]) / (2 * dx);
  float dvdy = (v[j+1][i] - v[j-1][i]) / (2 * dy);
  b[j][i] = rho * (1 / dt * (dudx + dvdy)
		 - dudx * dudx - 2 * dudy * dvdx - dvdy * dvdy);
}   

__global__ void copy_p() {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (j >= ny) return;
  
  pn[j][i] = p[j][i];
}

__global__ void compute_p() {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (j >= ny) return;

  float *a = &p[j][i];

  if (j == 0) j = 1;
  else if (j == ny-1) { *a = 0; return; }
  else if (i == 0) i = 1;
  else if (i == nx-1) i = nx-2;
  
  // Compute p[j][i]
  *a = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
             dx*dx * (pn[j+1][i] + pn[j-1][i]) -
             b[j][i] * dx*dx * dy*dy)
           / (2 * (dx*dx + dy*dy));
}

__global__ void copy_uv() {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (j >= ny) return;

  un[j][i] = u[j][i];
  vn[j][i] = v[j][i];
}

__global__ void compute_uv() {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx;
  int i = id % nx;
  if (j >= ny) return;
  
  // Compute u[j][i] and v[j][i]
  if (j == 0) {
    u[0][i] = 0;
    v[0][i] = 0;
    return;
  }
  if (j == ny-1) {
    u[ny-1][i] = 1;
    v[ny-1][i] = 0;
    return;
  }
  if (i == 0 || i == nx-1) {
    u[j][i] = 0;
    v[j][i] = 0;
    return;
  }
  
  u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                     - vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                     - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                     + nu * dt / (dx*dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                     + nu * dt / (dy*dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
  v[j][i] = vn[j][i] - un[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                     - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                     - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                     + nu * dt / (dx*dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                     + nu * dt / (dy*dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
}

float u_cpu[ny][nx];
float v_cpu[ny][nx];
float p_cpu[ny][nx];

int main() {
  int nt = 500;
  int nit = 50;

  init<<<N_BLOCK, N_THREAD>>>();
  cudaDeviceSynchronize();

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {
    compute_b<<<N_BLOCK, N_THREAD>>>();
    cudaDeviceSynchronize();
    for (int it=0; it<nit; it++) {
      copy_p<<<N_BLOCK, N_THREAD>>>();
      cudaDeviceSynchronize();

      compute_p<<<N_BLOCK, N_THREAD>>>();
      cudaDeviceSynchronize();
    }
    copy_uv<<<N_BLOCK, N_THREAD>>>();
    cudaDeviceSynchronize();

    compute_uv<<<N_BLOCK, N_THREAD>>>();
    cudaDeviceSynchronize();

    if (n % 10 == 0) {
      cudaMemcpyFromSymbol(u_cpu, u, sizeof(float) * ny * nx, 0);
      cudaMemcpyFromSymbol(v_cpu, v, sizeof(float) * ny * nx, 0);
      cudaMemcpyFromSymbol(p_cpu, p, sizeof(float) * ny * nx, 0);

      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u_cpu[j][i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v_cpu[j][i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p_cpu[j][i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}


