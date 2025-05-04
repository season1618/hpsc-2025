#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void thread(int *bucket, int *key) {
    int i = threadIdx.x;
    atomicAdd(&bucket[key[i]], 1);
}

int main() {
  int n = 50;
  int range = 5;

  int *key, *bucket;
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&key, n * sizeof(int));
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }

  thread<<<1, n>>>(bucket, key);
  cudaDeviceSynchronize();
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
