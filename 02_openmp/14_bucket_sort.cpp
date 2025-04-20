#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50000000;
  int range = 500;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    //printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
#pragma omp parallel for schedule(dynamic)
  for (int i=0; i<range; i++) {
    int base = offset[i];
#pragma omp parallel for
    for (int j=0; j<bucket[i]; j++) {
      key[base+j] = i;
    }
  }

  for (int i=0; i<n; i++) {
    //printf("%d ",key[i]);
  }
  printf("\n");
}
