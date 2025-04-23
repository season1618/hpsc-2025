#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  int idx[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;

    idx[i] = i;
  }

  __m512 x_vec = _mm512_load_ps(x);
  __m512 y_vec = _mm512_load_ps(y);
  __m512 m_vec = _mm512_load_ps(m);
  __m512i idx_vec = _mm512_load_si512(idx);
  for(int i=0; i<N; i++) {
    __m512 x_i = _mm512_set1_ps(x[i]);
    __m512 y_i = _mm512_set1_ps(y[i]);
    __m512 rx_vec = _mm512_sub_ps(x_i, x_vec);
    __m512 ry_vec = _mm512_sub_ps(y_i, y_vec);
    __m512 rx2_vec = _mm512_mul_ps(rx_vec, rx_vec);
    __m512 ry2_vec = _mm512_mul_ps(ry_vec, ry_vec);
    __m512 r2_vec = _mm512_add_ps(rx2_vec, ry2_vec);
    __m512 rinv_vec = _mm512_rsqrt14_ps(r2_vec);

    __mmask16 mask = _mm512_cmp_epu32_mask(_mm512_set1_epi32(i), idx_vec, _MM_CMPINT_NE);
    __m512 tmp = _mm512_mul_ps(
      _mm512_mul_ps(m_vec, rinv_vec),
      _mm512_mul_ps(rinv_vec, rinv_vec)
    );

    __m512 fx_vec = _mm512_mul_ps(rx_vec, tmp);
    __m512 fy_vec = _mm512_mul_ps(ry_vec, tmp);

    fx[i] -= _mm512_mask_reduce_add_ps(mask, fx_vec);
    fy[i] -= _mm512_mask_reduce_add_ps(mask, fy_vec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
