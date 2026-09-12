#include <stdint.h>
/* Single-thread reference kernels. No SME/NEON intrinsics or BLAS claims. */
void ed_quant(const uint8_t *q, const float *s, const float *x, float *y,
              int m, int k, int n, int g, int bits) {
    int ng=(k+g-1)/g, kp=ng*g, qmax=(1<<(bits-1))-1;
    for(int b=0;b<n;b++) for(int r=0;r<m;r++) {
        float total=0;
        for(int h=0;h<ng;h++) {
            float dot=0;
            for(int j=0;j<g && h*g+j<k;j++) {
                int idx=r*kp+h*g+j;
                int code=bits==4 ? ((q[idx/2] >> ((idx%2)*4)) & 15) : q[idx];
                dot += (float)(code-qmax)*x[b*k+h*g+j];
            }
            total += dot*s[r*ng+h];
        }
        y[b*m+r]=total;
    }
}
void ed_float(const float *w,const float *x,float *y,int m,int k,int n) {
    for(int b=0;b<n;b++) for(int r=0;r<m;r++) {
        float total=0;
        for(int j=0;j<k;j++) total += w[r*k+j]*x[b*k+j];
        y[b*m+r]=total;
    }
}
