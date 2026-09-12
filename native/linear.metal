#include <metal_stdlib>
using namespace metal;
struct Shape { uint m,k,n,g,bits; };
kernel void linear_q(device const uchar *q [[buffer(0)]],
                     device const float *s [[buffer(1)]],
                     device const float *x [[buffer(2)]],
                     device float *y [[buffer(3)]],
                     constant Shape &a [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if(id>=a.m*a.n) return;
    uint r=id%a.m,b=id/a.m,ng=(a.k+a.g-1)/a.g,kp=ng*a.g;
    int qm=(1<<(a.bits-1))-1;
    float total=0;
    for(uint h=0;h<ng;h++) {
        float dot=0;
        for(uint j=0;j<a.g && h*a.g+j<a.k;j++) {
            uint idx=r*kp+h*a.g+j;
            int code=a.bits==4 ? ((q[idx/2] >> ((idx%2)*4)) & 15) : q[idx];
            dot += float(code-qm)*x[b*a.k+h*a.g+j];
        }
        total += dot*s[r*ng+h];
    }
    y[id]=total;
}
kernel void linear_f(device const float *w [[buffer(0)]],
                     device const float *unused [[buffer(1)]],
                     device const float *x [[buffer(2)]],
                     device float *y [[buffer(3)]],
                     constant Shape &a [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if(id>=a.m*a.n) return;
    uint r=id%a.m,b=id/a.m; float total=0;
    for(uint j=0;j<a.k;j++) total += w[r*a.k+j]*x[b*a.k+j];
    y[id]=total;
}
