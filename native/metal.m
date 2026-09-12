#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>
typedef struct { uint32_t m,k,n,g,bits; } Shape;
@interface EDContext : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLComputePipelineState> pipeline;
@property id<MTLBuffer> w;
@property id<MTLBuffer> scales;
@property id<MTLBuffer> x;
@property id<MTLBuffer> y;
@property Shape shape;
@end
@implementation EDContext
@end
void *ed_metal_create(const char *source, const void *w, size_t nw,
 const float *s,size_t ns,const float *x,int m,int k,int n,int g,int bits) {
 @autoreleasepool {
    EDContext *c=[EDContext new];c.device=MTLCreateSystemDefaultDevice();
    if(!c.device) return NULL;
    NSError *err=nil;
    MTLCompileOptions *opt=[MTLCompileOptions new];opt.fastMathEnabled=NO;
    id<MTLLibrary> lib=[c.device newLibraryWithSource:[NSString stringWithUTF8String:source] options:opt error:&err];
    if(!lib) { fprintf(stderr,"Metal compile: %s\n",err.localizedDescription.UTF8String);return NULL; }
    c.pipeline=[c.device newComputePipelineStateWithFunction:[lib newFunctionWithName:bits==32?@"linear_f":@"linear_q"] error:&err];
    if(!c.pipeline)return NULL;
    c.queue=[c.device newCommandQueue];
    c.w=[c.device newBufferWithBytes:w length:nw options:MTLResourceStorageModeShared];
    c.scales=[c.device newBufferWithBytes:s length:ns options:MTLResourceStorageModeShared];
    c.x=[c.device newBufferWithBytes:x length:n*k*sizeof(float) options:MTLResourceStorageModeShared];
    c.y=[c.device newBufferWithLength:n*m*sizeof(float) options:MTLResourceStorageModeShared];
    c.shape=(Shape){m,k,n,g,bits};
    if(!c.queue || !c.w || !c.scales || !c.x || !c.y)return NULL;
    return (__bridge_retained void *)c;
 }
}
int ed_metal_run(void *ptr,float *out) {
 @autoreleasepool {
    EDContext *c=(__bridge EDContext *)ptr;
    id<MTLCommandBuffer> cb=[c.queue commandBuffer];
    id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    [en setComputePipelineState:c.pipeline];
    [en setBuffer:c.w offset:0 atIndex:0];[en setBuffer:c.scales offset:0 atIndex:1];
    [en setBuffer:c.x offset:0 atIndex:2];[en setBuffer:c.y offset:0 atIndex:3];
    Shape a=c.shape;[en setBytes:&a length:sizeof(a) atIndex:4];
    NSUInteger tg=MIN((NSUInteger)128,c.pipeline.maxTotalThreadsPerThreadgroup);
    [en dispatchThreads:MTLSizeMake(a.m*a.n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError)return -1;
    memcpy(out,c.y.contents,a.m*a.n*sizeof(float));return 0;
 }
}
void ed_metal_free(void *ptr) { if(ptr) { id c=CFBridgingRelease(ptr); (void)c; } }
