#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>

__device__ double power(double a, double b){return pow(a, b);}

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

float __host_start_time__, __host_end_time__;
float __elapsed_device_function_time__;

cudaEvent_t __timer_start__, __timer_stop__;

void start_device_timer(){
    /* Call to start DEVICE timer which can be stopped with `stop_device_timer`, returns nothing.
    */
    __elapsed_device_function_time__ = 0;
    
    HANDLE_ERROR(cudaEventCreate(&__timer_start__));
    HANDLE_ERROR(cudaEventCreate(&__timer_stop__));

    HANDLE_ERROR(cudaEventRecord(__timer_start__, 0));
}

float stop_device_timer(){
    /* Call to stop DEVICE timer started with `start_device_timer`, returns elapsed time in seconds.
    */
    HANDLE_ERROR(cudaEventRecord(__timer_stop__, 0));
    HANDLE_ERROR(cudaEventSynchronize(__timer_stop__));

    HANDLE_ERROR(cudaEventElapsedTime(&__elapsed_device_function_time__, __timer_start__, __timer_stop__) );

    HANDLE_ERROR(cudaEventDestroy(__timer_start__));
    HANDLE_ERROR(cudaEventDestroy(__timer_stop__));
    return __elapsed_device_function_time__ / 1000;
}


void start_host_timer(){
    /* Call to start HOST timer which can be stopped with `stop_host_timer`, returns nothing.
    */
    __host_start_time__ = clock();
}

float stop_host_timer(){
    /* Call to stop HOST timer started with `start_host_timer`, returns elapsed time in seconds.
    */
    __host_start_time__ = clock();
    return (__host_end_time__ - __host_start_time__) / CLOCKS_PER_SEC;
}__global__ void find_perfects(){
    int thread_count = (blockDim.x*gridDim.x);
    int index = (threadIdx.x+(blockDim.x*blockIdx.x));
    int num = (index+2);
    if ((index==1000000)){
        printf("%s%s", "Found!\n", "\n");
    };
    while (true){
        int sum = 1;
        float sqaure_root = sqrt((double)(num));
        for (int div = 2; div<ceil((double)(sqaure_root)); div++){
            if ((div==sqaure_root)){
                sum += sqaure_root;
            }
            else if (((num%div)==0)){
                sum += (div+((double)(num)/(double)(div)));
            };
        };
        if ((sum==num)){
            printf("%s%d%s", "%d\n", num, "\n");
        };
        num += thread_count;
    };
};
__device__ double f(double x){
    return sqrt((double)((1-power(x, 2))));
};
__global__ void integral(double* num){
    int a = -1;
    int b = 1;
    int thread_count = (blockDim.x*gridDim.x);
    int index = (threadIdx.x+(blockDim.x*blockIdx.x));
    atomicAdd(num, ((double)(((2*f((double)((((double)(((b-a)*index))/(double)(thread_count))+a))))*(b-a)))/(double)(thread_count)));
};
int main(){
    double* num;
    cudaMallocManaged(&num, 1*sizeof(double*));
    num[0] = 0;
    start_device_timer();
    integral<<<65535, 1024>>>((double*)(num));
    float elapsed = stop_device_timer();
    cudaDeviceSynchronize();
    printf("%.10f%s", elapsed, "\n");
    printf("%.10lf%s", num[0], "\n");
    cudaFree(num);
    return 0;
};
