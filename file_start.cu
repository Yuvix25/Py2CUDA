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
}