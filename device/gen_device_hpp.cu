/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 06:04:44 PM CET
 */

#ifndef MCGPU_DEVICE_GEN_DEVICE_HPP_CU
#define MCGPU_DEVICE_GEN_DEVICE_HPP_CU

#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>

int main( int argc, const char* argv[]) 
{
    static int nGpuArchCoresPerSM[] = { -1, 8, 32 };

    printf("#ifndef __DEVICE_HPP__\n");
    printf("#define __DEVICE_HPP__\n\n");

    int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		printf("// cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
        printf("\n#endif\n");
        exit(0);	
    }

    if (deviceCount == 0)
        printf("// There is no device supporting CUDA\n");

    int dev;
	int driverVersion = 0, runtimeVersion = 0;     
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("// There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("// There is 1 device supporting CUDA\n");
            else
                printf("// There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\n// Device %d: \"%s\"\n\n", dev, deviceProp.name);

        if (dev != 0)
            printf("#if 0\n");

    #if CUDART_VERSION >= 2020
        // Console log
		cudaDriverGetVersion(&driverVersion);
    		printf("  #define CUDA_Driver_Version             \"%d.%d\"\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
    		printf("  #define CUDA_Runtime_Version            \"%d.%d\"\n", runtimeVersion/1000, runtimeVersion%100);
    #endif
        printf("  #define CUDA_major                      %d\n", deviceProp.major);
        printf("  #define CUDA_minor                      %d\n", deviceProp.minor);

    		printf("  #define CUDA_global_memory_size         %u\n", (unsigned int) deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
        printf("  #define CUDA_multiProcessorCount        %d\n", deviceProp.multiProcessorCount);
        printf("  #define CUDA_numberOfCores              %d\n", nGpuArchCoresPerSM[deviceProp.major] * deviceProp.multiProcessorCount);
    #endif
        printf("  #define CUDA_totalConstMem              %u\n", (unsigned int) deviceProp.totalConstMem); 
        printf("  #define CUDA_sharedMemPerBlock          %u\n", (unsigned int) deviceProp.sharedMemPerBlock);
        printf("  #define CUDA_regsPerBlock               %d\n", deviceProp.regsPerBlock);
        printf("  #define CUDA_warpSize                   %d\n", deviceProp.warpSize);
        printf("  #define CUDA_maxThreadsPerBlock         %d\n", deviceProp.maxThreadsPerBlock);
        printf("  #define CUDA_maxThreadsDimX             %d\n", deviceProp.maxThreadsDim[0]);
        printf("  #define CUDA_maxThreadsDimY             %d\n", deviceProp.maxThreadsDim[1]);
        printf("  #define CUDA_maxThreadsDimZ             %d\n", deviceProp.maxThreadsDim[2]);
        printf("  #define CUDA_maxGridSizeX               %d\n", deviceProp.maxGridSize[0]);
        printf("  #define CUDA_maxGridSizeY               %d\n", deviceProp.maxGridSize[1]);
        printf("  #define CUDA_maxGridSizeZ               %d\n", deviceProp.maxGridSize[2]);
        printf("  #define CUDA_memPitch                   %u\n", (unsigned int) deviceProp.memPitch);
        printf("  #define CUDA_textureAlignment           %u\n", (unsigned int) deviceProp.textureAlignment);
        printf("  #define CUDA_clockRate                  %.2f\n", deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
        printf("  #define CUDA_deviceOverlap              %s\n", deviceProp.deviceOverlap ? "true" : "false");
    #endif
    #if CUDART_VERSION >= 2020
        printf("  #define CUDA_kernelExecTimeoutEnabled   %s\n", deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
        printf("  #define CUDA_integrated                 %s\n", deviceProp.integrated ? "true" : "false");
        printf("  #define CUDA_canMapHostMemory           %s\n", deviceProp.canMapHostMemory ? "true" : "false");
        printf("  #define CUDA_computeMode                \"%s\"\n", deviceProp.computeMode == cudaComputeModeDefault ?
                                    "Default (multiple host threads can use this device simultaneously)" :
                                    deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
                                    deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
    #endif
    #if CUDART_VERSION >= 3000
        printf("  #define CUDA_concurrentKernels          %s\n", deviceProp.concurrentKernels ? "true" : "false");
    #endif
    #if CUDART_VERSION >= 3010
        printf("  #define CUDA_ECCEnabled                 %s\n", deviceProp.ECCEnabled ? "true" : "false");
    #endif
        if (dev != 0)
            printf("#endif");

	}

    
    printf("\n#endif\n\n");
    exit(0);
}

#endif

