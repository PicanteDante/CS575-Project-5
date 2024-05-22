// Monte Carlo simulation of golf balls:

// system includes
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <assert.h>
#include <malloc.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

// Setting up the configurations for NUMTRIALS and BLOCKSIZE:
const int numTrialsSizes[] = { 1024, 4096, 16384, 65536, 262144, 1048576, 2097152, 4194304, 8388608 };
const int blockSizeSizes[] = { 8, 32, 64, 128, 256, 512, 1024 };

#define GRAVITY 32.2f

const float BEFOREY = 70.f;
const float AFTERY = 10.f;
const float DISTX = 50.f;

__device__ const float RADIUS = 5.f; // so that the device (GPU) can see this variable

const float BEFOREYDY = 5.f;
const float AFTERYDY = 1.f;
const float DISTXDX = 5.f;

float* hbeforey;
float* haftery;
float* hdistx;
int* hsuccesses;

float Ranf(float low, float high) {
     float r = (float)rand();               // 0 - RAND_MAX
     float t = r / (float)RAND_MAX;       // 0. - 1.
     return low + t * (high - low);
}

// call this if you want to force your program to use
// a different random number sequence every time you run it:
void TimeOfDaySeed() {
     struct tm y2k = { 0 };
     y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
     y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

     time_t timer;
     time(&timer);
     double seconds = difftime(timer, mktime(&y2k));
     unsigned int seed = (unsigned int)(1000. * seconds);    // milliseconds
     srand(seed);
}

void CudaCheckError() {
     cudaError_t e = cudaGetLastError();
     if (e != cudaSuccess) {
          fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
     }
}

__device__ float Sqr(float x) {
     return x * x;
}

__device__ float Length(float dx, float dy) {
     return sqrt(Sqr(dx) + Sqr(dy));
}

#define IN
#define OUT

__global__ void MonteCarlo(IN float* dbeforey, IN float* daftery, IN float* ddistx, OUT int* dsuccesses) {
     dsuccesses[0] = 0;

     // randomize everything:
     float beforey = dbeforey[0];
     float aftery = daftery[0];
     float distx = ddistx[0];

     // Horizontal velocity of the ball
     float vx = sqrt(2.0f * GRAVITY * (beforey - aftery));

     // Time to fall from aftery to the ground
     float t = sqrt(2.0f * aftery / GRAVITY);

     // Horizontal distance traveled during the time of fall
     float x = vx * t;

     if (fabs(x - distx) <= RADIUS)
          dsuccesses[0] += 1;
}

int main(int argc, char* argv[]) {
     TimeOfDaySeed(); // seed the random number generator

     //   Print CSV headers
     fprintf(stderr, "NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond, probability");

     for (int i = 0; i < sizeof(numTrialsSizes) / sizeof(numTrialsSizes[0]); i++) {
          int NUMTRIALS = numTrialsSizes[i];
          for (int j = 0; j < sizeof(blockSizeSizes) / sizeof(blockSizeSizes[0]); j++) {
               int BLOCKSIZE = blockSizeSizes[j];
               int NUMBLOCKS = NUMTRIALS / BLOCKSIZE;

               // Allocate host memory:
               hbeforey = (float*)malloc(NUMTRIALS * sizeof(float));
               haftery = (float*)malloc(NUMTRIALS * sizeof(float));
               hdistx = (float*)malloc(NUMTRIALS * sizeof(float));
               hsuccesses = (int*)malloc(NUMTRIALS * sizeof(int));

               // Fill the random-value arrays:
               for (int n = 0; n < NUMTRIALS; n++) {
                    hbeforey[n] = Ranf(BEFOREY - BEFOREYDY, BEFOREY + BEFOREYDY);
                    haftery[n] = Ranf(AFTERY - AFTERYDY, AFTERY + AFTERYDY);
                    hdistx[n] = Ranf(DISTX - DISTXDX, DISTX + DISTXDX);
               }

               // Allocate device memory:
               float* dbeforey, * daftery, * ddistx;
               int* dsuccesses;
               cudaMalloc((void**)(&dbeforey), NUMTRIALS * sizeof(float));
               cudaMalloc((void**)(&daftery), NUMTRIALS * sizeof(float));
               cudaMalloc((void**)(&ddistx), NUMTRIALS * sizeof(float));
               cudaMalloc((void**)(&dsuccesses), NUMTRIALS * sizeof(int));

               CudaCheckError();

               // Copy host memory to the device:
               cudaMemcpy(dbeforey, hbeforey, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
               cudaMemcpy(daftery, haftery, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
               cudaMemcpy(ddistx, hdistx, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);

               CudaCheckError();

               // Setup the execution parameters:
               dim3 threads(BLOCKSIZE, 1, 1);
               dim3 grid(NUMBLOCKS, 1, 1);

               // Create and start timer
               cudaDeviceSynchronize();

               // Allocate CUDA events that we'll use for timing:
               cudaEvent_t start, stop;
               cudaEventCreate(&start);
               CudaCheckError();
               cudaEventCreate(&stop);
               CudaCheckError();

               // Record the start event:
               cudaEventRecord(start, NULL);
               CudaCheckError();

               // Execute the kernel:
               MonteCarlo<<< grid, threads >>> (dbeforey, daftery, ddistx, dsuccesses);

               // Record the stop event:
               cudaEventRecord(stop, NULL);

               // Wait for the stop event to complete:
               cudaEventSynchronize(stop);

               float msecTotal = 0.0f;
               cudaEventElapsedTime(&msecTotal, start, stop);
               CudaCheckError();

               // Copy result from the device to the host:
               cudaMemcpy(hsuccesses, dsuccesses, NUMTRIALS * sizeof(int), cudaMemcpyDeviceToHost);
               CudaCheckError();

               // Compute the sum:
               int numSuccesses = 0;
               //for (int i = 0; i < NUMTRIALS; i++) {
                    numSuccesses += hsuccesses[0];
               //}

               float probability = (float)numSuccesses / (float)NUMTRIALS;

               // Compute and print the performance:
               double secondsTotal = 0.001 * (double)msecTotal;
               double trialsPerSecond = (float)NUMTRIALS / secondsTotal;
               double megaTrialsPerSecond = trialsPerSecond / 1000000.;

               fprintf(stderr, "%10d , %8d , %10.4lf , %6.2f\n", NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond, 100.f * probability);
               //fprintf(stderr, "Number of Trials = %10d, Blocksize = %8d, MegaTrials/Second = %10.4lf, Probability = %6.2f%%\n", NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond, 100. * probability);

               // Clean up device memory:
               cudaFree(dbeforey);
               cudaFree(daftery);
               cudaFree(ddistx);
               cudaFree(dsuccesses);

               // Clean up host memory:
               free(hbeforey);
               free(haftery);
               free(hdistx);
               free(hsuccesses);

               CudaCheckError();
          }
     }
}