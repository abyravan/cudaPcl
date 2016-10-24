/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <assert.h>
#include <nvidia/helper_cuda.h>

#define PI 3.141592653589793
#define UNASSIGNED 4294967294

/* computes b^TAb; 
 * assumes that A and b are in one piece in memory
 * written up for row-major A; but works for col major as well since
 * (b^TAb)^T = b^TA^Tb and row -> col major is transposing
 */
template<typename T>
__device__ inline T bTAb_3D(T *A, T *b)
{
  return b[0]*b[0]*A[0] + b[0]*b[1]*A[1] + b[0]*b[2]*A[2]
        +b[1]*b[0]*A[3] + b[1]*b[1]*A[4] + b[1]*b[2]*A[5]
        +b[2]*b[0]*A[6] + b[2]*b[1]*A[7] + b[2]*b[2]*A[8];
};

/* computes b^TAb; 
 * assumes that A and b are in one piece in memory
 * written up for row-major A; but works for col major as well since
 * (b^TAb)^T = b^TA^Tb and row -> col major is transposing
 */
template<typename T>
__device__ inline T bTAb_2D(T *A, T *b)
{
  return b[0]*b[0]*A[0] + b[0]*b[1]*A[1]
        +b[1]*b[0]*A[2] + b[1]*b[1]*A[3];
};

/* just base function - empty because we are specializing if you look down */
template<typename T>
__device__ inline T atomicAdd_(T* address, T val)
{};

/* atomic add for double */
template<>
__device__ inline double atomicAdd_<double>(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
          __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
};

template<>
__device__ inline float atomicAdd_<float>(float* address, float val)
{
  return atomicAdd(address,val);
};


