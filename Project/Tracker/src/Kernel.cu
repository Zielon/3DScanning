#include "cuda_runtime.h"
#include "device_launch_parameters.h"




__global__ void add(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}