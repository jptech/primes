#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK (1024)
//#define NUM_BLOCKS (64)

void usage()
{
	printf("Usage: prime_cpu N\n");
	exit(0);
}

__global__ void sum_primes( unsigned int *N, unsigned long long *sum )
{
	unsigned int index = ((blockIdx.x * gridDim.y + blockIdx.y) * THREADS_PER_BLOCK) + threadIdx.x;
	unsigned int i;
	unsigned int j;
	unsigned int max;
	unsigned long long blocksum = 0;
	__shared__ unsigned int block_arr[THREADS_PER_BLOCK];

	block_arr[threadIdx.x] = index;
	if(index < 2 || index >= *N) block_arr[threadIdx.x] = 0;

	max = sqrt( (double) (blockIdx.x * gridDim.y + blockIdx.y + 1) * THREADS_PER_BLOCK );
	if( sqrt( (double) *N) < max ) max = sqrt( (double) *N );

	for(i = threadIdx.x + 2; i <= max; i += THREADS_PER_BLOCK)
	{
		for(j = 0; j < THREADS_PER_BLOCK; j++)
		{
			if(block_arr[j] % i == 0 && block_arr[j] != i) block_arr[j] = 0;
		}
	}

	/*
	for(i = 2; i <= sqrt( (double) index ); i++)
	{
		if(index % i == 0)
		{
			block_arr[threadIdx.x] = 0;
			break;
		}
	}
	*/

	__syncthreads();

	if(threadIdx.x == 0)
	{
		//printf("X: %d Y: %x | Xdim: %d Ydim: %d Zdim: %d\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, gridDim.z);
		for(i = 0; i < THREADS_PER_BLOCK; i++)
		{
			blocksum += block_arr[i];
		}
		atomicAdd(sum, blocksum);
	}
}

int main(int argc, char **argv)
{
	unsigned int N;
	unsigned int num_threads, num_blocks, blockx, blocky;
	unsigned int *n_cuda;
	unsigned long long *sum_cuda;
	unsigned long long sum;
	struct timeval tv;
	double t0, t1;

	if(argc != 2)
		usage();

	N = atoi(argv[1]);

	gettimeofday(&tv, NULL);
	t0 = tv.tv_usec;
	t0 /= 1000000.0;
	t0 += tv.tv_sec;

	num_blocks = ceil( (double) N / THREADS_PER_BLOCK);
	num_threads = THREADS_PER_BLOCK;

	blockx = ceil( sqrt(num_blocks) );
	blocky = ceil( sqrt(num_blocks) );

	dim3 blocks(blockx, blocky);
	dim3 threads(num_threads);

	sum = 0;

	printf("Prime CUDA - alternate\n");
	printf("Blocks: %d (x: %d y: %d) tpb: %d\n", num_blocks, blockx, blocky, num_threads);

	cudaMalloc( (void **)&n_cuda, sizeof(unsigned int) );
	cudaMalloc( (void **)&sum_cuda, sizeof(unsigned long long) );

	cudaMemcpy( n_cuda, (void *) &N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	cudaMemcpy( sum_cuda, (void *) &sum, sizeof(unsigned long long), cudaMemcpyHostToDevice );

	sum_primes <<< blocks, threads >>> ( n_cuda, sum_cuda );

	cudaDeviceSynchronize();

	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	cudaMemcpy( &sum, sum_cuda, sizeof(unsigned long long), cudaMemcpyDeviceToHost );

	gettimeofday(&tv, NULL);
	t1 = tv.tv_usec;
	t1 /= 1000000.0;
	t1 += tv.tv_sec;

	printf("N: %d\n", N);
	printf("sum of primes up to N: %ld\n", sum);
	printf("Time elapsed: %lf\n\n", t1 - t0);

	cudaFree(n_cuda);
	cudaFree(sum_cuda);

	return 0;
}
