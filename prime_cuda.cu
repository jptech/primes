#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK (1024)
#define NUM_BLOCKS (64)

void usage()
{
	printf("Usage: prime_cpu N\n");
	exit(0);
}

__global__ void sum_primes( int *N, unsigned long long *sum )
{
	unsigned int i, j, notprime;
	unsigned int maxj = 0;
	unsigned int maxN = *N;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long long block_sum = 0;

	__shared__ unsigned long long sums[THREADS_PER_BLOCK];

	sums[threadIdx.x] = 0;

	i = index;
	if(i < 2) i += (blockDim.x * NUM_BLOCKS);

	for(; i < maxN; i += (blockDim.x * NUM_BLOCKS))
	{
		notprime = 0;
		maxj = sqrt( (double) i);
		for(j = 2; j <= maxj; j++)
		{
			if(i % j == 0)
			{
				notprime = 1;
				break;
			}
		}
		if(notprime == 0) sums[threadIdx.x] += i;
	}

	__syncthreads();

	if(threadIdx.x == 0)
	{
		for(i = 0; i < THREADS_PER_BLOCK; i++)
		{
			block_sum += sums[i];
		}
		atomicAdd(sum, block_sum);
	}

}

int main(int argc, char **argv)
{
	int N;
	int num_threads, num_blocks;
	int *n_cuda;
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

	num_blocks = NUM_BLOCKS;
	num_threads = THREADS_PER_BLOCK;
	sum = 0;

	cudaMalloc( (void **)&n_cuda, sizeof(int) );
	cudaMalloc( (void **)&sum_cuda, sizeof(unsigned long long) );

	cudaMemcpy( n_cuda, (void *) &N, sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( sum_cuda, (void *) &sum, sizeof(unsigned long long), cudaMemcpyHostToDevice );

	sum_primes <<< num_blocks, num_threads >>> ( n_cuda, sum_cuda );

	cudaMemcpy( &sum, sum_cuda, sizeof(unsigned long long), cudaMemcpyDeviceToHost );

	gettimeofday(&tv, NULL);
	t1 = tv.tv_usec;
	t1 /= 1000000.0;
	t1 += tv.tv_sec;

	printf("N: %d\n", N);
	printf("sum of primes up to N: %ld\n", sum);
	printf("Time elapsed: %lf\n", t1 - t0);

	cudaFree(n_cuda);
	cudaFree(sum_cuda);

	return 0;
}
