#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK (512)
#define DEFAULT_SIZE (10000000)
//#define NUM_BLOCKS (64)

void usage()
{
	printf("Usage: prime_cpu N\n");
	exit(0);
}

__global__ void sum_primes( unsigned int *offset, unsigned int *N, unsigned long long *sum )
{
	unsigned int index = (unsigned int) *offset + ((blockIdx.x * gridDim.y + blockIdx.y) * THREADS_PER_BLOCK) + threadIdx.x;
	unsigned int i;
	unsigned int j;
	unsigned int max;
	unsigned long long blocksum = 0;
	__shared__ unsigned int block_arr[THREADS_PER_BLOCK];

	block_arr[threadIdx.x] = index;

	if(index < 2 || index >= *N) block_arr[threadIdx.x] = 0;

	max = sqrt( (double) index + THREADS_PER_BLOCK );
	if( sqrt( (double) *N) < max ) max = sqrt( (double) *N );

	for(i = threadIdx.x + 2; i <= max; i += THREADS_PER_BLOCK)
	{
		for(j = 0; j < THREADS_PER_BLOCK; j++)
		{
			if(block_arr[j] % i == 0 && block_arr[j] != i) block_arr[j] = 0;
		}
	}

	__syncthreads();

	if(threadIdx.x == 0)
	{
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
	unsigned int *off_cuda;
	unsigned long long *sum_cuda;
	unsigned long long sum;
	struct timeval tv;
	double t0, t1;

	unsigned int subN, offset, size;

	if(argc != 2)
		usage();

	N = atoi(argv[1]);

	gettimeofday(&tv, NULL);
	t0 = tv.tv_usec;
	t0 /= 1000000.0;
	t0 += tv.tv_sec;

	cudaMalloc( (void **)&n_cuda, sizeof(unsigned int) );
	cudaMalloc( (void **)&off_cuda, sizeof(unsigned int) );
	cudaMalloc( (void **)&sum_cuda, sizeof(unsigned long long) );

	sum = 0;
	cudaMemcpy( sum_cuda, (void *) &sum, sizeof(unsigned long long), cudaMemcpyHostToDevice );

	printf("Prime CUDA - alternate\n");

	for(offset = 0; offset < N; offset += DEFAULT_SIZE)
	{
		if(offset + DEFAULT_SIZE < N) size = DEFAULT_SIZE;
		else size = N - offset;

		subN = offset + size;

		num_blocks = ceil( (double) size / THREADS_PER_BLOCK );
		num_threads = THREADS_PER_BLOCK;

		blockx = ceil( sqrt(num_blocks) );
		blocky = ceil( sqrt(num_blocks) );

		dim3 blocks(blockx, blocky);
		dim3 threads(num_threads);

		printf("subN: %d offset: %d size: %d\n", subN, offset, size);
		printf("Blocks: %d (x: %d y: %d) tpb: %d\n", num_blocks, blockx, blocky, num_threads);

		cudaMemcpy( n_cuda, (void *) &subN, sizeof(unsigned int), cudaMemcpyHostToDevice );
		cudaMemcpy( off_cuda, (void *) &offset, sizeof(unsigned int), cudaMemcpyHostToDevice );

		sum_primes <<< blocks, threads >>> ( off_cuda, n_cuda, sum_cuda );

		cudaDeviceSynchronize();

		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

	}

	cudaMemcpy( &sum, sum_cuda, sizeof(unsigned long long), cudaMemcpyDeviceToHost );

	gettimeofday(&tv, NULL);
	t1 = tv.tv_usec;
	t1 /= 1000000.0;
	t1 += tv.tv_sec;

	printf("N: %d\n", N);
	printf("sum of primes up to N: %lld\n", sum);
	printf("Time elapsed: %lf\n\n", t1 - t0);

	cudaFree(n_cuda);
	cudaFree(sum_cuda);

	return 0;
}
