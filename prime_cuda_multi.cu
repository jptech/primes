#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK (512)
#define DEFAULT_SIZE (160*160*THREADS_PER_BLOCK)

/* usage statement */
void usage()
{
	printf("Usage: prime_cuda N\n");
	exit(0);
}

__global__ void sum_primes( unsigned int *offset, unsigned int *N, unsigned long long *sum )
{
	unsigned int i, j, max;
	unsigned int index = (unsigned int) *offset + ((blockIdx.x * gridDim.y + blockIdx.y) * THREADS_PER_BLOCK) + threadIdx.x;
	unsigned long long blocksum = 0;
	__shared__ unsigned int block_arr[THREADS_PER_BLOCK];

	block_arr[threadIdx.x] = index;

	/* skip 0, 1, and anything beyond N */
	if(index < 2 || index >= *N) block_arr[threadIdx.x] = 0;

	/* determine last number to check in the primality check for this block */
	max = sqrt( (double) index + THREADS_PER_BLOCK );
	if( sqrt( (double) *N) < max ) max = sqrt( (double) *N );

	/* loop over each number in this block and check if it is divisable by i */
	for(i = threadIdx.x + 2; i <= max; i += THREADS_PER_BLOCK)
	{
		for(j = 0; j < THREADS_PER_BLOCK; j++)
		{
			if(block_arr[j] % i == 0 && block_arr[j] != i) block_arr[j] = 0;
		}
	}

	/* synchronize after the computation */
	__syncthreads();

	/* reduce the results from this block to a single value */
	if(threadIdx.x == 0)
	{
		for(i = 0; i < THREADS_PER_BLOCK; i++)
		{
			blocksum += block_arr[i];
		}
		/* add the sum from this block to the overall sum */
		atomicAdd(sum, blocksum);
	}
}

int main(int argc, char **argv)
{
	unsigned int N;
	unsigned int num_threads, num_blocks, blockx, blocky;
	unsigned int def_block_size, size_thresh;
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

	/* get the starting time before the prime summation call */
	gettimeofday(&tv, NULL);
	t0 = tv.tv_usec;
	t0 /= 1000000.0;
	t0 += tv.tv_sec;

	/* allocate the necessary CUDA device variables */
	cudaMalloc( (void **)&n_cuda, sizeof(unsigned int) );
	cudaMalloc( (void **)&off_cuda, sizeof(unsigned int) );
	cudaMalloc( (void **)&sum_cuda, sizeof(unsigned long long) );

	sum = 0;
	cudaMemcpy( sum_cuda, (void *) &sum, sizeof(unsigned long long), cudaMemcpyHostToDevice );

	def_block_size = DEFAULT_SIZE;
	size_thresh = 100000000;

	printf("Prime CUDA\n");

	for(offset = 0; offset < N; offset += def_block_size)
	{
		if(offset >= size_thresh)
		{
			size_thresh *= 4;
			def_block_size /= 2;
		}

		/* determine the subset of numbers to calculate for this CUDA kernel call */
		if(offset + def_block_size < N) size = def_block_size;
		else size = N - offset;

		subN = offset + size;

		/* determine how many blocks are needed */
		num_blocks = ceil( (double) size / THREADS_PER_BLOCK );
		num_threads = THREADS_PER_BLOCK;

		/* X and Y dimensions for the CUDA blocks */
		blockx = ceil( sqrt(num_blocks) );
		blocky = ceil( sqrt(num_blocks) );

		dim3 blocks(blockx, blocky);
		dim3 threads(num_threads);

		printf("subN: %d offset: %d size: %d\n", subN, offset, size);
		printf("Blocks: %d (x: %d y: %d) tpb: %d\n", num_blocks, blockx, blocky, num_threads);

		/* copy the variables for this run into the device memory */
		cudaMemcpy( n_cuda, (void *) &subN, sizeof(unsigned int), cudaMemcpyHostToDevice );
		cudaMemcpy( off_cuda, (void *) &offset, sizeof(unsigned int), cudaMemcpyHostToDevice );

		/* call the CUDA kernel */
		sum_primes <<< blocks, threads >>> ( off_cuda, n_cuda, sum_cuda );

		/* wait for it to finish */
		cudaDeviceSynchronize();

		/* check for errors */
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			/* print the CUDA error message and exit */
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

	}

	/* obtain the final result from the device memory */
	cudaMemcpy( &sum, sum_cuda, sizeof(unsigned long long), cudaMemcpyDeviceToHost );

	/* get the end time to calculate the total duration of the prime summation */
	gettimeofday(&tv, NULL);
	t1 = tv.tv_usec;
	t1 /= 1000000.0;
	t1 += tv.tv_sec;

	printf("N: %d\n", N);
	printf("sum of primes up to N: %lld\n", sum);
	printf("Time elapsed: %lf\n\n", t1 - t0);

	/* free device memory */
	cudaFree(n_cuda);
	cudaFree(off_cuda);
	cudaFree(sum_cuda);

	return 0;
}
