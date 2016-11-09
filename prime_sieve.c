#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

/* usage statement */
void usage()
{
	printf("Usage: prime_seive N\n");
	exit(0);
}

/* returns a summation of every prime number up to N */
long long sum_primes(int N)
{
	unsigned int i, j;
	unsigned int *primes;
	long long sum = 0;

	/* allocate an array to store N elements for the seive */
	primes = malloc( sizeof(unsigned int) * N );

	/* 0 and 1 are not prime */
	primes[0] = 0;
	primes[1] = 0;

	/* initialize the array */
	for(i = 2; i < N; i++)
	{
		primes[i] = 1;	
	}

	/* perform the seive starting at 2 */
	for(i = 2; i < N; i++)
	{
		if(primes[i] == 1)
		{
			for(j = 2; i * j < N; j++)
			{
				primes[i*j] = 0;
			}
		}
	}

	/* sum the results stored in the seive array */
	for(i = 0; i < N; i++)
	{
		if(primes[i] != 0)
		{
			sum += i;
		}
	}

	return sum;
}

int main(int argc, char **argv)
{
	int N;
	long long sum;
	double t0, t1;
	struct timeval tv;

	if(argc != 2)
		usage();

	N = atoi(argv[1]);

	/* get a start time before the sum_primes call */
	gettimeofday(&tv, NULL);
	t0 = tv.tv_usec;
	t0 /= 1000000.0;
	t0 += tv.tv_sec;

	sum = sum_primes(N);

	/* get an ending time after sum_primes returns */
	gettimeofday(&tv, NULL);
	t1 = tv.tv_usec;
	t1 /= 1000000.0;
	t1 += tv.tv_sec;

	printf("Prime CPU ST\n");
	printf("N: %d\n", N);
	printf("sum of primes up to N: %lld\n", sum);
	printf("Time elapsed: %lf\n\n", t1 - t0);

	return 0;
}
