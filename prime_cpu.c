#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

void usage()
{
	printf("Usage: prime_cpu N\n");
	exit(0);
}

static inline int is_prime(int N)
{
	int i;
	int max;

	max = sqrt(N);

	if(N < 2) return -1;

	for(i = 2; i <= max; i++)
	{
		if(N % i == 0)
			return -1;
	}

	return 0;
}

long long sum_primes(int N)
{
	int i;
	long long sum = 0;

	for(i = 0; i < N; i++)
	{
		if(is_prime(i) == 0)
		{
			//printf("PRIME: %d\n", i);
			sum += i;
		}
		else
		{
			//printf("NOT PRIME: %d\n", i);
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

	gettimeofday(&tv, NULL);
	t0 = tv.tv_usec;
	t0 /= 1000000.0;
	t0 += tv.tv_sec;

	sum = sum_primes(N);

	gettimeofday(&tv, NULL);
	t1 = tv.tv_usec;
	t1 /= 1000000.0;
	t1 += tv.tv_sec;

	printf("N: %d\n", N);
	printf("sum of primes up to N: %ld\n", sum);
	printf("Time elapsed: %lf\n", t1 - t0);

	return 0;
}
