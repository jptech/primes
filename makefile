CC = gcc
CUDACC = nvcc
CFLAGS = -O2 -Wall -fomit-frame-pointer -funroll-loops -finline-functions
CUDAFLAGS = -O2 -Wno-deprecated-gpu-targets

all: prime_cpu prime_cpu_mt prime_cuda

clean:
	rm -rf *.o prime_cpu prime_cpu_mt prime_cuda

prime_cpu: prime_cpu.c
	$(CC) $(CFLAGS) prime_cpu.c -o prime_cpu -lm

prime_cpu_mt: prime_cpu_mt.c
	$(CC) $(CFLAGS) -fopenmp prime_cpu_mt.c -o prime_cpu_mt -lm

prime_cuda: prime_cuda_multi.cu
	$(CUDACC) $(CUDAFLAGS) prime_cuda_multi.cu -o prime_cuda -lm
