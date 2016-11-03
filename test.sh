#!/bin/sh

make -j4
./prime_cpu         100000
./prime_cpu_mt      100000
./prime_cuda        100000
./prime_cuda_new    100000
./prime_cuda_multi  100000

./prime_cpu         1000000
./prime_cpu_mt      1000000
./prime_cuda        1000000
./prime_cuda_new    1000000
./prime_cuda_multi  1000000

./prime_cpu         10000000
./prime_cpu_mt      10000000
./prime_cuda        10000000
./prime_cuda_new    10000000
./prime_cuda_multi  10000000

./prime_cpu         25000000
./prime_cpu_mt      25000000
./prime_cuda        25000000
./prime_cuda_new    25000000
./prime_cuda_multi  25000000

./prime_cpu         100000000
./prime_cpu_mt      100000000
./prime_cuda_multi  100000000
