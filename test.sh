#!/bin/sh

make -j4
./prime_cpu    100000
./prime_cpu_mt 100000
./prime_cuda   100000

./prime_cpu    1000000
./prime_cpu_mt 1000000
./prime_cuda   1000000

./prime_cpu    10000000
./prime_cpu_mt 10000000
./prime_cuda   10000000

./prime_cpu    25000000
./prime_cpu_mt 25000000
./prime_cuda   25000000
