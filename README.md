ghio-mpi
====
guided hybrid input-output, an improved HIO developed at Academia Sinica

```
ghio2d input_file input_size support_size n_itrs n_gens n_replica
```

# Compilation

example
```
mpicxx GHIO_2D.cpp -I/pkg/intel/12/mkl/include/fftw/ -mkl -o ghio2d
```

Replaced `/pkg/intel/12/mkl/include/fftw/` with the path of `fftw` in your working environment.

# Execution

```
mpirun -np NUM_OF_CPUS ./ghio INPUT_FILE INPUT_SIZE SUPPORT_SZ N_ITERS N_GENS N_COPIES
```

* `INPUT_FILE`: Square-root of Fourier intensity. Must be non-FFT-shifted.

* `INPUT_SIZE`: dimension of `INPUT_SIZE`

* `SUPPORT_SZ`: dimension of the square spport

* `N_ITERS`: number of iterations in a generation

* `N_GENS`: number of geneations

* `N_COPIES`: number of independent copies. Multiple of `NUM_CPUS` is recommended.


## example

```
mpirun -np 20 ./ghio fimg.cdi 1375 311 2000 10 20
```

This command will solve input `intensity` with 2000 iterations in each of 20 generations, and with 20 CPUs. However, please note that the input must be **root-squared** and **non-fftshifted**.

# Credits
GHIO was developed and invented by C.-C. Chen et al.; this code was primarily writtend by T.-Y Lan.
Currently it is being maintained by P.-N. Li since 2012.
