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

example
```
mpirun -np 20 ./ghio intensity 499 92 2000 20 20
```

This command will solve input `intensity` with 2000 iterations in each of 20 generations, and with 20 CPUs. However, please note that the input must be **root-squared** and **non-fftshifted**.

# Credits
GHIO was developed and invented by C.-C. Chen et al.; this code was primarily writtend by T.-Y Lan.
Currently it is being maintained by P.-N. Li since 2012.
