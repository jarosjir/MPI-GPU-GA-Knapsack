MPI-GPU-GA-Knapsack
===================

## Outline

This package contains an efficient GPU implementation of the island model GA running the Knapsack benchmark. 
Each island is simulated on a single GPU. The migration is done via MPI.

## Compilation

- GPU Requirements:
  NVIDIA GTX 1080 series (architecture 6.0)

- Software Requirements:
  Compiler: 	g++ 9.3 or newer
		CUDA 10.0 or newer
		OpenMPI 4.0 or newer


You can compile each version by entering the particular directory and typing: 
```bash 
make
```

If you want to run a simple demo, type:
```bash
make run
```

It is essential for you to set CUDA paths in GPU version of makefile to be able to compile it.

## Reference
For more information visit: https://www.fit.vut.cz/research/publication/9860/?year=2012&author=Jaro%C5%A1
and read the content of 
Jaros, J.: Multi-GPU Island-Based Genetic Algorithm Solving the Knapsack Problem, 
In: 2012 IEEE World Congress on Computational Intelligence, CA, US, IEEE, 2012, p. 217-224, 
ISBN 978-1-4673-1508-1




