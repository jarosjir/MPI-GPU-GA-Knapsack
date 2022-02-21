# 
# File:        Readme
# Author:      Jiri Jaros
# Affiliation: Brno University of Technology
#              Faculty of Information Technology
#              
#              and
# 
#              The Australian National University
#              ANU College of Engineering & Computer Science
#
# Email:       jarosjir@fit.vutbr.cz
# Web:         www.fit.vutbr.cz/~jarosjir
# 
# 
# License:     This source code is distribute under OpenSouce GNU GPL license
#                
#              If using this code, please consider citation of related papers
#              at http://www.fit.vutbr.cz/~jarosjir/pubs.php        
#      
#
# 
# Created on 06 April 2012, 00:00 PM
#


Outline

This package contains an efficient GPU implementation of the island model GA running the 
Knapsack benchmark. Each island is simulated on a single GPU. The migration is done via MPI

You can compile each version by entering the particular directory and typing:	make
If you want to run a simple demo, type:	make run

It is essential for you to set CUDA paths in GPU version of makefile to be able to compile it.
For more information visit: http://www.fit.vutbr.cz/~jarosjir/pubs.php?id=9830&shortname=1
and read the content of 
Jaros, J.: Multi-GPU Island-Based Genetic Algorithm Solving the Knapsack Problem, 
In: 2012 IEEE World Congress on Computational Intelligence, CA, US, IEEE, 2012, p. 217-224, 
ISBN 978-1-4673-1508-1


CPU Requirements:
Intel Core i7 (SSE4.1) for SSE 
Intel Core 2  (NO SSE)

GPU Requirements:
NVIDIA GTX 4XX series (architecture 2.0)

Software Requirements:
Compiler: 	g++-4.4 or newer
		nvcc-4.0 or newer 
		mpic++ 1.4.3 (1.5.4)


