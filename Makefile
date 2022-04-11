# @file        Makefile
# @author      Jiri Jaros
#              Brno University of Technology
#              Faculty of Information Technology
#
#              and
#
#              The Australian National University
#              ANU College of Engineering & Computer Science
#
#              jarosjir@fit.vutbr.cz
#              www.fit.vutbr.cz/~jarosjir
#
# @brief       Header file of the knapsack global data class.
#              Data resides in GPU memory
#              This class maintains the benchmark data
#
# @date        08 June      2012, 00:00 (created)
#              11 April     2022, 21:10 (revised)
#
# @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
#
# This source code is distribute under OpenSouce GNU GPL license.
# If using this code, please consider citation of related papers
# at http://www.fit.vutbr.cz/~jarosjir/pubs.php
#




# Environment
CC=nvcc
CXX=nvcc

CUDA_ARCH = --generate-code arch=compute_50,code=sm_50 \
            --generate-code arch=compute_52,code=sm_52 \
            --generate-code arch=compute_53,code=sm_53 \
            --generate-code arch=compute_60,code=sm_60 \
            --generate-code arch=compute_61,code=sm_61 \
            --generate-code arch=compute_62,code=sm_62 \
            --generate-code arch=compute_70,code=sm_70 \
            --generate-code arch=compute_72,code=sm_72 \
            --generate-code arch=compute_75,code=sm_75 \
            --generate-code arch=compute_80,code=sm_80 \
            --generate-code arch=compute_86,code=sm_86



CXXFLAGS=  -Xptxas=-v -m64 -O3  --device-c ${CUDA_ARCH}
TARGET=mpi_gpuga_knapsack

#---------------------------------------------------------------------
# CHECK FLAGS FOR MPI
LDFLAGS= -L${EBROOTOPENMPI} -lmpi -lmpi_cxx
#-----------------------------------------------------------------------

#----------------------------------------------------------------
# CHANGE PATHS to CUDA!!

CXXINCLUDE=-I${EBROOTCUDA}/include -I${EBROOTCUDA}/samples/common/inc 
#----------------------------------------------------------------

all:		$(TARGET)	

$(TARGET):	main.o CUDAKernels.o Statistics.o Parameters.o Population.o Evolution.o GlobalKnapsackData.o
	$(CXX) $(LDFLAGS) main.o CUDAKernels.o Statistics.o Parameters.o Population.o Evolution.o GlobalKnapsackData.o -lm -o $@ $(LIBS) 



main.o : main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

%.o : %.cu
	$(CXX) $(CXXFLAGS) ${CXXINCLUDE} -c $<


# Clean Targets
clean: 
	/bin/rm -f *.o *.~ $(TARGET)

# you are likely to change this configuration. Please be aware of folowing: 
# Do not use stride with mpi processes mapping. (node 1 (0-4) node2 (5-8) node3 (9-12), etc. 
# All nodes have to be equipped with the same number of GPUs
run:
	mpirun -np 1 ./mpi_gpuga_knapsack -f ./Data/knap_40.txt -p 100 -g 50 -s 10

	


