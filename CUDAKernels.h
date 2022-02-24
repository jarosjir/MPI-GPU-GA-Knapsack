/*
 * File:        CUDAKernels.h
 * Author:      Jiri Jaros
 * Affiliation: Brno University of Technology
 *              Faculty of Information Technology
 *
 *              and
 *
 *              The Australian National University
 *              ANU College of Engineering & Computer Science
 *
 * Email:       jarosjir@fit.vutbr.cz
 * Web:         www.fit.vutbr.cz/~jarosjir
 *
 * Comments:    Header file of the GA evolution CUDA kernel
 *              This class controls the evolution process on a single GPU
 *
 *
 * License:     This source code is distribute under OpenSource GNU GPL license
 *
 *              If using this code, please consider citation of related papers
 *              at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 *
 *
 * Created on 08 June     2012, 00:00 PM
 * Revised on 22 February 2022, 12:00
 */

#ifndef CUDA_KERNELS_H
#define	CUDA_KERNELS_H


#include "Population.h"
#include "Statistics.h"
#include "GlobalKnapsackData.h"

/**
 * Check and report CUDA errors.
 * @param [in] sourceFileName   - Source file where the error happened.
 * @param [in] sourceLineNumber - Line where the error happened.
 */
void checkAndReportCudaError(const char* sourceFileName,
                             const int  sourceLineNumber);


// First Population generation
__global__ void FirstPopulationGenerationKernel(TPopulationData * PopData, unsigned int RandomSeed);

//Genetic Manipulation (Selection, Crossover, Mutation)
__global__ void GeneticManipulationKernel(TPopulationData * ParentsData, TPopulationData * OffspringData, unsigned int RandomSeed);


//Replacement
__global__ void ReplacementKernel(TPopulationData * ParentsData, TPopulationData * OffspringData, unsigned int RandomSeed);


// Calculate statistics
__global__ void CalculateStatistics(TStatDataToExchange * StatisticsData, TPopulationData * PopData);

// Select individuals to migration
__global__ void SelectEmigrantsKernel(TPopulationData * ParentsData, TPopulationData * EmigrantsToSend, unsigned int RandomSeed);


// Find the location of the best
__device__ int FindTheBestLocation(int threadIdx1D, TPopulationData * ParentsData);

//accept emigrants
__global__ void AcceptEmigrantsKernel(TPopulationData * ParentsData, TPopulationData *  EmigrantsToReceive, unsigned int RandomSeed);


// Calculate OneMax Fitness
__global__ void CalculateFintessOneMax(TPopulationData * PopData);


// Calculate Knapsack fitness
__global__ void CalculateKnapsackFintess(TPopulationData * PopData, TKnapsackData * GlobalData);



#endif	/* CUDA_KERNELS_H */

