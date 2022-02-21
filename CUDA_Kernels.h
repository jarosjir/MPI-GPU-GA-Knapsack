/* 
 * File:        CUDA_Kernels.h
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
 * Created on 08 June 2012, 00:00 PM
 */

#ifndef CUDA_KERNELS_H
#define	CUDA_KERNELS_H



#include "GPU_Population.h"
#include "GPU_Statistics.h"
#include "GlobalKnapsackData.h"



/*
 * Simple binary GPU lock
 */
struct TGPU_Lock
{
    int *mutex;

    TGPU_Lock();
    ~TGPU_Lock();

    __device__ void Lock();     // lock 
    __device__ void Unlock();   // unlock
};// end of TGPU_Lock
//------------------------------------------------------------------------------

/*
 * Vector GPU lock
 */
struct TGPU_Vector_Lock
{
    int *mutex;
    int  size;
    
     TGPU_Vector_Lock(const int size);
    ~TGPU_Vector_Lock();

    __device__ void Lock  (const int Idx);     // Lock
    __device__ void Unlock(const int Idx);     // Unlock
    
}; // end of TGPU_Vector_Lock
//------------------------------------------------------------------------------


// First Population generation
__global__ void FirstPopulationGenerationKernel(TPopulationData * PopData, unsigned int RandomSeed);

//Genetic Manipulation (Selection, Crossover, Mutation)
__global__ void GeneticManipulationKernel(TPopulationData * ParentsData, TPopulationData * OffspringData, unsigned int RandomSeed);


//Replacement
__global__ void ReplacementKernel(TPopulationData * ParentsData, TPopulationData * OffspringData, unsigned int RandomSeed);


// Calculate statistics
__global__ void CalculateStatistics(TStatDataToExchange * StatisticsData, TPopulationData * PopData, TGPU_Lock Lock);

// Select individuals to migration
__global__ void SelectEmigrantsKernel(TPopulationData * ParentsData, TPopulationData * EmigrantsToSend, unsigned int RandomSeed);


// Find the location of the best
__device__ int FindTheBestLocation(int threadIdx1D, TPopulationData * ParentsData);

//accept emigrants
__global__ void AcceptEmigrantsKernel(TPopulationData * ParentsData, TPopulationData *  EmigrantsToReceive, TGPU_Vector_Lock  VectorLock, unsigned int RandomSeed);


// Calculate OneMax Fitness
__global__ void CalculateFintessOneMax(TPopulationData * PopData);


// Calculate Knapsack fitness
__global__ void CalculateKnapsackFintess(TPopulationData * PopData, TKnapsackData * GlobalData);



#endif	/* CUDA_KERNELS_H */

