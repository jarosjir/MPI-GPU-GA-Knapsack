/**
 * @file        CUDAKernels.h
 * @author      Jiri Jarocu
 *              Brno University of Technology
 *              Faculty of Information Technology
 *
 *              and
 *
 *              The Australian National University
 *              ANU College of Engineering & Computer Science
 *
 *              jarosjir@fit.vutbr.cz
 *              www.fit.vutbr.cz/~jarosjir
 *
 * @brief       Header file of the GA evolution CUDA kernel
 *              This class controls the evolution process on a single GPU
 *
 * @date        08 June      2012, 00:00 (created)
 *              21 April     2022, 10:42 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */


#include <limits.h>
#include <stdexcept>

#include "Random123/philox.h"

#include "Population.h"
#include "Parameters.h"
#include "GlobalKnapsackData.h"

#include "CUDAKernels.h"

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

// Two different random generators
using RNG_2x32 = r123::Philox2x32;
using RNG_4x32 = r123::Philox4x32;

/// Extern variable for constant memory.
__constant__  EvolutionParameters gpuEvolutionParameters;

/**
 * @class Semaphore
 * Semaphore class for reduction kernels.
 */
class Semaphore
{
  public:
    /// Default constructor.
    Semaphore() = default;
    /// Default destructor.
    ~Semaphore() = default;

    /// Acquire semaphore.
    __device__ void acquire()
    {
      while (atomicCAS((int *)&mutex, 0, 1) != 0);
      __threadfence();
    }

    /// Release semaphore.
    __device__ void release()
    {
      mutex = 0;
      __threadfence();
    }

  private:
    /// Mutex for the semaphore.
    volatile int mutex = 0;
};// end of Semaphore
//----------------------------------------------------------------------------------------------------------------------

/// Global semaphore variable.
__device__ Semaphore semaphore;
/// Vector semaphore variable, MAX 256 locks!!!!!!
constexpr int kMaxVectorSemaphores = 256;
/// Array of semaphore for a vector lock
__device__ Semaphore vectorSemaphore[kMaxVectorSemaphores];


/**
 * Generate two random numbers.
 * @param key     - Key for the random generator.
 * @param counter - Counter for the random generator.
 * @return        - Two random values.
 */
__device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                   unsigned int counter);

/**
 * Shared memory reduction using a single warp.
 * @param [in, out] data      - data to be reduced.
 * @param [in]      threadIdx - CUDA thread idx.
 */
template<class T>
__device__ void halfWarpReduce(volatile T* data,
                               int         threadIdx);

/**
 * Get index into the array.
 * @params [in] chromosomeIdx - Chromosome index.
 * @params [in] geneIdx       - Gene index.
 * @returns Index into the population array.
 */
inline __device__ int getIndex(unsigned int chromosomeIdx,
                               unsigned int geneIdx);

/**
 * Select the better individual using the tournament selection.
 * @param [in] population   - Population of individuals.
 * @param [in] randomValue1 - Fist random value.
 * @param [in] randomValue2 - Second random value.
 * @return Index of the selected individual
 */
inline __device__ int selectBetter(const PopulationData* population,
                                   unsigned int          randomValue1,
                                   unsigned int          randomValue2);

/**
 * Select the worse individual using the tournament selection.
 * @param [in] population   - Population of individuals.
 * @param [in] randomValue1 - Fist random value.
 * @param [in] randomValue2 - Second random value.
 * @return Index of the selected individual.
 */
inline __device__ int selectWorse(const PopulationData* population,
                                  unsigned int          randomValue1,
                                  unsigned int          randomValue2);

/**
 * Perform uniform crossover on 32 genes.
 * @param [out] offspring1  - First offspring.
 * @param [out] offspring2  - Second offspring.
 * @param [in]  parent1     - First parent.
 * @param [in]  parent2     - Second parent.
 * @param [in]  mask        - Mask to perform crossover.
 */
inline __device__ void uniformCrossover(Gene&        offspring1,
                                        Gene&        offspring2,
                                        const Gene&  parent1,
                                        const Gene&  parent2,
                                        unsigned int mask);

/**
 * Perform bit flip mutation on a selected genes.
 * @param [in, out] offspring1   - first offspring to be mutated.
 * @param [in, out] offspring2   - second offspring to be mutated.
 * @param [in]      randomValue1 - first random value.
 * @param [in]      randomValue2 - second random values.
 * @param [in]      bitIdx       - bit to be flipped.
 */
inline __device__ void bitFlipMutation(Gene&        offspring1,
                                       Gene&        offspring2,
                                       unsigned int randomValue1,
                                       unsigned int randomValue2,
                                       int          bitIdx);

/**
 * Find the location of the best individual.
 * @param [in] population  - population to search through.
 * @param [in] threadIdx1D - thread idx.
 * @return Index of the best individual in the population.
 */
 __device__ int findTheBestLocation(const PopulationData* population,
                                    int                   threadIdx1D);


//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Implementation --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Device Functions --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Device random number generation.
 */
inline __device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                          unsigned int counter)
{
  RNG_2x32 rng;

  return rng({0, counter}, {key});
}// end of generateTwoRndValues
//----------------------------------------------------------------------------------------------------------------------


/**
 * Half warp reduce.
 */
template<class T>
__device__ void halfWarpReduce(volatile T* data,
                               int         threadIdx)
{
  if (threadIdx < WARP_SIZE / 2)
  {
    data[threadIdx] += data[threadIdx + 16];
    //__syncwarp();
    data[threadIdx] += data[threadIdx + 8];
    //__syncwarp();
    data[threadIdx] += data[threadIdx + 4];
    //__syncwarp();
    data[threadIdx] += data[threadIdx + 2];
    //__syncwarp();
    data[threadIdx] += data[threadIdx + 1];
  }
}// end of halfWarpReduce
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get index into population.
 */
inline __device__ int getIndex(unsigned int chromosomeIdx,
                               unsigned int geneIdx)
{
  return (chromosomeIdx * gpuEvolutionParameters.chromosomeSize + geneIdx);
}// end of getIndex
//----------------------------------------------------------------------------------------------------------------------


/**
 * Select the better individual.
 */
__device__ int selectBetter(const PopulationData* population,
                            unsigned int          randomValue1,
                            unsigned int          randomValue2)
{
  unsigned int idx1 = randomValue1 % (population->populationSize);
  unsigned int idx2 = randomValue2 % (population->populationSize);

  return (population->fitness[idx1] > population->fitness[idx2]) ? idx1 : idx2;
}// selectBetter
//----------------------------------------------------------------------------------------------------------------------

/**
 * Select the worse individual.
 */
inline __device__ int selectWorse(const PopulationData* population,
                                  unsigned int          randomValue1,
                                  unsigned int          randomValue2)
{
  unsigned int idx1 = randomValue1 % (population->populationSize);
  unsigned int idx2 = randomValue2 % (population->populationSize);

  return (population->fitness[idx1] < population->fitness[idx2]) ? idx1 : idx2;
}// Selection
//----------------------------------------------------------------------------------------------------------------------

/**
 * Uniform Crossover
 * Flip bites of parents to produce parents
 */
inline __device__ void uniformCrossover(Gene&        offspring1,
                                        Gene&        offspring2,
                                        const Gene&  parent1,
                                        const Gene&  parent2,
                                        unsigned int mask)
{
  offspring1 = (~mask & parent1) | ( mask & parent2);
  offspring2 = ( mask & parent1) | (~mask & parent2);
}// end of uniformCrossover
//----------------------------------------------------------------------------------------------------------------------

/**
 * BitFlip Mutation.
 * Invert selected bit
 */
inline __device__ void bitFlipMutation(Gene&        offspring1,
                                       Gene&        offspring2,
                                       unsigned int randomValue1,
                                       unsigned int randomValue2,
                                       int          bitIdx)
{
  if (randomValue1 < gpuEvolutionParameters.mutationUintBoundary)
  {
    offspring1 ^= (1 << bitIdx);
  }
  if (randomValue2< gpuEvolutionParameters.mutationUintBoundary)
  {
    offspring2 ^= (1 << bitIdx);
  }
}// end of bitFlipMutation
//----------------------------------------------------------------------------------------------------------------------

/**
 * Find the best solutions.
 */
__device__ int findTheBestLocation(const PopulationData* population,
                                   int                   threadIdx1D)
{
  __shared__ Fitness sharedMaxValue[BLOCK_SIZE];
  __shared__ int     sharedMaxIdx  [BLOCK_SIZE];

  // Clear shared buffer
  sharedMaxValue[threadIdx1D] = Fitness(0);
  sharedMaxIdx[threadIdx1D]   = 0;


  //  Reduction to shared memory
  for (int i = threadIdx1D; i < gpuEvolutionParameters.populationSize; i += BLOCK_SIZE)
  {
    const Fitness fitnessValue = population->fitness[i];

    if (fitnessValue > sharedMaxValue[threadIdx1D])
    {
      sharedMaxValue[threadIdx1D] = fitnessValue;
      sharedMaxIdx[threadIdx1D]   = i;
    }
  }

  __syncthreads();

  // Reduction in shared memory
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
  {
	  if (threadIdx1D < stride)
    {
      if (sharedMaxValue[threadIdx1D] < sharedMaxValue[threadIdx1D + stride])
      {
        sharedMaxValue[threadIdx1D] = sharedMaxValue[threadIdx1D + stride];
        sharedMaxIdx[threadIdx1D]   = sharedMaxIdx[threadIdx1D + stride];
      }
    }
	__syncthreads();
  }

  __syncthreads();

  return sharedMaxIdx[0];
}// end of findTheBestLocation
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Global Kernels ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Check and report CUDA errors.
 * if there's an error the code exits.
 */
void checkAndReportCudaError(const char* sourceFileName,
                             const int   sourceLineNumber)
{
  const cudaError_t cudaError = cudaGetLastError();

  if (cudaError != cudaSuccess)
  {
    fprintf(stderr,
            "Error in the CUDA routine: \"%s\"\nFile name: %s\nLine number: %d\n",
            cudaGetErrorString(cudaError),
            sourceFileName,
            sourceLineNumber);

    exit(EXIT_FAILURE);
  }
}// end of checkAndReportCudaError
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize Population before run.
 */
__global__ void cudaGenerateFirstPopulation(PopulationData* populationData,
                                            unsigned int    randomSeed)
{
  const size_t stride = blockDim.x * gridDim.x;

  RNG_2x32::ctr_type randomValues;

  const int nGenes = populationData->chromosomeSize * populationData->populationSize;

  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < nGenes)
  {
    randomValues = generateTwoRndValues(i, randomSeed);
    populationData->population[i] = randomValues.v[0];

    i += stride;
    if (i < nGenes)
    {
      populationData->population[i] = randomValues.v[1];
    }
    i += stride;
  }


   for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < populationData->populationSize; i += stride)
   {
      populationData->fitness[i]    = 0.0f;
   }

}// end of cudaGenerateFirstPopulation
//----------------------------------------------------------------------------------------------------------------------


/**
 * Genetic Manipulation (Selection, Crossover, Mutation).
 */
__global__ void cudaGeneticManipulation(const PopulationData* parentsData,
                                        PopulationData*       offspringData,
                                        unsigned int          randomSeed)
{
  const int geneIdx = threadIdx.x;
  const int chromosomeIdx = 2 * (threadIdx.y + blockIdx.y * blockDim.y);

  // Init random generator.
  RNG_4x32  rng_4x32;
  RNG_4x32::key_type key     = {{static_cast<unsigned int>(geneIdx), static_cast<unsigned int>(chromosomeIdx)}};
  RNG_4x32::ctr_type counter = {{0, 0, randomSeed ,0xbeeff00d}};
  RNG_4x32::ctr_type randomValues;


  // If having enough offsprings, return
  if (chromosomeIdx >= gpuEvolutionParameters.offspringPopulationSize)
  {
    return;
  }

  // Produce new offspring
  __shared__ int  parent1Idx[CHR_PER_BLOCK];
  __shared__ int  parent2Idx[CHR_PER_BLOCK];
  __shared__ bool crossoverFlag[CHR_PER_BLOCK];

  //---------------------------------------------- selection ---------------------------------------------------------//
  if ((threadIdx.y == 0) && (threadIdx.x < CHR_PER_BLOCK))
  {
    counter.incr();
    randomValues = rng_4x32(counter, key);

    parent1Idx[threadIdx.x] = selectBetter(parentsData, randomValues.v[0], randomValues.v[1]);
    parent2Idx[threadIdx.x] = selectBetter(parentsData, randomValues.v[2], randomValues.v[3]);

    counter.incr();
    randomValues = rng_4x32(counter, key);
    crossoverFlag[threadIdx.x] = randomValues.v[0] < gpuEvolutionParameters.crossoverUintBoundary;
  }

  __syncthreads();

  //-------------------------------------------- Manipulation  -------------------------------------------------------//

  // Go through two chromosomes and do uniform crossover and mutation
  for (int geneIdx = threadIdx.x; geneIdx < gpuEvolutionParameters.chromosomeSize; geneIdx += WARP_SIZE)
  {
    Gene geneParent1 = parentsData->population[getIndex(parent1Idx[threadIdx.y], geneIdx)];
    Gene geneParent2 = parentsData->population[getIndex(parent2Idx[threadIdx.y], geneIdx)];

    Gene geneOffspring1 = 0;
    Gene geneOffspring2 = 0;

    // Crossover
    if (crossoverFlag[threadIdx.y])
    {
      counter.incr();
      randomValues = rng_4x32(counter, key);
      uniformCrossover(geneOffspring1, geneOffspring2, geneParent1, geneParent2, randomValues.v[0]);
    }
    else
    {
      geneOffspring1 = geneParent1;
      geneOffspring2 = geneParent2;
    }


    // Mutation
    for (int bitId = 0; bitId < gpuEvolutionParameters.intBlockSize; bitId += 2)
    {
      counter.incr();
      randomValues = rng_4x32(counter, key);

      bitFlipMutation(geneOffspring1, geneOffspring2, randomValues.v[0], randomValues.v[1], bitId);
      bitFlipMutation(geneOffspring1, geneOffspring2, randomValues.v[2], randomValues.v[3], bitId + 1);
    }// for

    offspringData->population[getIndex(chromosomeIdx    , geneIdx)] = geneOffspring1;
    offspringData->population[getIndex(chromosomeIdx + 1, geneIdx)] = geneOffspring2;
  }
}// end of cudaGeneticManipulation
//----------------------------------------------------------------------------------------------------------------------

/**
 * Replacement kernel (Selection, Crossover, Mutation).
 */
__global__ void cudaReplacement(const PopulationData* parentsData,
                                PopulationData*       offspringData,
                                unsigned int          randomSeed)
{
  const int chromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;

  // Init random generator.
  RNG_2x32::ctr_type randomValues;
  __shared__ unsigned int offspringIdx[CHR_PER_BLOCK];

  // If having enough offsprings, return.
  if (chromosomeIdx >= gpuEvolutionParameters.populationSize)
  {
    return;
  }

  // Select offspring
  if (threadIdx.x == 0)
  {
    randomValues = generateTwoRndValues(chromosomeIdx, randomSeed);
    offspringIdx[threadIdx.y] = randomValues.v[0] % (gpuEvolutionParameters.offspringPopulationSize);
  }

  __syncthreads();


  // Replacement
  if (parentsData->fitness[chromosomeIdx] < offspringData->fitness[offspringIdx[threadIdx.y]])
  {
    // Copy data
    for (int geneIdx = threadIdx.x; geneIdx < gpuEvolutionParameters.chromosomeSize; geneIdx += WARP_SIZE)
    {
      parentsData->population[getIndex(chromosomeIdx, geneIdx)]
              = offspringData->population[getIndex(offspringIdx[threadIdx.y], geneIdx)];

    }

    if (threadIdx.x == 0)
    {
      parentsData->fitness[chromosomeIdx] = offspringData->fitness[offspringIdx[threadIdx.y]];
    }
  } // Replacement
}// end of cudaReplacement
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate statistics.
 */
__global__ void cudaCalculateStatistics(StatisticsData*       statisticsData,
                                        const PopulationData* populationData)
{
  __shared__ Fitness sharedMax[BLOCK_SIZE];
  __shared__ int     sharedMaxIdx[BLOCK_SIZE];
  __shared__ Fitness sharedMin[BLOCK_SIZE];

  __shared__ float sharedSum[BLOCK_SIZE];
  __shared__ float sharedSum2[BLOCK_SIZE];

  //Clear shared buffers
  sharedMax[threadIdx.x]    = Fitness(0);
  sharedMaxIdx[threadIdx.x] = 0;
  sharedMin[threadIdx.x]    = Fitness(UINT_MAX);

  sharedSum[threadIdx.x]  = 0.0f;;
  sharedSum2[threadIdx.x] = 0.0f;;

  __syncthreads();

  Fitness fitnessValue;

  // Reduction to shared memory
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < gpuEvolutionParameters.populationSize;
       i += blockDim.x * gridDim.x)
  {
    fitnessValue = populationData->fitness[i];
    if (fitnessValue > sharedMax[threadIdx.x])
    {
      sharedMax[threadIdx.x]    = fitnessValue;
      sharedMaxIdx[threadIdx.x] = i;
    }

    if (fitnessValue < sharedMin[threadIdx.x])
    {
      sharedMin[threadIdx.x] = fitnessValue;
    }

    sharedMin[threadIdx.x] = min(sharedMin[threadIdx.x], fitnessValue);

    sharedSum[threadIdx.x]  += fitnessValue;
    sharedSum2[threadIdx.x] += fitnessValue * fitnessValue;
  }

  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
	  if (threadIdx.x < stride)
    {
      if (sharedMax[threadIdx.x] < sharedMax[threadIdx.x + stride])
      {
        sharedMax[threadIdx.x]    = sharedMax[threadIdx.x + stride];
        sharedMaxIdx[threadIdx.x] = sharedMaxIdx[threadIdx.x + stride];
      }
      if (sharedMin[threadIdx.x] > sharedMin[threadIdx.x + stride])
      {
        sharedMin[threadIdx.x] = sharedMin[threadIdx.x + stride];
      }
      sharedSum[threadIdx.x]  += sharedSum[threadIdx.x + stride];
      sharedSum2[threadIdx.x] += sharedSum2[threadIdx.x + stride];
    }
	__syncthreads();
  }

  __syncthreads();

  // Write to Global Memory using a single thread per block and a semaphore
  if (threadIdx.x == 0)
  {
    semaphore.acquire();

    if (statisticsData->maxFitness < sharedMax[threadIdx.x])
    {
      statisticsData->maxFitness = sharedMax[threadIdx.x];
      statisticsData->indexBest  = sharedMaxIdx[threadIdx.x];
    }

    if (statisticsData->minFitness > sharedMin[threadIdx.x])
    {
      statisticsData->minFitness = sharedMin[threadIdx.x];
    }

    semaphore.release();

    atomicAdd(&(statisticsData->sumFitness),  sharedSum [threadIdx.x]);
    atomicAdd(&(statisticsData->sum2Fitness), sharedSum2[threadIdx.x]);
  }
}// end of cudaCalculateStatistics
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate Knapsack fitness.
 */
__global__ void cudaCalculateKnapsackFintess(PopulationData*     populationData,
                                             const KnapsackData* globalData)
{
  __shared__ PriceType  priceGlobalData[WARP_SIZE];
  __shared__ WeightType weightGlobalData[WARP_SIZE];

  __shared__ PriceType  priceValues[CHR_PER_BLOCK][WARP_SIZE];
  __shared__ WeightType weightValues[CHR_PER_BLOCK][WARP_SIZE];

  const int geneInBlockIdx = threadIdx.x;
  const int chromosomeIdx  = threadIdx.y + blockIdx.y * blockDim.y;

  // If not having anything to evaluate, return.
  if (chromosomeIdx >= populationData->populationSize)
  {
    return;
  }

  priceValues[threadIdx.y][threadIdx.x]  = PriceType(0);
  weightValues[threadIdx.y][threadIdx.x] = WeightType(0);

  // Calculate weight and price in parallel
  for (int intBlockIdx = 0; intBlockIdx < gpuEvolutionParameters.chromosomeSize; intBlockIdx++)
  {
    // Load Data
    if (threadIdx.y == 0)
    {
      priceGlobalData[geneInBlockIdx]  = globalData->itemPrice [intBlockIdx * gpuEvolutionParameters.intBlockSize
                                                                + geneInBlockIdx];
      weightGlobalData[geneInBlockIdx] = globalData->itemWeight[intBlockIdx * gpuEvolutionParameters.intBlockSize
                                                                + geneInBlockIdx];
    }

    const Gene actGene = ((populationData->population[getIndex(chromosomeIdx, intBlockIdx)]) >> geneInBlockIdx) &
                          Gene(1);

    __syncthreads();

    // Calculate Price and Weight
    priceValues[threadIdx.y][geneInBlockIdx]  += actGene * priceGlobalData[geneInBlockIdx];
    weightValues[threadIdx.y][geneInBlockIdx] += actGene * weightGlobalData[geneInBlockIdx];
  }

  // Everything above is warp synchronous.
  __syncwarp();
  halfWarpReduce(priceValues [threadIdx.y], threadIdx.x);
  halfWarpReduce(weightValues[threadIdx.y], threadIdx.x);
  __syncwarp();

  // write the result
  if (threadIdx.x == 0)
  {
    Fitness result = Fitness(priceValues[threadIdx.y][0]);

    // Penalize
    if (weightValues[threadIdx.y][0] > globalData->knapsackCapacity)
    {
      const Fitness penalty = (weightValues[threadIdx.y][0] - globalData->knapsackCapacity);

      result = result - globalData->maxPriceWightRatio * penalty;
      if (result < 0 ) result = Fitness(0);
    }

    populationData->fitness[chromosomeIdx] = result;
   } // if
}// end of cudaCalculateKnapsackFintess
//----------------------------------------------------------------------------------------------------------------------

/**
 * Select select emigrants and place them into new population
 */
__global__ void cudaSelectEmigrants(const PopulationData* parentsData,
                                    PopulationData*       emigrantsToSend,
                                    unsigned int          randomSeed)
{
  // Which chromosomes were selected (Idx)
  __shared__ int sharedEmigrantIdx[CHR_PER_BLOCK];

  const int      chromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;

  RNG_2x32::ctr_type randomValues;


  // Selection - let's use the first warp since CHR_PER_BLOCK is always smaller than WARP_SIZE.
  if ((threadIdx.y == 0) && (threadIdx.x < CHR_PER_BLOCK))
  {
    // chromosomeIdx is the same of all threadIdx.y, thus we need to add + threadIdx.x
    randomValues = generateTwoRndValues(chromosomeIdx + threadIdx.x, randomSeed);
    sharedEmigrantIdx[threadIdx.x] = selectBetter(parentsData, randomValues.v[0], randomValues.v[1]);
  }

  // Find the local island best solution by the 1st block.
  int localBestIdx = 0;
  if (blockIdx.y == 0)
  {
    __syncthreads();

    localBestIdx = findTheBestLocation(parentsData, threadIdx.y * blockDim.x + threadIdx.x);

    // First emigrant is the best solution, replace the selected
    if ((threadIdx.x == 0) && (threadIdx.y == 0))
    {
      sharedEmigrantIdx[0] = localBestIdx;
    }
  }

  __syncthreads();

  // Copy chromosomes
  if (chromosomeIdx < gpuEvolutionParameters.emigrantCount)
  {
    for (int geneIdx = threadIdx.x; geneIdx < gpuEvolutionParameters.chromosomeSize; geneIdx += WARP_SIZE)
    {
      emigrantsToSend->population[getIndex(chromosomeIdx, geneIdx)] =
               parentsData->population[getIndex(sharedEmigrantIdx[threadIdx.y], geneIdx)];
    }

    if (threadIdx.x == 0)
    {
      emigrantsToSend->fitness[chromosomeIdx] = parentsData->fitness[sharedEmigrantIdx[threadIdx.y]];
    }
  }
} // end of cudaSelectEmigrants
//----------------------------------------------------------------------------------------------------------------------


/**
 * Accept emigrants (give them into population).
 */
__global__ void cudaAcceptEmigrants(PopulationData*        parentsData,
                                    const PopulationData*  emigrantsToAccept,
                                    unsigned int           randomSeed)
{
  const int chromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;

  RNG_2x32::ctr_type randomValues;
  __shared__ int     sharedParrentToReplaceIdx[CHR_PER_BLOCK];

  if (chromosomeIdx >= gpuEvolutionParameters.emigrantCount)
  {
    return;
  }

  // Selection
  if (threadIdx.x == 0)
  {
    randomValues = generateTwoRndValues(chromosomeIdx, randomSeed);
    sharedParrentToReplaceIdx[threadIdx.y] = selectWorse(parentsData, randomValues.v[0], randomValues.v[1]);

    vectorSemaphore[sharedParrentToReplaceIdx[threadIdx.y] % kMaxVectorSemaphores].acquire();
  }


  // Replacement  - per warp computing - no barrier necessary
  if (parentsData->fitness[sharedParrentToReplaceIdx[threadIdx.y]] < emigrantsToAccept->fitness[chromosomeIdx])
  {
    // Copy data
    for (int geneIdx = threadIdx.x; geneIdx < gpuEvolutionParameters.chromosomeSize; geneIdx += WARP_SIZE)
    {
      parentsData->population[getIndex(sharedParrentToReplaceIdx[threadIdx.y], geneIdx)]
              = emigrantsToAccept->population[getIndex(chromosomeIdx, geneIdx)];

    }

    if (threadIdx.x == 0)
    {
      parentsData->fitness[sharedParrentToReplaceIdx[threadIdx.y]] = emigrantsToAccept->fitness[chromosomeIdx];
    }

  } // replacement


  // Only first thread unlocks the lock
  if (threadIdx.x == 0)
  {
    vectorSemaphore[sharedParrentToReplaceIdx[threadIdx.y] % kMaxVectorSemaphores].release();
  }
}// end of cudaAcceptEmigrants
//----------------------------------------------------------------------------------------------------------------------
