/*
 * File:        CUDAKernels.cu
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
 * Revised on 24 February 2022, 16:17 PM
 */


#include <limits.h>
#include <stdio.h>
#include "Random123/philox.h"

#include <stdexcept>

#include "Population.h"
#include "Parameters.h"
#include "GlobalKnapsackData.h"


#include "CUDAKernels.h"

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

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
__device__ Semaphore vectorSemaphore[kMaxVectorSemaphores];


/*
 * Vector GPU lock
 */
template<int maxSize>
class VectorSemaphore
{
  public:
    /// Default constructor is not allowed
    VectorSemaphore() = delete;
    /// Default specifying the size of the lock.
    VectorSemaphore(const int size = 256) : size(size)
    {
      if (size > maxSize)
      {
        throw std::out_of_range("Maximum size of the vector lock exceeded!");
      }
    };
    /// Destructor
    ~VectorSemaphore();

    /**
     * Acquire one of semaphore.
     * @param [in] idx - Id of the semaphore to acquire.
     */
    __device__ void acquire(const int idx)
    {
       while (atomicCAS((int *)&mutex[idx], 0, 1) != 0);
      __threadfence();
    };
    /**
     * Release one of semaphore.
     * @param [in] idx - Id of the semaphore to release.
     */
    __device__ void release(const int idx)
    {
      mutex[idx] = 0;
      __threadfence();
    };


  private:
    int mutex[maxSize] {0};
    const int size;
}; // end of VectorSemaphore
//----------------------------------------------------------------------------------------------------------------------

/// Global semaphore variable.
//__device__ VectorSemaphore<256> vectorSemaphore();



using namespace r123;

// two different random generators
typedef r123::Philox2x32 RNG_2x32;
typedef r123::Philox4x32 RNG_4x32;

// Generate two random numbers
__device__ void TwoRandomINTs (RNG_2x32::ctr_type *RandomValues, unsigned int Key, unsigned int Counter);


// Get index of array
inline __device__ int  GetIndex(unsigned int ChromosomeIdx, unsigned int GeneIdx);

// Select an individual from a population
inline __device__ int  Selection(PopulationData * ParentsData, unsigned int Random1, unsigned int Random2);
// Select an individual pick the worse (for replacement)
inline __device__ int  Selection_Worse(PopulationData * ParentsData, unsigned int Random1, unsigned int Random2);

// Perform Uniform Crossover
inline __device__ void CrossoverUniformFlip(Gene& GeneOffspring1, Gene& GeneOffspring2,
                                            Gene GeneParent1    , Gene GeneParent2,
                                            unsigned int RandomValue);

// Perform BitFlip mutation
inline __device__ void MutationBitFlip(Gene& GeneOffspring1, Gene& GeneOffspring2,
                                       unsigned int RandomValue1,unsigned int RandomValue2, int BitID);

// Reduction kernels
__device__ void HalfWarpReducePrice (volatile PriceType * sdata, int tid);
__device__ void HalfWarpReduceWeight(volatile WeightType* sdata, int tid);
__device__ void HalfWarpReduceGene(volatile Gene* sdata, int tid);


//----------------------------------------------------------------------------//
//                              Kernel implementation                         //
//----------------------------------------------------------------------------//



//----------------------------------------------------------------------------//
//                                DeviceFunctions Kernels                     //
//----------------------------------------------------------------------------//







/*
 * Device random number generation
 *
 * @param RandomValues - Returned random values
 * @param Key
 * @param Counter
 *
 */
inline __device__ void TwoRandomINTs(RNG_2x32::ctr_type *RandomValues,
                                     unsigned int Key, unsigned int Counter){
    RNG_2x32 rng;

    RNG_2x32::ctr_type counter={{0,Counter}};
    RNG_2x32::key_type key={{Key}};

    *RandomValues = rng(counter, key);
}// end of TwoRandomINTs
//------------------------------------------------------------------------------



/*
 * GetIndex to population
 * @param ChromosomeIdx
 * @param genIdx
 * @return 1D index
 *
 */
inline __device__ int  GetIndex(unsigned int ChromosomeIdx, unsigned int GeneIdx){

    return (ChromosomeIdx * gpuEvolutionParameters.chromosomeSize + GeneIdx);

}// end of GetIndex
//------------------------------------------------------------------------------

/*
 * Half warp reduce for price
 *
 * @param sdata  - data to reduce
 * @param tid    - idx of thread
 *
 */
__device__ void HalfWarpReducePrice(volatile PriceType * sdata, int tid){
    if (tid < WARP_SIZE/2) {
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
}// end of HalfWarpReducePrice
//------------------------------------------------------------------------------


/*
 * Half warp reduce for Weight
 *
 * @param sdata  - data to reduce
 * @param tid    - idx of thread
 *
 *
 */
__device__ void HalfWarpReduceWeight(volatile WeightType* sdata, int tid){
    if (tid < WARP_SIZE/2) {
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
}// end of HalfWarpReducePrice
//------------------------------------------------------------------------------


/*
 * Half Warp reduction for TGene
 *
 * @param sdata  - data to reduce
 * @param tid    - idx of thread
 */
__device__ void HalfWarpReduceGene(volatile Gene* sdata, int tid){
    if (tid < WARP_SIZE/2) {
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
}// end of HalfWarpReduceGene
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                                DeviceFunctions Kernels                     //
//----------------------------------------------------------------------------//



/*
 * Select one individual
 * @param ParentData - Parent population
 * @param Random1    - First random value
 * @param Random2    - Second random value
 * @return           - idx of the selected individual
 */
inline __device__ int Selection(PopulationData * ParentsData, unsigned int Random1, unsigned int Random2){


    unsigned int Idx1 = Random1 % (ParentsData->populationSize);
    unsigned int Idx2 = Random2 % (ParentsData->populationSize);

    return (ParentsData->fitness[Idx1] > ParentsData->fitness[Idx2]) ? Idx1 : Idx2;
}// Selection
//------------------------------------------------------------------------------


/*
 * Select one individual (the worse one)
 * @param ParentData - Parent population
 * @param Random1    - First random value
 * @param Random2    - Second random value
 * @return           - idx of the selected individual
 */
inline __device__ int Selection_Worse(PopulationData * ParentsData, unsigned int Random1, unsigned int Random2){


    unsigned int Idx1 = Random1 % (ParentsData->populationSize);
    unsigned int Idx2 = Random2 % (ParentsData->populationSize);

    return (ParentsData->fitness[Idx1] < ParentsData->fitness[Idx2]) ? Idx1 : Idx2;
}// Selection
//------------------------------------------------------------------------------


/*
 * Uniform Crossover
 * Flip bites of parents to produce parents
 *
 * @param       GeneOffspring1 - Returns first offspring (one gene)
 * @param       GeneOffspring2 - Returns second offspring (one gene)
 * @param       GeneParent1    - First parent (one gene)
 * @param       GeneParent2    - Second parent (one gene)
 * @param       Mask           - Random value for mask
 *
 */
inline __device__ void CrossoverUniformFlip(Gene& GeneOffspring1, Gene& GeneOffspring2,
                                            Gene GeneParent1    , Gene GeneParent2,
                                            unsigned int RandomValue){


    GeneOffspring1 =  (~RandomValue  & GeneParent1) | ( RandomValue  & GeneParent2);
    GeneOffspring2 =  ( RandomValue  & GeneParent1) | (~RandomValue  & GeneParent2);


}// end of CrossoverUniformFlip
//------------------------------------------------------------------------------



/*
 * BitFlip Mutation
 * Invert selected bit
 *
 * @param       GeneOffspring1 - Returns first offspring (one gene)
 * @param       GeneOffspring2 - Returns second offspring (one gene)
 * @param       RandomValue1   - Random value 1
 * @param       RandomValue2   - Random value 2
 * @param       BitID          - Bit to mutate

 */
inline __device__ void MutationBitFlip(Gene& GeneOffspring1, Gene& GeneOffspring2,
                                      unsigned int RandomValue1,unsigned int RandomValue2, int BitID){

  //GeneOffspring1 ^= ((unsigned int)(RandomValue1 < GPU_EvolutionParameters.MutationUINTBoundary) << BitID);
  //GeneOffspring2 ^= ((unsigned int)(RandomValue2 < GPU_EvolutionParameters.MutationUINTBoundary) << BitID);
  if (RandomValue1 < gpuEvolutionParameters.mutationUintBoundary) GeneOffspring1 ^= (1 << BitID);
  if (RandomValue2 < gpuEvolutionParameters.mutationUintBoundary) GeneOffspring2 ^= (1 << BitID);



}// end of MutationBitFlip
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                                Global Kernels                              //
//----------------------------------------------------------------------------//


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

/*
 * Initialize Population before run
 * @params   Population
 * @params   RandomNumbers
 *
 */
__global__ void FirstPopulationGenerationKernel(PopulationData * PopData, unsigned int RandomSeed){

   size_t i      = threadIdx.x + blockIdx.x*blockDim.x;
   size_t stride = blockDim.x * gridDim.x;

   RNG_2x32::ctr_type RandomValues;

   const int PopulationDIM = PopData->chromosomeSize * PopData->populationSize;

   while (i < PopulationDIM) {

       TwoRandomINTs(&RandomValues, i, RandomSeed);
       PopData->population[i] = RandomValues.v[0];

       i += stride;
       if (i < PopulationDIM) {
          PopData->population[i] = RandomValues.v[1];
       }
       i += stride;
    }

   i  = threadIdx.x + blockIdx.x*blockDim.x;
   while (i < PopData->populationSize){
        PopData->fitness[i]    = 0.0f;
        i += stride;
   }

}// end of PopulationInitializationKernel
//------------------------------------------------------------------------------






/*
 * Genetic Manipulation (Selection, Crossover, Mutation)
 *
 * @param ParentsData
 * @param OffspringData
 * @param RandomSeed
 *
 */
__global__ void GeneticManipulationKernel(PopulationData * ParentsData, PopulationData * OffspringData,
                                    unsigned int RandomSeed){
    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = 2* (threadIdx.y + blockIdx.y * blockDim.y);

    //-- Init Random --//
    RNG_4x32  rng_4x32;
    RNG_4x32::key_type key    ={{GeneIdx, ChromosomeIdx}};
    RNG_4x32::ctr_type counter={{0, 0, RandomSeed ,0xbeeff00d}};
    RNG_4x32::ctr_type RandomValues;




    if (ChromosomeIdx >= gpuEvolutionParameters.offspringPopulationSize) return;

    __shared__ int  Parent1_Idx  [CHR_PER_BLOCK];
    __shared__ int  Parent2_Idx  [CHR_PER_BLOCK];
    __shared__ bool CrossoverFlag[CHR_PER_BLOCK];

    //------------------------------------------------------------------------//
    //------------------------ selection -------------------------------------//
    //------------------------------------------------------------------------//
    if ((threadIdx.y == 0) && (threadIdx.x < CHR_PER_BLOCK)){
        counter.incr();
        RandomValues = rng_4x32(counter, key);

        Parent1_Idx[threadIdx.x] = Selection(ParentsData, RandomValues.v[0], RandomValues.v[1]);
        Parent2_Idx[threadIdx.x] = Selection(ParentsData, RandomValues.v[2], RandomValues.v[3]);

        counter.incr();
        RandomValues = rng_4x32(counter, key);
        CrossoverFlag[threadIdx.x] = RandomValues.v[0] < gpuEvolutionParameters.crossoverUintBoundary;
    }


    __syncthreads(); // to distribute the selected chromosomes
    //------------------------------------------------------------------------//
    //------------------------ Manipulation  ---------------------------------//
    //------------------------------------------------------------------------//

    //-- Go through two chromosomes and do uniform crossover and mutation--//
    while (GeneIdx < gpuEvolutionParameters.chromosomeSize){
        Gene GeneParent1 = ParentsData->population[GetIndex(Parent1_Idx[threadIdx.y], GeneIdx)];
        Gene GeneParent2 = ParentsData->population[GetIndex(Parent2_Idx[threadIdx.y], GeneIdx)];

        Gene GeneOffspring1 = 0;
        Gene GeneOffspring2 = 0;

        //-- crossover --//
        if (CrossoverFlag[threadIdx.y]) {

            counter.incr();
            RandomValues = rng_4x32(counter, key);
            CrossoverUniformFlip(GeneOffspring1, GeneOffspring2, GeneParent1, GeneParent2, RandomValues.v[0]);

        } else {
            GeneOffspring1 = GeneParent1;
            GeneOffspring2 = GeneParent2;
        }


        //-- mutation --//
        for (int BitID = 0; BitID < gpuEvolutionParameters.intBlockSize; BitID+=2){

            counter.incr();
            RandomValues = rng_4x32(counter, key);

            MutationBitFlip(GeneOffspring1, GeneOffspring2, RandomValues.v[0],RandomValues.v[1], BitID);
            MutationBitFlip(GeneOffspring1, GeneOffspring2, RandomValues.v[2],RandomValues.v[3], BitID+1);


         }// for

        OffspringData->population[GetIndex(ChromosomeIdx  , GeneIdx)] = GeneOffspring1;
        OffspringData->population[GetIndex(ChromosomeIdx+1, GeneIdx)] = GeneOffspring2;

        GeneIdx += WARP_SIZE;
    }


}// end of GeneticManipulation
//------------------------------------------------------------------------------




/*
 * Replacement kernel (Selection, Crossover, Mutation)
 *
 * @param ParentsData
 * @param OffspringData
 * @param RandomSeed
 */
__global__ void ReplacementKernel(PopulationData * ParentsData, PopulationData * OffspringData, unsigned int RandomSeed){



    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;

    //-- Init Random --//
    RNG_2x32::ctr_type RandomValues;
    __shared__ unsigned int OffspringIdx_SHM[CHR_PER_BLOCK];

    if (ChromosomeIdx >= gpuEvolutionParameters.populationSize) return;


    //-- select offspring --//
    if (threadIdx.x == 0){
       TwoRandomINTs(&RandomValues, ChromosomeIdx, RandomSeed);
       OffspringIdx_SHM[threadIdx.y]  = RandomValues.v[0] % (gpuEvolutionParameters.offspringPopulationSize);

    }

    __syncthreads();


    //------- replacement --------//
    if (ParentsData->fitness[ChromosomeIdx] < OffspringData->fitness[OffspringIdx_SHM[threadIdx.y]]){

        //-- copy data --//
        while (GeneIdx < gpuEvolutionParameters.chromosomeSize){
            ParentsData->population[GetIndex(ChromosomeIdx, GeneIdx)] = OffspringData->population[GetIndex(OffspringIdx_SHM[threadIdx.y], GeneIdx)];
            GeneIdx +=  WARP_SIZE;
        }

        if (threadIdx.x == 0) ParentsData->fitness[ChromosomeIdx] = OffspringData->fitness[OffspringIdx_SHM[threadIdx.y]];

    } // replacement

}// end of ReplacementKernel
//------------------------------------------------------------------------------






/*
 * Calculate statistics
 *
 * @param StatisticsData
 * @param PopopulationData
 * @param GPULock
 *
 */
__global__ void CalculateStatistics(StatisticsData * StatisticsData, PopulationData * PopData){

  int i      = threadIdx.x + blockDim.x*blockIdx.x;
  int stride = blockDim.x*gridDim.x;

  __shared__ Fitness shared_Max    [BLOCK_SIZE];
  __shared__ int      shared_Max_Idx[BLOCK_SIZE];
  __shared__ Fitness shared_Min    [BLOCK_SIZE];

  __shared__ float shared_Sum    [BLOCK_SIZE];
  __shared__ float shared_Sum2   [BLOCK_SIZE];


    //-- Clear shared buffer --//

  shared_Max    [threadIdx.x] = Fitness(0);
  shared_Max_Idx[threadIdx.x] = 0;
  shared_Min    [threadIdx.x] = Fitness(UINT_MAX);

  shared_Sum    [threadIdx.x] = 0.0f;;
  shared_Sum2   [threadIdx.x] = 0.0f;;

  __syncthreads();

  Fitness FitnessValue;

  //-- Reduction to shared memory --//
  while (i < gpuEvolutionParameters.populationSize){

      FitnessValue = PopData->fitness[i];
      if (FitnessValue > shared_Max[threadIdx.x]){
          shared_Max    [threadIdx.x] = FitnessValue;
          shared_Max_Idx[threadIdx.x] = i;
      }

      if (FitnessValue < shared_Min[threadIdx.x]){
          shared_Min    [threadIdx.x] = FitnessValue;
      }

      shared_Sum [threadIdx.x] += FitnessValue;
      shared_Sum2[threadIdx.x] += FitnessValue*FitnessValue;

      i += stride;
  }

  __syncthreads();

  //-- Reduction in shared memory --//

  for (int stride = blockDim.x/2; stride > 0; stride /= 2){
  	if (threadIdx.x < stride) {
            if (shared_Max[threadIdx.x] < shared_Max[threadIdx.x + stride]){
               shared_Max    [threadIdx.x] = shared_Max    [threadIdx.x + stride];
               shared_Max_Idx[threadIdx.x] = shared_Max_Idx[threadIdx.x + stride];
            }
            if (shared_Min[threadIdx.x] > shared_Min[threadIdx.x + stride]){
               shared_Min [threadIdx.x] = shared_Min[threadIdx.x + stride];
            }
            shared_Sum [threadIdx.x] += shared_Sum [threadIdx.x + stride];
            shared_Sum2[threadIdx.x] += shared_Sum2[threadIdx.x + stride];
        }
	__syncthreads();
  }

  __syncthreads();


  // Write to Global Memory using a single thread per block and a semaphore
  if (threadIdx.x == 0)
  {
    semaphore.acquire();

    if (StatisticsData->maxFitness < shared_Max[threadIdx.x])
    {
      StatisticsData->maxFitness = shared_Max[threadIdx.x];
      StatisticsData->indexBest  = shared_Max_Idx[threadIdx.x];
    }

    if (StatisticsData->minFitness > shared_Min[threadIdx.x])
    {
      StatisticsData->minFitness = shared_Min[threadIdx.x];
    }

    semaphore.release();
    atomicAdd(&(StatisticsData->sumFitness),  shared_Sum [threadIdx.x]);
    atomicAdd(&(StatisticsData->sum2Fitness), shared_Sum2[threadIdx.x]);
  }

}// end of CalculateStatistics
//------------------------------------------------------------------------------





/*
 * Calculate Knapsack fitness
 *
 * Each warp working with 1 32b gene. Diferent warps different individuals
 *
 * @param PopData
 * @param GlobalData
 *
 */
__global__ void CalculateKnapsackFintess(PopulationData * PopData, KnapsackData * GlobalData){


    __shared__ PriceType  PriceGlobalData_SHM [WARP_SIZE];
    __shared__ WeightType WeightGlobalData_SHM[WARP_SIZE];


    __shared__ PriceType  PriceValues_SHM [CHR_PER_BLOCK] [WARP_SIZE];
    __shared__ WeightType WeightValues_SHM[CHR_PER_BLOCK] [WARP_SIZE];


    int GeneInBlockIdx = threadIdx.x;
    int ChromosomeIdx  = threadIdx.y + blockIdx.y * blockDim.y;

    if (ChromosomeIdx >= PopData->populationSize) return;

    Gene ActGene;

    //------------------------------------------------------//

    PriceValues_SHM [threadIdx.y] [threadIdx.x] = PriceType(0);
    WeightValues_SHM[threadIdx.y] [threadIdx.x] = WeightType(0);



    //-- Calculate weight and price in parallel
    for (int IntBlockIdx = 0; IntBlockIdx < gpuEvolutionParameters.chromosomeSize; IntBlockIdx++){

                //--------------Load Data -------------//
        if (threadIdx.y == 0) {
                PriceGlobalData_SHM [GeneInBlockIdx] = GlobalData->itemPrice [IntBlockIdx * gpuEvolutionParameters.intBlockSize + GeneInBlockIdx];
                WeightGlobalData_SHM[GeneInBlockIdx] = GlobalData->itemWeight[IntBlockIdx * gpuEvolutionParameters.intBlockSize + GeneInBlockIdx];
        }

        ActGene = ((PopData->population[GetIndex(ChromosomeIdx, IntBlockIdx)]) >> GeneInBlockIdx) & Gene(1);

        __syncthreads();

        //-- Calculate Price and Weight --//

        PriceValues_SHM [threadIdx.y] [GeneInBlockIdx] += ActGene * PriceGlobalData_SHM  [GeneInBlockIdx];
        WeightValues_SHM[threadIdx.y] [GeneInBlockIdx] += ActGene * WeightGlobalData_SHM [GeneInBlockIdx];
    }

     //------------------------------------------------------//
     //--    PER WARP computing - NO BARRIRER NECSSARY     --//
     //------------------------------------------------------//

    //__syncthreads();
    HalfWarpReducePrice (PriceValues_SHM [threadIdx.y], threadIdx.x);
    HalfWarpReduceWeight(WeightValues_SHM[threadIdx.y], threadIdx.x);

    //__syncthreads();

    //------------------------------------------------------//
    //--    PER WARP computing - NO BARRIRER NECSSARY     --//
    //------------------------------------------------------//

    // threadIdx.x ==0 calculate final Fitness --//
    if (threadIdx.x == 0){

        Fitness result = Fitness(PriceValues_SHM [threadIdx.y][0]);


        if (WeightValues_SHM[threadIdx.y][0] > GlobalData->knapsackCapacity){
            Fitness Penalty = (WeightValues_SHM[threadIdx.y][0] - GlobalData->knapsackCapacity);

            result = result  - GlobalData->maxPriceWightRatio * Penalty;
            if (result < 0 ) result = Fitness(0);
            //result = TFitness(0);
        }

        PopData->fitness[ChromosomeIdx] = result;


   } // if


}// end of CalculateKnapsackFintess
//-----------------------------------------------------------------------------


/*
 * Find the best solution
 *
 * @param threadIdx1D
 * @param ParentsData
 *
 */
__device__ int FindTheBestLocation(int threadIdx1D, PopulationData * ParentsData){


  __shared__ Fitness shared_Max    [BLOCK_SIZE];
  __shared__ int      shared_Max_Idx[BLOCK_SIZE];


    //-- Clear shared buffer --//
  shared_Max    [threadIdx1D] = Fitness(0);
  shared_Max_Idx[threadIdx1D] = 0;


  Fitness FitnessValue;

  int i = threadIdx1D;


  //-- Reduction to shared memory --//
  while (i < gpuEvolutionParameters.populationSize){

      FitnessValue = ParentsData->fitness[i];
      if (FitnessValue > shared_Max[threadIdx1D]){
          shared_Max    [threadIdx1D] = FitnessValue;
          shared_Max_Idx[threadIdx1D] = i;
      }

      i += BLOCK_SIZE;
  }


  __syncthreads();

  //-- Reduction in shared memory --//
  for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2){
	if (threadIdx1D < stride) {
            if (shared_Max[threadIdx1D] < shared_Max[threadIdx1D + stride]){
               shared_Max    [threadIdx1D] = shared_Max    [threadIdx1D + stride];
               shared_Max_Idx[threadIdx1D] = shared_Max_Idx[threadIdx1D + stride];
            }
        }
	__syncthreads();
  }

  __syncthreads();


  return shared_Max_Idx[0];


}// end of FindTheBestLocation
//------------------------------------------------------------------------------


/*
 * Select select emigrants and place them into new population
 * @param ParentsData      - Parent population
 * @return EmigrantsToSend - Buffer for emmigrants
 * @param RandomSeed
 *
 */
__global__ void SelectEmigrantsKernel(PopulationData * ParentsData, PopulationData * EmigrantsToSend,
                                      unsigned int RandomSeed){

    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;


    RNG_2x32::ctr_type RandomValues;
    __shared__ int     EmigrantIdx_SHM[CHR_PER_BLOCK];


    if (ChromosomeIdx >= gpuEvolutionParameters.emigrantCount) return;



    //------------------------ selection -------------------------------------//
    if ((threadIdx.y == 0) && (threadIdx.x < CHR_PER_BLOCK)){

       TwoRandomINTs(&RandomValues, ChromosomeIdx, RandomSeed);
       EmigrantIdx_SHM[threadIdx.x] = Selection(ParentsData, RandomValues.v[0], RandomValues.v[1]);

    }


    //--------------- reduction for finding the best solution in BLOCK 0 ---------------//
    int LocalBestIdx = 0;
    if (blockIdx.y == 0) {
        LocalBestIdx = FindTheBestLocation(threadIdx.y * blockDim.x + threadIdx.x , ParentsData);
    }



    //-- first emigrant is the best solution --//
    if ((ChromosomeIdx == 0) && (threadIdx.x == 0)) {
        EmigrantIdx_SHM[0] =  LocalBestIdx;
    }
    __syncthreads();


    //------------------------ data copy -------------------------------------//

    while (GeneIdx < gpuEvolutionParameters.chromosomeSize){
        EmigrantsToSend->population[GetIndex(ChromosomeIdx, GeneIdx)] = ParentsData->population[GetIndex(EmigrantIdx_SHM[threadIdx.y], GeneIdx)];
        GeneIdx +=  WARP_SIZE;
    }

    if (threadIdx.x == 0) EmigrantsToSend->fitness[ChromosomeIdx] = ParentsData->fitness[EmigrantIdx_SHM[threadIdx.y]];


} // end of SelectEmigrantsKernel
//------------------------------------------------------------------------------


/*
 * Accept emigrants (give them into population)
 *
 * @param ParentsData
 * @param EmigrantsToReceive
 * @param VectorLock
 * @param RandomSeed
 *
 */
__global__ void AcceptEmigrantsKernel(PopulationData * ParentsData, PopulationData *  EmigrantsToReceive,
                                      unsigned int RandomSeed ){

    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;


    RNG_2x32::ctr_type RandomValues;
    __shared__ int     ParrentToReplaceIdx_SHM[CHR_PER_BLOCK];
    volatile   int*    ParentToRelaceIdx = ParrentToReplaceIdx_SHM;

    if (ChromosomeIdx >= gpuEvolutionParameters.emigrantCount) return;


    //------------------------ selection -------------------------------------//

    if (threadIdx.x == 0) {
       TwoRandomINTs(&RandomValues, ChromosomeIdx, RandomSeed);
       ParentToRelaceIdx[threadIdx.y] = Selection_Worse(ParentsData, RandomValues.v[0], RandomValues.v[1]);

       vectorSemaphore[ParentToRelaceIdx[threadIdx.y] % kMaxVectorSemaphores].acquire();
    }



    //------------------------------------------------------//
    //--    PER WARP computing - NO BARRIRER NECSSARY     --//
    //------------------------------------------------------//

    //------- replacement --------//
    if (ParentsData->fitness[ParentToRelaceIdx[threadIdx.y]] < EmigrantsToReceive->fitness[ChromosomeIdx]){

        //-- copy data --//
        while (GeneIdx < gpuEvolutionParameters.chromosomeSize){
            ParentsData->population[GetIndex(ParentToRelaceIdx[threadIdx.y], GeneIdx)] = EmigrantsToReceive->population[GetIndex(ChromosomeIdx, GeneIdx)];
            GeneIdx +=  WARP_SIZE;
        }

        if (threadIdx.x == 0) ParentsData->fitness[ParentToRelaceIdx[threadIdx.y]] = EmigrantsToReceive->fitness[ChromosomeIdx];

    } // replacement

    //------------------------------------------------------//
    //--    PER WARP computing - NO BARRIRER NECSSARY     --//
    //------------------------------------------------------//

    // only first thread unlocks the lock
    if (threadIdx.x == 0) {
        vectorSemaphore[ParentToRelaceIdx[threadIdx.y] % kMaxVectorSemaphores].release();
    }


}// end of AcceptEmigrantsKernel
//------------------------------------------------------------------------------
