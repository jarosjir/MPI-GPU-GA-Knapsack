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

__constant__  TEvolutionParameters GPU_EvolutionParameters;

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
inline __device__ int  Selection(TPopulationData * ParentsData, unsigned int Random1, unsigned int Random2);
// Select an individual pick the worse (for replacement)
inline __device__ int  Selection_Worse(TPopulationData * ParentsData, unsigned int Random1, unsigned int Random2);

// Perform Uniform Crossover
inline __device__ void CrossoverUniformFlip(TGene& GeneOffspring1, TGene& GeneOffspring2,
                                            TGene GeneParent1    , TGene GeneParent2,
                                            unsigned int RandomValue);

// Perform BitFlip mutation
inline __device__ void MutationBitFlip(TGene& GeneOffspring1, TGene& GeneOffspring2,
                                       unsigned int RandomValue1,unsigned int RandomValue2, int BitID);

// Reduction kernels
__device__ void HalfWarpReducePrice (volatile TPriceType * sdata, int tid);
__device__ void HalfWarpReduceWeight(volatile TWeightType* sdata, int tid);
__device__ void HalfWarpReduceGene(volatile TGene* sdata, int tid);


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

    return (ChromosomeIdx * GPU_EvolutionParameters.ChromosomeSize + GeneIdx);

}// end of GetIndex
//------------------------------------------------------------------------------

/*
 * Half warp reduce for price
 *
 * @param sdata  - data to reduce
 * @param tid    - idx of thread
 *
 */
__device__ void HalfWarpReducePrice(volatile TPriceType * sdata, int tid){
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
__device__ void HalfWarpReduceWeight(volatile TWeightType* sdata, int tid){
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
__device__ void HalfWarpReduceGene(volatile TGene* sdata, int tid){
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
inline __device__ int Selection(TPopulationData * ParentsData, unsigned int Random1, unsigned int Random2){


    unsigned int Idx1 = Random1 % (ParentsData->PopulationSize);
    unsigned int Idx2 = Random2 % (ParentsData->PopulationSize);

    return (ParentsData->Fitness[Idx1] > ParentsData->Fitness[Idx2]) ? Idx1 : Idx2;
}// Selection
//------------------------------------------------------------------------------


/*
 * Select one individual (the worse one)
 * @param ParentData - Parent population
 * @param Random1    - First random value
 * @param Random2    - Second random value
 * @return           - idx of the selected individual
 */
inline __device__ int Selection_Worse(TPopulationData * ParentsData, unsigned int Random1, unsigned int Random2){


    unsigned int Idx1 = Random1 % (ParentsData->PopulationSize);
    unsigned int Idx2 = Random2 % (ParentsData->PopulationSize);

    return (ParentsData->Fitness[Idx1] < ParentsData->Fitness[Idx2]) ? Idx1 : Idx2;
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
inline __device__ void CrossoverUniformFlip(TGene& GeneOffspring1, TGene& GeneOffspring2,
                                            TGene GeneParent1    , TGene GeneParent2,
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
inline __device__ void MutationBitFlip(TGene& GeneOffspring1, TGene& GeneOffspring2,
                                      unsigned int RandomValue1,unsigned int RandomValue2, int BitID){

  //GeneOffspring1 ^= ((unsigned int)(RandomValue1 < GPU_EvolutionParameters.MutationUINTBoundary) << BitID);
  //GeneOffspring2 ^= ((unsigned int)(RandomValue2 < GPU_EvolutionParameters.MutationUINTBoundary) << BitID);
  if (RandomValue1 < GPU_EvolutionParameters.MutationUINTBoundary) GeneOffspring1 ^= (1 << BitID);
  if (RandomValue2 < GPU_EvolutionParameters.MutationUINTBoundary) GeneOffspring2 ^= (1 << BitID);



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
__global__ void FirstPopulationGenerationKernel(TPopulationData * PopData, unsigned int RandomSeed){

   size_t i      = threadIdx.x + blockIdx.x*blockDim.x;
   size_t stride = blockDim.x * gridDim.x;

   RNG_2x32::ctr_type RandomValues;

   const int PopulationDIM = PopData->ChromosomeSize * PopData->PopulationSize;

   while (i < PopulationDIM) {

       TwoRandomINTs(&RandomValues, i, RandomSeed);
       PopData->Population[i] = RandomValues.v[0];

       i += stride;
       if (i < PopulationDIM) {
          PopData->Population[i] = RandomValues.v[1];
       }
       i += stride;
    }

   i  = threadIdx.x + blockIdx.x*blockDim.x;
   while (i < PopData->PopulationSize){
        PopData->Fitness[i]    = 0.0f;
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
__global__ void GeneticManipulationKernel(TPopulationData * ParentsData, TPopulationData * OffspringData,
                                    unsigned int RandomSeed){
    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = 2* (threadIdx.y + blockIdx.y * blockDim.y);

    //-- Init Random --//
    RNG_4x32  rng_4x32;
    RNG_4x32::key_type key    ={{GeneIdx, ChromosomeIdx}};
    RNG_4x32::ctr_type counter={{0, 0, RandomSeed ,0xbeeff00d}};
    RNG_4x32::ctr_type RandomValues;




    if (ChromosomeIdx >= GPU_EvolutionParameters.OffspringPopulationSize) return;

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
        CrossoverFlag[threadIdx.x] = RandomValues.v[0] < GPU_EvolutionParameters.CrossoverUINTBoundary;
    }


    __syncthreads(); // to distribute the selected chromosomes
    //------------------------------------------------------------------------//
    //------------------------ Manipulation  ---------------------------------//
    //------------------------------------------------------------------------//

    //-- Go through two chromosomes and do uniform crossover and mutation--//
    while (GeneIdx < GPU_EvolutionParameters.ChromosomeSize){
        TGene GeneParent1 = ParentsData->Population[GetIndex(Parent1_Idx[threadIdx.y], GeneIdx)];
        TGene GeneParent2 = ParentsData->Population[GetIndex(Parent2_Idx[threadIdx.y], GeneIdx)];

        TGene GeneOffspring1 = 0;
        TGene GeneOffspring2 = 0;

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
        for (int BitID = 0; BitID < GPU_EvolutionParameters.IntBlockSize; BitID+=2){

            counter.incr();
            RandomValues = rng_4x32(counter, key);

            MutationBitFlip(GeneOffspring1, GeneOffspring2, RandomValues.v[0],RandomValues.v[1], BitID);
            MutationBitFlip(GeneOffspring1, GeneOffspring2, RandomValues.v[2],RandomValues.v[3], BitID+1);


         }// for

        OffspringData->Population[GetIndex(ChromosomeIdx  , GeneIdx)] = GeneOffspring1;
        OffspringData->Population[GetIndex(ChromosomeIdx+1, GeneIdx)] = GeneOffspring2;

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
__global__ void ReplacementKernel(TPopulationData * ParentsData, TPopulationData * OffspringData, unsigned int RandomSeed){



    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;

    //-- Init Random --//
    RNG_2x32::ctr_type RandomValues;
    __shared__ unsigned int OffspringIdx_SHM[CHR_PER_BLOCK];

    if (ChromosomeIdx >= GPU_EvolutionParameters.PopulationSize) return;


    //-- select offspring --//
    if (threadIdx.x == 0){
       TwoRandomINTs(&RandomValues, ChromosomeIdx, RandomSeed);
       OffspringIdx_SHM[threadIdx.y]  = RandomValues.v[0] % (GPU_EvolutionParameters.OffspringPopulationSize);

    }

    __syncthreads();


    //------- replacement --------//
    if (ParentsData->Fitness[ChromosomeIdx] < OffspringData->Fitness[OffspringIdx_SHM[threadIdx.y]]){

        //-- copy data --//
        while (GeneIdx < GPU_EvolutionParameters.ChromosomeSize){
            ParentsData->Population[GetIndex(ChromosomeIdx, GeneIdx)] = OffspringData->Population[GetIndex(OffspringIdx_SHM[threadIdx.y], GeneIdx)];
            GeneIdx +=  WARP_SIZE;
        }

        if (threadIdx.x == 0) ParentsData->Fitness[ChromosomeIdx] = OffspringData->Fitness[OffspringIdx_SHM[threadIdx.y]];

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
__global__ void CalculateStatistics(TStatDataToExchange * StatisticsData, TPopulationData * PopData){

  int i      = threadIdx.x + blockDim.x*blockIdx.x;
  int stride = blockDim.x*gridDim.x;

  __shared__ TFitness shared_Max    [BLOCK_SIZE];
  __shared__ int      shared_Max_Idx[BLOCK_SIZE];
  __shared__ TFitness shared_Min    [BLOCK_SIZE];

  __shared__ float shared_Sum    [BLOCK_SIZE];
  __shared__ float shared_Sum2   [BLOCK_SIZE];


    //-- Clear shared buffer --//

  shared_Max    [threadIdx.x] = TFitness(0);
  shared_Max_Idx[threadIdx.x] = 0;
  shared_Min    [threadIdx.x] = TFitness(UINT_MAX);

  shared_Sum    [threadIdx.x] = 0.0f;;
  shared_Sum2   [threadIdx.x] = 0.0f;;

  __syncthreads();

  TFitness FitnessValue;

  //-- Reduction to shared memory --//
  while (i < GPU_EvolutionParameters.PopulationSize){

      FitnessValue = PopData->Fitness[i];
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

    if (StatisticsData->MaxFitness < shared_Max[threadIdx.x])
    {
      StatisticsData->MaxFitness = shared_Max[threadIdx.x];
      StatisticsData->IndexBest  = shared_Max_Idx[threadIdx.x];
    }

    if (StatisticsData->MinFitness > shared_Min[threadIdx.x])
    {
      StatisticsData->MinFitness = shared_Min[threadIdx.x];
    }

    semaphore.release();
    atomicAdd(&(StatisticsData->SumFitness),  shared_Sum [threadIdx.x]);
    atomicAdd(&(StatisticsData->Sum2Fitness), shared_Sum2[threadIdx.x]);
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
__global__ void CalculateKnapsackFintess(TPopulationData * PopData, TKnapsackData * GlobalData){


    __shared__ TPriceType  PriceGlobalData_SHM [WARP_SIZE];
    __shared__ TWeightType WeightGlobalData_SHM[WARP_SIZE];


    __shared__ TPriceType  PriceValues_SHM [CHR_PER_BLOCK] [WARP_SIZE];
    __shared__ TWeightType WeightValues_SHM[CHR_PER_BLOCK] [WARP_SIZE];


    int GeneInBlockIdx = threadIdx.x;
    int ChromosomeIdx  = threadIdx.y + blockIdx.y * blockDim.y;

    if (ChromosomeIdx >= PopData->PopulationSize) return;

    TGene ActGene;

    //------------------------------------------------------//

    PriceValues_SHM [threadIdx.y] [threadIdx.x] = TPriceType(0);
    WeightValues_SHM[threadIdx.y] [threadIdx.x] = TWeightType(0);



    //-- Calculate weight and price in parallel
    for (int IntBlockIdx = 0; IntBlockIdx < GPU_EvolutionParameters.ChromosomeSize; IntBlockIdx++){

                //--------------Load Data -------------//
        if (threadIdx.y == 0) {
                PriceGlobalData_SHM [GeneInBlockIdx] = GlobalData->ItemPrice [IntBlockIdx * GPU_EvolutionParameters.IntBlockSize + GeneInBlockIdx];
                WeightGlobalData_SHM[GeneInBlockIdx] = GlobalData->ItemWeight[IntBlockIdx * GPU_EvolutionParameters.IntBlockSize + GeneInBlockIdx];
        }

        ActGene = ((PopData->Population[GetIndex(ChromosomeIdx, IntBlockIdx)]) >> GeneInBlockIdx) & TGene(1);

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

        TFitness result = TFitness(PriceValues_SHM [threadIdx.y][0]);


        if (WeightValues_SHM[threadIdx.y][0] > GlobalData->KnapsackCapacity){
            TFitness Penalty = (WeightValues_SHM[threadIdx.y][0] - GlobalData->KnapsackCapacity);

            result = result  - GlobalData->MaxPriceWightRatio * Penalty;
            if (result < 0 ) result = TFitness(0);
            //result = TFitness(0);
        }

        PopData->Fitness[ChromosomeIdx] = result;


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
__device__ int FindTheBestLocation(int threadIdx1D, TPopulationData * ParentsData){


  __shared__ TFitness shared_Max    [BLOCK_SIZE];
  __shared__ int      shared_Max_Idx[BLOCK_SIZE];


    //-- Clear shared buffer --//
  shared_Max    [threadIdx1D] = TFitness(0);
  shared_Max_Idx[threadIdx1D] = 0;


  TFitness FitnessValue;

  int i = threadIdx1D;


  //-- Reduction to shared memory --//
  while (i < GPU_EvolutionParameters.PopulationSize){

      FitnessValue = ParentsData->Fitness[i];
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
__global__ void SelectEmigrantsKernel(TPopulationData * ParentsData, TPopulationData * EmigrantsToSend,
                                      unsigned int RandomSeed){

    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;


    RNG_2x32::ctr_type RandomValues;
    __shared__ int     EmigrantIdx_SHM[CHR_PER_BLOCK];


    if (ChromosomeIdx >= GPU_EvolutionParameters.EmigrantCount) return;



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

    while (GeneIdx < GPU_EvolutionParameters.ChromosomeSize){
        EmigrantsToSend->Population[GetIndex(ChromosomeIdx, GeneIdx)] = ParentsData->Population[GetIndex(EmigrantIdx_SHM[threadIdx.y], GeneIdx)];
        GeneIdx +=  WARP_SIZE;
    }

    if (threadIdx.x == 0) EmigrantsToSend->Fitness[ChromosomeIdx] = ParentsData->Fitness[EmigrantIdx_SHM[threadIdx.y]];


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
__global__ void AcceptEmigrantsKernel(TPopulationData * ParentsData, TPopulationData *  EmigrantsToReceive,
                                      unsigned int RandomSeed ){

    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;


    RNG_2x32::ctr_type RandomValues;
    __shared__ int     ParrentToReplaceIdx_SHM[CHR_PER_BLOCK];
    volatile   int*    ParentToRelaceIdx = ParrentToReplaceIdx_SHM;

    if (ChromosomeIdx >= GPU_EvolutionParameters.EmigrantCount) return;


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
    if (ParentsData->Fitness[ParentToRelaceIdx[threadIdx.y]] < EmigrantsToReceive->Fitness[ChromosomeIdx]){

        //-- copy data --//
        while (GeneIdx < GPU_EvolutionParameters.ChromosomeSize){
            ParentsData->Population[GetIndex(ParentToRelaceIdx[threadIdx.y], GeneIdx)] = EmigrantsToReceive->Population[GetIndex(ChromosomeIdx, GeneIdx)];
            GeneIdx +=  WARP_SIZE;
        }

        if (threadIdx.x == 0) ParentsData->Fitness[ParentToRelaceIdx[threadIdx.y]] = EmigrantsToReceive->Fitness[ChromosomeIdx];

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
