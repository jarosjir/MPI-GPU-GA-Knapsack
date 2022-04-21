/**
 * @file        Statistics.cu
 * @author      Jiri Jaros
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
 * @brief       Implementation file of the GA statistics
 *              This class maintains and collects GA statistics
 *
 * @date        08 June 2012 2012, 00:00 (created)
 *              21 April     2022, 10:18 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#include <mpi.h>
#include <helper_cuda.h>
#include <malloc.h>

#include "Statistics.h"
#include "CUDAKernels.h"


//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Constructor of the class.
 */
Statistics::Statistics()
{
  const Parameters& Params = Parameters::getInstance();

  mGlobalDerivedStat       = nullptr;
  mReceiveStatDataBuffer   = nullptr;
  mReceiveIndividualBuffer = nullptr;

  // for MPI collection function
  if (Params.getIslandIdx() == 0)
  {
    mGlobalDerivedStat       = (DerivedStats *)   memalign(64, sizeof(DerivedStats));
    mReceiveStatDataBuffer   = (StatisticsData *) memalign(64, sizeof(StatisticsData) * Params.getIslandCount());
    mReceiveIndividualBuffer = (Gene *)           memalign(64, sizeof(Gene)
                                                                * Params.getChromosomeSize() * Params.getIslandCount());
  }

  // Allocate CUDA memory
  allocateCudaMemory();
}// end of Statistics
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class
 */
Statistics::~Statistics()
{
  //  Free host memory
  if (mGlobalDerivedStat)
  {
    free(mGlobalDerivedStat);
    mGlobalDerivedStat = nullptr;
  }

  if (mReceiveStatDataBuffer)
  {
    free(mReceiveStatDataBuffer);
    mReceiveStatDataBuffer = nullptr;
  }

  if (mReceiveIndividualBuffer)
  {
    free(mReceiveIndividualBuffer);
    mReceiveIndividualBuffer = nullptr;
  }

  freeCudaMemory();
}// end of Statistics
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print best individual as a string.
 */
std::string Statistics::getBestIndividualStr(KnapsackData* globalKnapsackData) const
{
  /// Lambda function to convert 1 int into a bit string
  auto convertIntToBitString= [] (Gene value, int nValidDigits) -> std::string
  {
    std::string str = "";

    for (int bit = 0; bit < nValidDigits; bit++)
    {
      str += ((value & (1 << bit)) == 0) ? "0" : "1";
      str += (bit % 8 == 7) ? " " : "";
    }

    for (int bit = nValidDigits; bit < 32; bit++)
    {
      str += 'x';
      str += (bit % 8 == 7) ? " " : "";
    }

    return str;
  };// end of convertIntToBitString

  std::string bestChromozome = "";

  const int nBlocks = globalKnapsackData->originalNumberOfItems / 32;

  for (int blockId = 0; blockId < nBlocks; blockId++)
  {
    bestChromozome += convertIntToBitString(mLocalBestIndividual[blockId], 32) + "\n";
  }

  // Reminder
  if (globalKnapsackData->originalNumberOfItems % 32 > 0 )
  {
    bestChromozome += convertIntToBitString(mLocalBestIndividual[nBlocks],
                                            globalKnapsackData->originalNumberOfItems % 32);
  }

 return bestChromozome;
}// end of getBestIndividualStr
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate global statistics.
 */
void Statistics::calculate(GPUPopulation* population,
                           bool           printBest)
{
  const Parameters& params = Parameters::getInstance();

  // Calculate local statistics
  calculateLocalStats(population, printBest);

  // Collect statistics data
  MPI_Gather(mHostStatData         , sizeof(StatisticsData), MPI_BYTE,
             mReceiveStatDataBuffer, sizeof(StatisticsData), MPI_BYTE, 0, MPI_COMM_WORLD);


  if (printBest)
  {
    // Collect individuals
    MPI_Gather(mLocalBestIndividual    , params.getChromosomeSize(), MPI_UNSIGNED,
               mReceiveIndividualBuffer, params.getChromosomeSize(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  }

  // only master calculates the global statistics
  if (params.getIslandIdx() == 0)
  {
    calculateGlobalStatistics(printBest);
  }
}// end of calculate
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate GPU memory.
 */
void Statistics::allocateCudaMemory()
{
  // Allocate Host basic structure
  checkCudaErrors(cudaHostAlloc<StatisticsData>(&mHostStatData,  sizeof(StatisticsData), cudaHostAllocDefault));

  // Allocate Host basic structure
  checkCudaErrors(cudaHostAlloc<Gene>(&mLocalBestIndividual,
                                      sizeof(Gene) * Parameters::getInstance().getChromosomeSize(),
                                      cudaHostAllocDefault));

  // Device data
  checkCudaErrors(cudaMalloc<StatisticsData>(&mLocalDeviceStatData,  sizeof(StatisticsData)));
}// end of allocateCudaMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free GPU memory.
 */
void Statistics::freeCudaMemory()
{
  // Free CPU Best individual
  checkCudaErrors(cudaFreeHost(mLocalBestIndividual));

  //  Free structure
  checkCudaErrors(cudaFreeHost(mHostStatData));

  // Free whole structure
  checkCudaErrors(cudaFree(mLocalDeviceStatData));
}// end of freeCudaMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize statistics before computation.
 */
void Statistics::initStatistics()
{
  mHostStatData->maxFitness  = Fitness(0);
  mHostStatData->minFitness  = Fitness(UINT_MAX);
  mHostStatData->sumFitness  = 0.0f;
  mHostStatData->sum2Fitness = 0.0f;
  mHostStatData->indexBest   = 0;

  // Copy 4 statistics values
  checkCudaErrors(cudaMemcpy(mLocalDeviceStatData, mHostStatData, sizeof(StatisticsData), cudaMemcpyHostToDevice));
}// end of initStatistics
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate local statistics and download them to the host.
 */
void Statistics::calculateLocalStats(GPUPopulation* population,
                                     bool           printBest)
{
  // Initialize statistics
  initStatistics();

  // Run the CUDA kernel to calculate statistics
  cudaCalculateStatistics<<<Parameters::getInstance().getNumberOfDeviceSMs() * 2, BLOCK_SIZE >>>
                         (mLocalDeviceStatData, population->getDeviceData());


  // Copy data down to host
  copyFromDevice(population, printBest);
}// end of calculateLocalStats
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from GPU Statistics structure to CPU.
 */
void Statistics::copyFromDevice(GPUPopulation* population,
                                bool           printBest)
{
  // Copy 4 statistics values
  checkCudaErrors(cudaMemcpy(mHostStatData, mLocalDeviceStatData, sizeof(StatisticsData), cudaMemcpyDeviceToHost));

  //  Copy of chromosome
  if (printBest)
  {
    population->copyIndividualFromDevice(mLocalBestIndividual, mHostStatData->indexBest);
  }
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Summarize statistics Local statistics to global.
 */
void Statistics::calculateGlobalStatistics(bool printBest)
{

  mGlobalDerivedStat->bestIslandIdx = 0;

  mHostStatData->maxFitness  = mReceiveStatDataBuffer[0].maxFitness;
  mHostStatData->minFitness  = mReceiveStatDataBuffer[0].minFitness;
  mHostStatData->sumFitness  = mReceiveStatDataBuffer[0].sumFitness;
  mHostStatData->sum2Fitness = mReceiveStatDataBuffer[0].sum2Fitness;

  const Parameters& params = Parameters::getInstance();

  // Numeric statistics
  for (int i = 1; i < params.getIslandCount(); i++)
  {
    if (mHostStatData->maxFitness < mReceiveStatDataBuffer[i].maxFitness)
    {
      mHostStatData->maxFitness = mReceiveStatDataBuffer[i].maxFitness;
      mGlobalDerivedStat->bestIslandIdx = i;
    }

    if (mHostStatData->minFitness > mReceiveStatDataBuffer[i].minFitness)
    {
      mHostStatData->minFitness = mReceiveStatDataBuffer[i].minFitness;
    }

    mHostStatData->sumFitness  += mReceiveStatDataBuffer[i].sumFitness;
    mHostStatData->sum2Fitness += mReceiveStatDataBuffer[i].sum2Fitness;
  }


 mGlobalDerivedStat->avgFitness = mHostStatData->sumFitness / (params.getPopulationSize() * params.getIslandCount());
 mGlobalDerivedStat->divergence = sqrt(fabs((mHostStatData->sum2Fitness /
                                              (params.getPopulationSize() * params.getIslandCount()) -
                                             mGlobalDerivedStat->avgFitness * mGlobalDerivedStat->avgFitness))
                                  );
}// end of calculateGlobalStatistics
//----------------------------------------------------------------------------------------------------------------------
