/**
 * @file        Evolution.cu
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
 * @brief       Implementation file of the GA evolution
 *              This class controls the evolution process on multiple GPUs across many nodes.
 *
 * @date        08 June      2012, 00:00 (created)
 *              11 April     2022, 21:02 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#include <mpi.h>
#include <sys/time.h>

#include "Evolution.h"
#include "Statistics.h"
#include "CUDAKernels.h"
#include "Parameters.h"

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class
 */
Evolution::Evolution(int argc, char **argv)
  : mParams(Parameters::getInstance()),
    mActGeneration(0),
    mMasterPopulation(nullptr),
    mOffspringPopulation(nullptr),
    mDeviceEmigrantsToSend(nullptr),
    mDeviceEmigrantsToReceive(nullptr),
    mHostEmigrantsToSend(nullptr),
    mHostEmigrantsToReceive(nullptr),
    mGlobalData()

{
  // Parse command line
  mParams.parseCommandline(argc,argv);

  // Load data from disk
  mGlobalData.loadFromFile();

  // Create populations on GPU
  mMasterPopulation         = new GPUPopulation(mParams.getPopulationSize(), mParams.getChromosomeSize());
  mOffspringPopulation      = new GPUPopulation(mParams.getOffspringPopulationSize(), mParams.getChromosomeSize());

  // Create buffers on GPU
  mDeviceEmigrantsToSend    = new GPUPopulation(mParams.getEmigrantCount(), mParams.getChromosomeSize());
  mDeviceEmigrantsToReceive = new GPUPopulation(mParams.getEmigrantCount(), mParams.getChromosomeSize());

  // Create buffers on CPU
  mHostEmigrantsToSend      = new CPUPopulation(mParams.getEmigrantCount(), mParams.getChromosomeSize());
  mHostEmigrantsToReceive   = new CPUPopulation(mParams.getEmigrantCount(), mParams.getChromosomeSize());

  // Create statistics
  mStatistics               = new Statistics();

  initRandomSeed();
}// end of Evolution
//----------------------------------------------------------------------------------------------------------------------


/**
 * Destructor of the class
 */
Evolution::~Evolution()
{
  delete mMasterPopulation;
  delete mOffspringPopulation;

  delete mDeviceEmigrantsToSend;
  delete mDeviceEmigrantsToReceive;

  delete mHostEmigrantsToSend;
  delete mHostEmigrantsToReceive;

  delete mStatistics;
}// end of Destructor
//----------------------------------------------------------------------------------------------------------------------

/**
 * Run Evolution
 */
void Evolution::run()
{
  initialize();

  runEvolutionCycle();
}// end of run
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Initialize seed
 */
void Evolution::initRandomSeed()
{
  struct timeval tp1;

  gettimeofday(&tp1, NULL);

  mRandomSeed = (tp1.tv_sec / (mParams.getIslandIdx() + 1)) * tp1.tv_usec;
};// end of initRandomSeed
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialization
 */
void Evolution::initialize()
{
  mActGeneration = 0;

  // Store parameters on GPU and print them out
  mParams.copyToDevice();
  mParams.printOutAllParameters();


  // Initialize population
  cudaGenerateFirstPopulation<<<mParams.getNumberOfDeviceSMs() * 2, BLOCK_SIZE>>>
                             (mMasterPopulation->getDeviceData(),
                              getRandomSeed());

  dim3 nBlocks;

  nBlocks.x = 1;
  nBlocks.y = (mParams.getPopulationSize() / (CHR_PER_BLOCK) +1);
  nBlocks.z = 1;

  dim3 nThreads;
  nThreads.x = WARP_SIZE;
  nThreads.y = CHR_PER_BLOCK;
  nThreads.z = 1;


 cudaCalculateKnapsackFintess<<<nBlocks, nThreads>>>
                             (mMasterPopulation->getDeviceData(),
                             mGlobalData.getDeviceData());
}// end of initialize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Run evolutionary cycle for defined number of generations.
 */
void Evolution::runEvolutionCycle()
{
  dim3 nBlocks;
  dim3 nThreads;

  nThreads.x = WARP_SIZE;
  nThreads.y = CHR_PER_BLOCK;
  nThreads.z = 1;

  // Evaluate generations
  for (mActGeneration = 1; mActGeneration < mParams.getNumOfGenerations(); mActGeneration++)
  {
    // Migration
    if (mActGeneration % mParams.getMigrationInterval() == 0)
    {
      migrate();
    }


    //  Selection
    nBlocks.x = 1;
    nBlocks.y = (mParams.getOffspringPopulationSize() % (CHR_PER_BLOCK << 1)  == 0) ?
                    mParams.getOffspringPopulationSize() / (CHR_PER_BLOCK << 1)  :
                    mParams.getOffspringPopulationSize() / (CHR_PER_BLOCK << 1) + 1;
    nBlocks.z = 1;

    cudaGeneticManipulation<<<nBlocks, nThreads>>>
                           (mMasterPopulation->getDeviceData(),
                            mOffspringPopulation->getDeviceData(),
                            getRandomSeed());

    // Evaluation

    nBlocks.x = 1;
    nBlocks.y = (mParams.getOffspringPopulationSize() % (CHR_PER_BLOCK)  == 0) ?
                    mParams.getOffspringPopulationSize() / (CHR_PER_BLOCK)  :
                    mParams.getOffspringPopulationSize() / (CHR_PER_BLOCK) + 1;
    nBlocks.z = 1;

    cudaCalculateKnapsackFintess<<<nBlocks, nThreads>>>
                                (mOffspringPopulation->getDeviceData(),
                                 mGlobalData.getDeviceData());


    // Replacement
    nBlocks.x = 1;
    nBlocks.y = (mParams.getPopulationSize() % (CHR_PER_BLOCK)  == 0) ?
                     mParams.getPopulationSize() / (CHR_PER_BLOCK)  :
                     mParams.getPopulationSize() / (CHR_PER_BLOCK) + 1;
    nBlocks.z = 1;

    cudaReplacement<<<nBlocks, nThreads>>>
                   (mMasterPopulation->getDeviceData(),
                    mOffspringPopulation->getDeviceData(),
                    getRandomSeed());


    // Statistics
    if (mActGeneration % mParams.getStatisticsInterval() == 0)
    {
      mStatistics->calculate(mMasterPopulation, mParams.getPrintBest());

      if (isMaster())
      {
        printf("Generation %6d, BestIsland %d, MaxFitness %6f, MinFitness %6f, AvgFitness %6f, Diver %6f \n",
                  mActGeneration, mStatistics->getBestIslandIdx(),
                  mStatistics->getMaxFitness(), mStatistics->getMinFitness(),
                  mStatistics->getAvgFitness(), mStatistics->getDivergence());


        if (mParams.getPrintBest())
        {
          printf("%s\n", mStatistics->getBestIndividualStr(mGlobalData.getHostData()).c_str());
        }
      }// isMaster
    } // print stat
    /// Check error per generation
    checkAndReportCudaError(__FILE__,__LINE__);
  }// generations



  // Final statistics
  mStatistics->calculate(mMasterPopulation, true);
  if (isMaster())
  {
    printf("--------------------------------------------------------------------------------------------------\n");
    printf("FinalBestIsland %d, FinalMaxFitness %6f, FinalMinFitness %6f, FinalAvgFitness %6f, FinalDiver %6f \n",
                mStatistics->getBestIslandIdx(),
                mStatistics->getMaxFitness(), mStatistics->getMinFitness(),
                mStatistics->getAvgFitness(), mStatistics->getDivergence());
  }// is master
}// end of runEvolutionCycle
//----------------------------------------------------------------------------------------------------------------------

/**
 * Migrate individuals between GPUs.
 */
void Evolution::migrate()
{
  dim3 nBlocks;
  dim3 nThreads;

  // Communication consists of 4 messages
  constexpr int nMessages = 4;

  MPI_Status  status [nMessages];
  MPI_Request request[nMessages];

  int mpiTarget;
  int mpiSource;

  // Send to the right
  mpiTarget = (mParams.getIslandIdx() + 1) % mParams.getIslandCount();

  // Send to the left
  mpiSource = (mParams.getIslandIdx() == 0) ? mpiSource = mParams.getIslandCount() - 1
                                            : mpiSource = mParams.getIslandIdx() - 1;


  // Receive fitness values and immigrants
  MPI_Irecv(mHostEmigrantsToReceive->getHostData()->fitness,
            mParams.getEmigrantCount(),
            MPI_FLOAT,
            mpiSource,
            kMpiDataTag,
            MPI_COMM_WORLD,
            &request[2]);

  MPI_Irecv(mHostEmigrantsToReceive->getHostData()->population,
            mParams.getEmigrantCount() * mParams.getChromosomeSize(),
            MPI_UNSIGNED,
            mpiSource,
            kMpiDataTag,
            MPI_COMM_WORLD,
            &request[3]);


  nThreads.x = WARP_SIZE;
  nThreads.y = CHR_PER_BLOCK;
  nThreads.z = 1;

  nBlocks.x  = 1;
  nBlocks.y  = (mParams.getEmigrantCount() % CHR_PER_BLOCK  == 0) ?
                   mParams.getEmigrantCount() / CHR_PER_BLOCK :
                   mParams.getEmigrantCount() / CHR_PER_BLOCK  + 1;
  nBlocks.z  = 1;


  // Fill emigrant population
  cudaSelectEmigrants<<<nBlocks, nThreads>>>
                     (mMasterPopulation->getDeviceData(),
                      mDeviceEmigrantsToSend->getDeviceData(),
                      getRandomSeed());


  mDeviceEmigrantsToSend->copyFromDevice(mHostEmigrantsToSend->getHostData());

  // Send emigrants
  MPI_Isend(mHostEmigrantsToSend->getHostData()->fitness,
            mParams.getEmigrantCount(),
            MPI_FLOAT,
            mpiTarget,
            kMpiDataTag,
            MPI_COMM_WORLD,
            &request[0]);

  MPI_Isend(mHostEmigrantsToSend->getHostData()->population,
            mParams.getEmigrantCount() * mParams.getChromosomeSize(),
            MPI_UNSIGNED,
            mpiTarget,
            kMpiDataTag,
            MPI_COMM_WORLD,
            &request[1]);

  MPI_Waitall(nMessages, request, status);

  // Fill emigrant population
  mDeviceEmigrantsToReceive->copyToDevice(mHostEmigrantsToReceive->getHostData());

  cudaAcceptEmigrants<<<nBlocks, nThreads>>>
                     (mMasterPopulation->getDeviceData(),
                      mDeviceEmigrantsToReceive->getDeviceData(),
                      getRandomSeed());
}// end of migrate
//----------------------------------------------------------------------------------------------------------------------