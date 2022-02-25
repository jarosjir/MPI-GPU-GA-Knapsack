/**
 * @file        Parameters.cpp
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
 * @brief       Header file of the parameter class.
 *              This class maintains all the parameters of evolution.
 *
 * @date        30 March     2012, 00:00 (created)
 *              25 February  2022, 20:01 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */


#include <mpi.h>
#include <getopt.h>
#include <helper_cuda.h>

#include "Parameters.h"
#include "CUDAKernels.h"

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Copy of Evolutionary parameters in device constant memory.
extern __constant__  EvolutionParameters gpuEvolutionParameters;

// Singleton initialization
bool Parameters::sInstanceFlag             = false;
Parameters* Parameters::sSingletonInstance = nullptr;


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Get instance of Parameters
 */
Parameters& Parameters::getInstance()
{
  if(!sInstanceFlag)
  {
    sSingletonInstance = new Parameters();
    sInstanceFlag = true;
    return *sSingletonInstance;
  }
  else
  {
    return *sSingletonInstance;
  }
}// end of Parameters::getInstance
//----------------------------------------------------------------------------------------------------------------------

/**
 * Load parameters from command line
 */
void Parameters::parseCommandline(int    argc,
                                  char** argv)
{
  float offspringPercentage = 0.5f;
  float emigrantPercentage = 0.1f;
  char c;

  while ((c = getopt (argc, argv, "p:g:m:c:o:e:n:f:s:bh")) != -1)
  {
    switch (c)
    {
      case 'p':
      {
        if (atoi(optarg) != 0)
        {
          mEvolutionParameters.populationSize = atoi(optarg);
        }
        break;
      }
      case 'g':
      {
        if (atoi(optarg) != 0)
        {
          mEvolutionParameters.numOfGenerations = atoi(optarg);
        }
        break;
      }
      case 'm':
      {
        if (atof(optarg) != 0)
        {
          mEvolutionParameters.mutationPst = atof(optarg);
        }
        break;
      }
     case 'c':
     {
        if (atof(optarg) != 0)
        {
          mEvolutionParameters.crossoverPst = atof(optarg);
        }
        break;
      }
      case 'o':
      {
        if (atof(optarg) != 0)
        {
          offspringPercentage = atof(optarg);
        }
        break;
      }
      case 'e':
      {
        if (atof(optarg) != 0)
        {
          emigrantPercentage = atof(optarg);
        }
        break;
      }
      case 'n':
      {
        if (atoi(optarg) != 0)
        {
          mEvolutionParameters.migrationInterval = atoi(optarg);
        }
        break;
      }
      case 's':
      {
        if (atoi(optarg) != 0)
        {
          mEvolutionParameters.statisticsInterval = atoi(optarg);
        }
        break;
      }
      case 'b':
      {
        mPrintBest = true;
        break;
      }
      case 'f':
      {
        mGlobalDataFileName  = optarg;
        break;
      }
      case 'h':
      {
        printUsageAndExit();
        break;
      }
      default:
      {
        printUsageAndExit();
      }
    }
  }

  // Set population size to be even.
  if (mEvolutionParameters.populationSize % 2 == 1)
  {
    mEvolutionParameters.populationSize++;
  }

  mEvolutionParameters.offspringPopulationSize = (int) (offspringPercentage * mEvolutionParameters.populationSize);
  if (mEvolutionParameters.offspringPopulationSize == 0)
  {
    mEvolutionParameters.offspringPopulationSize = 2;
  }
  if (mEvolutionParameters.offspringPopulationSize % 2 == 1)
  {
    mEvolutionParameters.offspringPopulationSize++;
  }

  // Check emigrant count and set it at least to 1
  mEvolutionParameters.emigrantCount = (int) (emigrantPercentage * mEvolutionParameters.populationSize);
  if (mEvolutionParameters.emigrantCount == 0)
  {
    mEvolutionParameters.emigrantCount = 1;
  }
  if ((mEvolutionParameters.emigrantCount % 2) == 0)
  {
    mEvolutionParameters.emigrantCount++;
  }

  if (mEvolutionParameters.emigrantCount > mEvolutionParameters.populationSize)
  {
    mEvolutionParameters.emigrantCount = mEvolutionParameters.populationSize;
  }

  if (mEvolutionParameters.migrationInterval < 0)
  {
    mEvolutionParameters.migrationInterval = 1;
  }

  // Set UINT mutation threshold to faster comparison
  mEvolutionParameters.mutationUintBoundary  = (unsigned int) ((float) UINT_MAX * mEvolutionParameters.mutationPst);
  mEvolutionParameters.crossoverUintBoundary = (unsigned int) ((float) UINT_MAX * mEvolutionParameters.crossoverPst);

  // Set island Idx and Island count
  MPI_Comm_rank(MPI_COMM_WORLD, &mEvolutionParameters.islandIdx);
  MPI_Comm_size(MPI_COMM_WORLD, &mEvolutionParameters.islandCount);

  setDevice();
} // end of parseCommandline
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy parameters to the GPU constant memory.
 */
void Parameters::copyToDevice()
{
  checkCudaErrors(cudaMemcpyToSymbol(gpuEvolutionParameters, &mEvolutionParameters, sizeof(mEvolutionParameters)));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Set device idx attached to the MPI process.
 */
void Parameters::setDevice()
{
  // Get number of devices per node (must be uniform across nodes!)
  int nDevices = -1;
  checkCudaErrors(cudaGetDeviceCount(&nDevices));

  // MPI processes are consecutive on the node. All nodes have to be equipped
  // with the same number of GPUs
  mDeviceIdx = mEvolutionParameters.islandIdx % nDevices;

  checkCudaErrors(cudaSetDevice(mDeviceIdx));


  cudaDeviceProp 	prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, mDeviceIdx));


  mNumberOfDeviceSM = prop.multiProcessorCount;
}// end of setDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print all parameters.
 */
void Parameters::printOutAllParameters()
{
  if (mEvolutionParameters.islandIdx == 0)
  {
    printf("-----------------------------------------\n");
    printf("--- Evolution parameters --- \n");
    printf("Population size:     %d\n", mEvolutionParameters.populationSize);
    printf("Offspring size:      %d\n", mEvolutionParameters.offspringPopulationSize);
    printf("Chromosome int size: %d\n", mEvolutionParameters.chromosomeSize);
    printf("Chromosome size:     %d\n", mEvolutionParameters.chromosomeSize * mEvolutionParameters.intBlockSize);

    printf("Num of generations:  %d\n", mEvolutionParameters.numOfGenerations);
    printf("\n");


    printf("Crossover pst:       %f\n", mEvolutionParameters.crossoverPst);
    printf("Mutation  pst:       %f\n", mEvolutionParameters.mutationPst);
    printf("Crossover int:       %u\n", mEvolutionParameters.crossoverUintBoundary);
    printf("Mutation  int:       %u\n", mEvolutionParameters.mutationUintBoundary);
    printf("\n");

    printf("Emigrant count:      %d\n", mEvolutionParameters.emigrantCount);
    printf("Migration interval:  %d\n", mEvolutionParameters.migrationInterval);
    printf("Island count:        %d\n", mEvolutionParameters.islandCount);
    printf("Statistics interval: %d\n", mEvolutionParameters.statisticsInterval);

    printf("\n");
    printf("Data File: %s\n",mGlobalDataFileName.c_str());
    printf("-----------------------------------------\n");
  }
}// end of printOutAllParameters
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Private methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class
 */
Parameters::Parameters()
{
  mEvolutionParameters.populationSize      = 128;
  mEvolutionParameters.chromosomeSize      = 128;
  mEvolutionParameters.numOfGenerations    = 100;

  mEvolutionParameters.mutationPst         = 0.01f;
  mEvolutionParameters.crossoverPst        = 0.7f;
  mEvolutionParameters.offspringPopulationSize = (int) (0.5f * mEvolutionParameters.populationSize);

  mEvolutionParameters.islandCount         = 1;
  mEvolutionParameters.emigrantCount       = 1;
  mEvolutionParameters.migrationInterval   = 1;
  mEvolutionParameters.statisticsInterval  = 1;

  mEvolutionParameters.intBlockSize        = sizeof(int) * 8;
  mGlobalDataFileName                      = "";

  mPrintBest                              = false;
  mEvolutionParameters.islandIdx           = 0;

}// end of Parameters
//----------------------------------------------------------------------------------------------------------------------

/**
 * print usage of the algorithm
 */
void Parameters::printUsageAndExit()
{
  if (mEvolutionParameters.islandIdx == 0)
  {
    fprintf(stderr, "Parameters for the genetic algorithm solving knapsack problem: \n");
    fprintf(stderr, "  -p population_size\n");
    fprintf(stderr, "  -g number_of_generations\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "  -m mutation_rate\n");
    fprintf(stderr, "  -c crossover_rate\n");
    fprintf(stderr, "  -o offspring_rate\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "  -e emigrants_rate\n");
    fprintf(stderr, "  -n migration_interval\n");
    fprintf(stderr, "  -s statistics_interval\n");

    fprintf(stderr, "  -b print best individual\n");
    fprintf(stderr, "  -f benchmark_file_name\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "Default population_size       = 128\n");
    fprintf(stderr, "Default number_of_generations = 100\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "Default mutation_rate  = 0.01\n");
    fprintf(stderr, "Default crossover_rate = 0.7\n");
    fprintf(stderr, "Default offspring_rate = 0.5\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "Default island_count        = 1\n");
    fprintf(stderr, "Default migration_interval  = 1\n");
    fprintf(stderr, "Default statistics_interval = 1\n");

    fprintf(stderr, "Default benchmark_file_name = knapsack_data.txt\n");
  }

  MPI_Finalize();
  exit(EXIT_FAILURE);
}// end of printUsageAndExit
//----------------------------------------------------------------------------------------------------------------------
