/**
 * @file        Population.cpp
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
 * @brief       Header file of the GA population.
 *              This class maintains and GA populations.
 *
 * @date        08 June      2012, 00:00 (created)
 *              11 April     2022, 20:59 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#include <stdexcept>
#include <helper_cuda.h>

#include "Population.h"
#include "CUDAKernels.h"


//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- GPUPopulation ----------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class.
 */
GPUPopulation::GPUPopulation(const int populationSize,
                             const int chromosomeSize)
{
  mHostPopulationHandler.chromosomeSize = chromosomeSize;
  mHostPopulationHandler.populationSize = populationSize;

  allocateMemory();
}// end of GPUPopulation
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class
 */
GPUPopulation::~GPUPopulation()
{
  freeMemory();
}// end of ~GPUPopulation
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from CPU population structure to GPU
 * Both population must have the same size (sizes not being copied)!!
 *
 */
void GPUPopulation::copyToDevice(const PopulationData* hostPopulation)
{
  // Basic data check
  if (hostPopulation->chromosomeSize != mHostPopulationHandler.chromosomeSize)
  {
    throw std::out_of_range("Wrong chromosome size in copyToDevice function");
  }

  if (hostPopulation->populationSize != mHostPopulationHandler.populationSize)
  {
    throw std::out_of_range("Wrong population size in copyToDevice function");
  }

  // Copy chromosomes
  checkCudaErrors(cudaMemcpy(mHostPopulationHandler.population,
                             hostPopulation->population,
                             sizeof(Gene) * mHostPopulationHandler.chromosomeSize *
                                mHostPopulationHandler.populationSize,
                             cudaMemcpyHostToDevice));

  // Copy fitness values
  checkCudaErrors(cudaMemcpy(mHostPopulationHandler.fitness,
                             hostPopulation->fitness,
                             sizeof(Fitness) * mHostPopulationHandler.populationSize,
                             cudaMemcpyHostToDevice));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from GPU population structure to CPU
 * Both population must have the same size (sizes not copied)!!
 */
void GPUPopulation::copyFromDevice(PopulationData* hostPopulation)
{
  if (hostPopulation->chromosomeSize != mHostPopulationHandler.chromosomeSize)
  {
    throw std::out_of_range("Wrong chromosome size in copyFromDevice function");
  }

  if (hostPopulation->populationSize != mHostPopulationHandler.populationSize)
  {
    throw std::out_of_range("Wrong population size in copyFromDevice function");
  }

  // Copy chromosomes
  checkCudaErrors(cudaMemcpy(hostPopulation->population,
                             mHostPopulationHandler.population,
                             sizeof(Gene) * mHostPopulationHandler.chromosomeSize *
                                 mHostPopulationHandler.populationSize,
                             cudaMemcpyDeviceToHost));

  // Copy fitness values
  checkCudaErrors(cudaMemcpy(hostPopulation->fitness,
                             mHostPopulationHandler.fitness,
                             sizeof(Fitness) * mHostPopulationHandler.populationSize,
                             cudaMemcpyDeviceToHost));
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------


/**
 * Copy data from different population (both on the same GPU)
 * No size check!!!
 */
void GPUPopulation::copyOnDevice(const GPUPopulation* sourceDevicePopulation)
{
  if (sourceDevicePopulation->mHostPopulationHandler.chromosomeSize != mHostPopulationHandler.chromosomeSize)
  {
    throw std::out_of_range("Wrong chromosome size in copyOnDevice function");
  }

  if (sourceDevicePopulation->mHostPopulationHandler.populationSize != mHostPopulationHandler.populationSize)
  {
    throw std::out_of_range("Wrong population size in copyOnDevice function");
  }

  // Copy chromosomes
  checkCudaErrors(cudaMemcpy(mHostPopulationHandler.population,
                             sourceDevicePopulation->mHostPopulationHandler.population,
                             sizeof(Gene) * mHostPopulationHandler.chromosomeSize *
                                 mHostPopulationHandler.populationSize,
                             cudaMemcpyDeviceToDevice));



  // Copy fitness values
  checkCudaErrors(cudaMemcpy(mHostPopulationHandler.fitness,
                             sourceDevicePopulation->mHostPopulationHandler.fitness,
                             sizeof(Fitness) * mHostPopulationHandler.populationSize,
                             cudaMemcpyDeviceToDevice));
}// end of copyOnDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy out only one individual
 *
 */
void GPUPopulation::copyIndividualFromDevice(Gene* individual,
                                             int   index)
{
  checkCudaErrors(cudaMemcpy(individual,
                             &(mHostPopulationHandler.population[index * mHostPopulationHandler.chromosomeSize]),
                             sizeof(Gene) * mHostPopulationHandler.chromosomeSize,
                             cudaMemcpyDeviceToHost));
}// end of copyIndividualFromDevice
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- GPUPopulation ----------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory.
 */
void GPUPopulation::allocateMemory()
{
  // Allocate data structure
  checkCudaErrors(cudaMalloc<PopulationData>(&mDeviceData, sizeof(PopulationData)));


  // Allocate Population data
  checkCudaErrors(cudaMalloc<Gene>(&(mHostPopulationHandler.population),
                                   sizeof(Gene) * mHostPopulationHandler.chromosomeSize *
                                   mHostPopulationHandler.populationSize));

  // Allocate Fitness data
  checkCudaErrors(cudaMalloc<Fitness>(&(mHostPopulationHandler.fitness),
                                      sizeof(Fitness) * mHostPopulationHandler.populationSize));


  // Copy structure to GPU
  checkCudaErrors(cudaMemcpy(mDeviceData,
                             &mHostPopulationHandler,
                             sizeof(PopulationData),
                             cudaMemcpyHostToDevice));
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/*
 * Free memory.
 */
void GPUPopulation::freeMemory()
{
  // Free population data
  checkCudaErrors(cudaFree(mHostPopulationHandler.population));
  //Free Fitness data
  checkCudaErrors(cudaFree(mHostPopulationHandler.fitness));
  // Free whole structure
  checkCudaErrors(cudaFree(mDeviceData));
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- CPUPopulation ----------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class
 */
CPUPopulation::CPUPopulation(const int populationSize,
                             const int chromosomeSize)
{
  mHostData = new(PopulationData);
  mHostData->chromosomeSize = chromosomeSize;
  mHostData->populationSize = populationSize;

  allocateMemory();
}// end of CPUPopulation
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class
 */
CPUPopulation::~CPUPopulation()
{
  freeMemory();

  delete (mHostData);
}// end of ~CPUPopulation
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- CPUPopulation ----------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory
 */
void CPUPopulation::allocateMemory()
{
  // Allocate Population on the host side
  checkCudaErrors(cudaHostAlloc<Gene>(&(mHostData->population),
                                      sizeof(Gene) * mHostData->chromosomeSize * mHostData->populationSize,
                                      cudaHostAllocDefault));


  // Allocate fitness on the host side
  checkCudaErrors(cudaHostAlloc<Fitness>(&(mHostData->fitness),
                                         sizeof(Fitness) * mHostData->populationSize,
                                         cudaHostAllocDefault));

}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory
 */
void CPUPopulation::freeMemory()
{
  // Free population on the host side
  checkCudaErrors(cudaFreeHost(mHostData->population));

  // Free fitness on the host side
  checkCudaErrors(cudaFreeHost(mHostData->fitness));
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------
