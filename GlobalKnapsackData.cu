/*
 * File:        GlobalKnapsackData.cu
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
 * Comments:    Implementation file of the knapsack global data class.
 *              This class maintains the benchmark data
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
 * Revised on 24 February 2022, 18:59 PM
 */


#include <helper_cuda.h>
#include <fstream>

#include "GlobalKnapsackData.h"
#include "Parameters.h"

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

static const char* const ERROR_FILE_NOT_FOUND = "Global Benchmark Data: File not found\n";


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Destructor of the class
 */
GlobalKnapsackData::~GlobalKnapsackData()
{
  freeMemory();
}// end of GlobalKnapsackData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Load data from file, filename given in Parameter class.
 */
void GlobalKnapsackData::loadFromFile()
{
  // Get instance of Parameter class
  Parameters& params = Parameters::getInstance();

  // Open file with benchmark data
  std::ifstream fr(params.getBenchmarkFileName().c_str());
  if (!fr.is_open())
  {
    fprintf(stderr, ERROR_FILE_NOT_FOUND);
    exit(EXIT_FAILURE);
  }

  // Read number of items
  int numberOfItems = 0;
  fr >> numberOfItems;

  const int originalNumberOfItems = numberOfItems;

  // Calculate padding
  int overhead = numberOfItems % (params.getIntBlockSize() * WARP_SIZE);
  if (overhead != 0)
  {
    numberOfItems = numberOfItems + ((params.getIntBlockSize() * WARP_SIZE) - overhead);
  }

  // Allocate memory for arrays
  allocateMemory(numberOfItems);

  mHostData->numberOfItems         = numberOfItems;
  mHostData->originalNumberOfItems = originalNumberOfItems;


  // Load prices
  for (size_t i = 0; i < originalNumberOfItems; i++)
  {
    fr >> mHostData->itemPrice[i];
  }
  // add padding
  for (size_t i = originalNumberOfItems; i < numberOfItems; i++)
  {
    mHostData->itemPrice[i] = PriceType(0);
  }


  // Load weights
  for (size_t i = 0; i < originalNumberOfItems; i++)
  {
    fr >> mHostData->itemWeight[i];
  } // add padding
  for (size_t i = originalNumberOfItems; i < numberOfItems; i++)
  {
    mHostData->itemWeight[i] = PriceType(0);
  }

  // Get max Price/Weight ratio
  mHostData->maxPriceWightRatio = 0.0f;

  for (size_t i = 0; i < originalNumberOfItems; i++)
  {
    if (mHostData->itemWeight[i] != 0)
    {
      float ratio = mHostData->itemPrice[i] / mHostData->itemWeight[i];
      if (ratio > mHostData->maxPriceWightRatio)
      {
        mHostData->maxPriceWightRatio = ratio;
      }
    }
  }

  // Read Knapsack capacity
  fr >> mHostData->knapsackCapacity;

  // Update chromosome size in parameters
  params.setChromosomeSize(numberOfItems / params.getIntBlockSize());


  // Upload global data to device memory
  copyToDevice();
}// end of loadFromFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory
 */
void GlobalKnapsackData::allocateMemory(int numberOfItems)
{
  //------------------------- Host allocation ------------------------------//
  checkCudaErrors(
      cudaHostAlloc<KnapsackData>(&mHostData,  sizeof(KnapsackData), cudaHostAllocDefault)
  );

  checkCudaErrors(
      cudaHostAlloc<PriceType>(&mHostData->itemPrice,  sizeof(PriceType) * numberOfItems, cudaHostAllocDefault)
  );

  checkCudaErrors(
      cudaHostAlloc<WeightType>(&mHostData->itemWeight,  sizeof(WeightType) * numberOfItems, cudaHostAllocDefault)
  );


  //----------------------- Device allocation ------------------------------//
  checkCudaErrors(
      cudaMalloc<KnapsackData>(&mDeviceData,  sizeof(KnapsackData) )
  );

  checkCudaErrors(
      cudaMalloc<PriceType>(&mDeviceItemPriceHandler,  sizeof(PriceType) * numberOfItems)
  );

  checkCudaErrors(
      cudaMalloc<WeightType>(&mDeviceItemWeightHandler, sizeof(WeightType) * numberOfItems)
  );
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free Memory.
 */
void GlobalKnapsackData::freeMemory()
{

    //------------------------- Host allocation ------------------------------//
    checkCudaErrors(cudaFreeHost(mHostData->itemPrice));
    checkCudaErrors(cudaFreeHost(mHostData->itemWeight));
    checkCudaErrors(cudaFreeHost(mHostData));

    //----------------------- Device allocation ------------------------------//
    checkCudaErrors(cudaFree(mDeviceData));
    checkCudaErrors(cudaFree(mDeviceItemPriceHandler));
    checkCudaErrors(cudaFree(mDeviceItemWeightHandler));

}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Upload Data to Device.
 */
void GlobalKnapsackData::copyToDevice()
{
  // Copy basic structure - struct data
  checkCudaErrors(cudaMemcpy(mDeviceData, mHostData, sizeof(KnapsackData), cudaMemcpyHostToDevice));


  // Set pointer of the ItemPrice vector into the struct on GPU (link struct and vector)
  checkCudaErrors(
      cudaMemcpy(&(mDeviceData->itemPrice), &mDeviceItemPriceHandler, sizeof(PriceType*),cudaMemcpyHostToDevice)
  );


    // Set pointer of the ItemWeight vector into struct on GPU (link struct and vector)
  checkCudaErrors(
      cudaMemcpy(&(mDeviceData->itemWeight), &mDeviceItemWeightHandler, sizeof(WeightType*), cudaMemcpyHostToDevice)
  );

  // Copy prices
  checkCudaErrors(
      cudaMemcpy(mDeviceItemPriceHandler, mHostData->itemPrice,  sizeof(PriceType) * mHostData->numberOfItems,
                    cudaMemcpyHostToDevice)
  );

  // Copy weights
  checkCudaErrors(
      cudaMemcpy(mDeviceItemWeightHandler, mHostData->itemWeight, sizeof(WeightType) * mHostData->numberOfItems,
                 cudaMemcpyHostToDevice)
  );

}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------