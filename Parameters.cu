/*
 * File:        Parameters.cu
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
 * Comments:    The implementation  parameter class. It contains all the parameters of
 *              GA and knapsack
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
 * Revised on 24 February 2022, 16:24 PM
 */


#include <iostream>
#include <cuda_runtime.h>
#include <mpi.h>
#include <getopt.h>

#include "Parameters.h"
#include "CUDAKernels.h"

//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//
extern __constant__  TEvolutionParameters GPU_EvolutionParameters;

// Singleton initialization
bool TParameters::pTParametersInstanceFlag = false;
TParameters* TParameters::pTParametersSingle = NULL;


//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              public methods                                //
//----------------------------------------------------------------------------//

/*
 * Get instance of TPrarams
 */
TParameters* TParameters::GetInstance(){
    if(! pTParametersInstanceFlag)
    {
        pTParametersSingle = new TParameters();
        pTParametersInstanceFlag = true;
        return pTParametersSingle;
    }
    else
    {
        return pTParametersSingle;
    }
}// end of TParameters::GetInstance
//-----------------------------------------------------------------------------


/*
 * Load parameters from command line
 *
 * @param argc
 * @param argv
 *
 */
void TParameters::LoadParametersFromCommandLine(int argc, char **argv){


   float OffspringPercentage = 0.5f;
   float EmigrantPercentage = 0.1f;
   char c;



   while ((c = getopt (argc, argv, "p:g:m:c:o:e:n:f:s:bh")) != -1){
       switch (c){
          case 'p':{
              if (atoi(optarg) != 0) EvolutionParameters.PopulationSize = atoi(optarg);
              break;
          }
          case 'g': {
              if (atoi(optarg) != 0) EvolutionParameters.NumOfGenerations = atoi(optarg);
              break;
          }


          case 'm': {
              if (atof(optarg) != 0) EvolutionParameters.MutationPst = atof(optarg);
              break;
          }
          case 'c': {
              if (atof(optarg) != 0) EvolutionParameters.CrossoverPst = atof(optarg);
              break;
          }
          case 'o': {
              if (atof(optarg) != 0) OffspringPercentage = atof(optarg);;
              break;
          }


         case 'e': {
              if (atof(optarg) != 0) EmigrantPercentage = atof(optarg);;
              break;
          }
          case 'n': {
              if (atoi(optarg) != 0) EvolutionParameters.MigrationInterval = atoi(optarg);
              break;
          }

         case 's': {
              if (atoi(optarg) != 0) EvolutionParameters.StatisticsInterval = atoi(optarg);
              break;
          }

         case 'b': {
              FPrintBest = true;
              break;
          }

         case 'f': {
              GlobalDataFileName  = optarg;
              break;
          }
          case 'h':{

             PrintUsageAndExit();
             break;
          }
          default:{

               PrintUsageAndExit();
          }
       }
   }

   // Set population size to be even.
   if (EvolutionParameters.PopulationSize % 2 == 1) EvolutionParameters.PopulationSize++;

   EvolutionParameters.OffspringPopulationSize = (int) (OffspringPercentage * EvolutionParameters.PopulationSize);
   if (EvolutionParameters.OffspringPopulationSize == 0) EvolutionParameters.OffspringPopulationSize = 2;
   if (EvolutionParameters.OffspringPopulationSize % 2 == 1) EvolutionParameters.OffspringPopulationSize++;

     // Check emigrant count and set it at least to 1
   EvolutionParameters.EmigrantCount = (int) (EmigrantPercentage * EvolutionParameters.PopulationSize);
   if (EvolutionParameters.EmigrantCount == 0)  EvolutionParameters.EmigrantCount = 1;
   if ((EvolutionParameters.EmigrantCount % 2) == 0) EvolutionParameters.EmigrantCount++;

   if (EvolutionParameters.EmigrantCount > EvolutionParameters.PopulationSize)  EvolutionParameters.EmigrantCount = EvolutionParameters.PopulationSize;

   if (EvolutionParameters.MigrationInterval < 0) EvolutionParameters.MigrationInterval = 1;

    // Set UINT mutation threshold to faster comparison
   EvolutionParameters.MutationUINTBoundary  = (unsigned int) ((float) UINT_MAX * EvolutionParameters.MutationPst);
   EvolutionParameters.CrossoverUINTBoundary = (unsigned int) ((float) UINT_MAX * EvolutionParameters.CrossoverPst);


   // Set island Idx and Island count
   MPI_Comm_rank(MPI_COMM_WORLD, &EvolutionParameters.IslandIdx);
   MPI_Comm_size(MPI_COMM_WORLD, &EvolutionParameters.IslandCount);

   SetGPU();

} // end of LoadParametersFromCommandLine
//------------------------------------------------------------------------------


/*
 * Copy parameters to the GPU constant memory
 */
void TParameters::StoreParamsOnGPU()
{
  cudaMemcpyToSymbol(GPU_EvolutionParameters, &EvolutionParameters, sizeof(TEvolutionParameters));
  checkAndReportCudaError(__FILE__,__LINE__);
}// end of StoreParamsOnGPU
//------------------------------------------------------------------------------


/*
 * Return GPU id attached to the MPI process
 */
void TParameters::SetGPU()
{
    // Get number of devices per node (must be uniform accross nodes!)
    int nDevices = -1;
    cudaGetDeviceCount(&nDevices);
    checkAndReportCudaError(__FILE__,__LINE__);

    // MPI processes are consecutive on the node. All nodes have to be equipped
    // with the same number of GPUs
    FGPUIdx = EvolutionParameters.IslandIdx % nDevices;

    cudaSetDevice(FGPUIdx);
    checkAndReportCudaError(__FILE__,__LINE__);

    cudaDeviceProp 	prop;
    cudaGetDeviceProperties (&prop, FGPUIdx);
    checkAndReportCudaError(__FILE__,__LINE__);

    FGPU_SM_Count = prop.multiProcessorCount;

}// end of GetGPUIdx
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              private methods                               //
//----------------------------------------------------------------------------//

/*
 * Constructor of the class
 */
TParameters::TParameters(){

    EvolutionParameters.PopulationSize      = 128;
    EvolutionParameters.ChromosomeSize      = 128;
    EvolutionParameters.NumOfGenerations    = 100;

    EvolutionParameters.MutationPst         = 0.01f;
    EvolutionParameters.CrossoverPst        = 0.7f;
    EvolutionParameters.OffspringPopulationSize = (int) (0.5f * EvolutionParameters.PopulationSize);

    EvolutionParameters.IslandCount         = 1;
    EvolutionParameters.EmigrantCount       = 1;
    EvolutionParameters.MigrationInterval   = 1;
    EvolutionParameters.StatisticsInterval  = 1;

    EvolutionParameters.IntBlockSize        = sizeof(int)*8;
    GlobalDataFileName                      = "";

    FPrintBest                              = false;
    EvolutionParameters.IslandIdx           = 0;

}// end of TParameters
//------------------------------------------------------------------------------

/*
 * print usage of the algorithm
 */
void TParameters::PrintUsageAndExit(){

  if (EvolutionParameters.IslandIdx == 0){
      cerr << "Usage: " << endl;
      cerr << "  -p Population_size\n";
      cerr << "  -g Number_of_generations\n";
      cerr << endl;

      cerr << "  -m mutation_rate\n";
      cerr << "  -c crossover_rate\n";
      cerr << "  -o offspring_rate\n";
      cerr << endl;

      cerr << "  -e emigrants_rate\n";
      cerr << "  -n migration_interval\n";
      cerr << "  -s statistics_interval\n";
      cerr << endl;

      cerr << "  -b print best individual\n";
      cerr << "  -f benchmark_file_name\n";


      cerr << endl;
      cerr << "Default Population_size       = 128"  << endl;
      cerr << "Default Number_of_generations = 100" << endl;
      cerr << endl;

      cerr << "Default mutation_rate  = 0.01" << endl;
      cerr << "Default crossover_rate = 0.7" << endl;
      cerr << "Default offspring_rate = 0.5" << endl;
      cerr << endl;

      cerr << "Default island_count        = 1"   << endl;
      cerr << "Default emigrants_rate      = 0.1" << endl;
      cerr << "Default migration_interval  = 1"   << endl;
      cerr << "Default statistics_interval = 1"   << endl;

      cerr << "Default benchmark_file_name = knapsack_data.txt\n";

  }
  MPI_Finalize();
  exit(1);

}// end of PrintUsage
//------------------------------------------------------------------------------





/*
 * Print all parameters
 *
 */
void TParameters::PrintAllParameters(){

    if (EvolutionParameters.IslandIdx == 0){
        printf("-----------------------------------------\n");
        printf("--- Evolution parameters --- \n");
        printf("Population size:     %d\n", EvolutionParameters.PopulationSize);
        printf("Offspring size:      %d\n", EvolutionParameters.OffspringPopulationSize);
        printf("Chromosome int size: %d\n", EvolutionParameters.ChromosomeSize);
        printf("Chromosome size:     %d\n", EvolutionParameters.ChromosomeSize * EvolutionParameters.IntBlockSize);

        printf("Num of generations:  %d\n", EvolutionParameters.NumOfGenerations);
        printf("\n");


        printf("Crossover pst:       %f\n", EvolutionParameters.CrossoverPst);
        printf("Mutation  pst:       %f\n", EvolutionParameters.MutationPst);
        printf("Crossover int:       %u\n", EvolutionParameters.CrossoverUINTBoundary);
        printf("Mutation  int:       %u\n", EvolutionParameters.MutationUINTBoundary);
        printf("\n");

        printf("Emigrant count:      %d\n", EvolutionParameters.EmigrantCount);
        printf("Migration interval:  %d\n", EvolutionParameters.MigrationInterval);
        printf("Island count:        %d\n", EvolutionParameters.IslandCount);
        printf("Statistics interval: %d\n", EvolutionParameters.StatisticsInterval);

        printf("\n");
        printf("Data File: %s\n",GlobalDataFileName.c_str());
        printf("-----------------------------------------\n");
    }

}// end of PrintAllParameters
//------------------------------------------------------------------------------
