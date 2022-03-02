/*
 * File:        Evolution.cu
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
 * Comments:    Implementation file of the GA evolution
 *              This class controls the evolution process on multicore CPU
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
 * Revised on 24 February 2022, 16:27 PM
 */

#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>

#include "Evolution.h"
#include "Statistics.h"
#include "CUDAKernels.h"
#include "Parameters.h"

using namespace std;


//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//

static const int MPI_TAG_DATA  = 100;



//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              public methods                                //
//----------------------------------------------------------------------------//

/*
 * Constructor of the class
 */
TGPU_Evolution::TGPU_Evolution(int argc, char **argv)
  : Params(Parameters::getInstance())
{

    MasterPopulation       = NULL;
    OffspringPopulation    = NULL;

    GPU_EmigrantsToSend     = NULL;
    GPU_EmigrantsToReceive  = NULL;
    CPU_EmigrantsToSend     = NULL;
    CPU_EmigrantsToReceive  = NULL;

    GPUStatistics          = NULL;


    // Create parameter class

    // Parse command line
    Params.parseCommandline(argc,argv);

    // Load data from disk
    GlobalData.loadFromFile();

    // Create populations on GPU
    MasterPopulation       = new TGPU_Population(Params.getPopulationSize(), Params.getChromosomeSize());
    OffspringPopulation    = new TGPU_Population(Params.getOffspringPopulationSize(), Params.getChromosomeSize());


    // Create buffers on GPU
    GPU_EmigrantsToSend     = new TGPU_Population(Params.getEmigrantCount(), Params.getChromosomeSize());
    GPU_EmigrantsToReceive  = new TGPU_Population(Params.getEmigrantCount(), Params.getChromosomeSize());
    // Create buffers on CPU
    CPU_EmigrantsToSend     = new TCPU_Population(Params.getEmigrantCount(), Params.getChromosomeSize());
    CPU_EmigrantsToReceive  = new TCPU_Population(Params.getEmigrantCount(), Params.getChromosomeSize());

    // Create Vector lock
    //MigrationVectorLock    = new TGPU_Vector_Lock(Params.PopulationSize());

    // Create statistics
    GPUStatistics          = new Statistics();

    FActGeneration = 0;
    InitSeed();

}// end of TGPU_Evolution
//------------------------------------------------------------------------------


/*
 * Destructor of the class
 */
TGPU_Evolution::~TGPU_Evolution(){


    if (MasterPopulation)       delete MasterPopulation;
    if (OffspringPopulation)    delete OffspringPopulation;

    if (GPU_EmigrantsToSend)    delete GPU_EmigrantsToSend;
    if (GPU_EmigrantsToReceive) delete GPU_EmigrantsToReceive;

    if (CPU_EmigrantsToSend)    delete CPU_EmigrantsToSend;
    if (CPU_EmigrantsToReceive) delete CPU_EmigrantsToReceive;

    if (GPUStatistics)          delete GPUStatistics;



}// end of Destructor
//------------------------------------------------------------------------------

/*
 * Run Evolution
 */
void TGPU_Evolution::Run(){

    Initialize();

    RunEvolutionCycle();

}// end of Run
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              protected methods                             //
//----------------------------------------------------------------------------//


/*
 * Initialize seed
 */
void TGPU_Evolution::InitSeed() {

  struct timeval tp1;

  gettimeofday(&tp1, NULL);


  FRandomSeed = (tp1.tv_sec / (Params.getIslandIdx()+1)) * tp1.tv_usec;

};// end of InitSeed
//------------------------------------------------------------------------------


/*
 * Initialization
 */
void TGPU_Evolution::Initialize(){


    FActGeneration = 0;

    // Store parameters on GPU and print them out
    Params.copyToDevice();
    Params.printOutAllParameters();



    //-- set elements count --//
    int Elements = Params.getChromosomeSize() * Params.getPopulationSize();


    //-- Initialize population --//
    FirstPopulationGenerationKernel
            <<<Params.getNumberOfDeviceSMs() * 2, BLOCK_SIZE>>>
            (MasterPopulation->DeviceData, GetSeed());

    dim3 Blocks;
    dim3 Threads;

    Blocks.x = 1;
    Blocks.y = (Params.getPopulationSize() / (CHR_PER_BLOCK) +1);
    Blocks.z = 1;


    Threads.x = WARP_SIZE;
    Threads.y = CHR_PER_BLOCK;
    Threads.z = 1;


    CalculateKnapsackFintess
            <<<Blocks, Threads>>>
                (MasterPopulation->DeviceData, GlobalData.getDeviceData());


}// end of TGPU_Evolution
//------------------------------------------------------------------------------




/*
 * Run evolutionary cycle for defined number of generations
 *
 */
void TGPU_Evolution::RunEvolutionCycle(){


    dim3 Blocks;
    dim3 Threads;

    Threads.x = WARP_SIZE;
    Threads.y = CHR_PER_BLOCK;
    Threads.z = 1;



    // Evaluate generations
    for (FActGeneration = 1; FActGeneration < Params.getNumOfGenerations(); FActGeneration++) {

          //------------- Migration -----------//
/*          if (FActGeneration % Params.MigrationInterval() == 0) {
              Migrate();
          }*/


          //-------------Selection -----------//
          Blocks.x = 1;
          Blocks.y = (Params.getOffspringPopulationSize() % (CHR_PER_BLOCK << 1)  == 0) ?
                            Params.getOffspringPopulationSize() / (CHR_PER_BLOCK << 1)  :
                            Params.getOffspringPopulationSize() / (CHR_PER_BLOCK << 1) + 1;

          Blocks.z = 1;

          GeneticManipulationKernel
                  <<<Blocks, Threads>>>
                  (MasterPopulation->DeviceData, OffspringPopulation->DeviceData, GetSeed());



          //----------- Evaluation ---------//

          Blocks.x = 1;
          Blocks.y = (Params.getOffspringPopulationSize() % (CHR_PER_BLOCK)  == 0) ?
                            Params.getOffspringPopulationSize() / (CHR_PER_BLOCK)  :
                            Params.getOffspringPopulationSize() / (CHR_PER_BLOCK) + 1;
          Blocks.z = 1;


          CalculateKnapsackFintess
                <<<Blocks, Threads>>>
                    (OffspringPopulation->DeviceData, GlobalData.getDeviceData());


          //----------- Replacement ---------//


          Blocks.x = 1;
          Blocks.y = (Params.getPopulationSize() % (CHR_PER_BLOCK)  == 0) ?
                            Params.getPopulationSize() / (CHR_PER_BLOCK)  :
                            Params.getPopulationSize() / (CHR_PER_BLOCK) + 1;
          Blocks.z = 1;



          ReplacementKernel
                  <<<Blocks, Threads>>>
                  (MasterPopulation->DeviceData, OffspringPopulation->DeviceData, GetSeed());



          TCPU_Population pop(Params.getPopulationSize(), Params.getChromosomeSize());


          MasterPopulation->CopyOut(pop.HostData);
          //printf("First idx %f\n", pop.HostData->Fitness[0]);

          /*GPUStatistics->Calculate(MasterPopulation, Params.GetPrintBest());
          printf("Island %d, Best Fitness %f\n", Params.getIslandIdx(), GPUStatistics->GetMaxFitness());*/


          if (FActGeneration % Params.getStatisticsInterval() == 0){
              GPUStatistics->calculate(MasterPopulation, Params.getPrintBest());



              if (IsMaster()) {
                    printf("Generation %6d, BestIsland %d, MaxFitness %6f, MinFitness %6f, AvgFitness %6f, Diver %6f \n",
                        FActGeneration, GPUStatistics->getBestIslandIdx(),
                        GPUStatistics->getMaxFitness(), GPUStatistics->getMinFitness(),
                        GPUStatistics->getAvgFitness(), GPUStatistics->getDivergence());


                  if (Params.getPrintBest())  printf("%s\n", GPUStatistics->getBestIndividualStr(GlobalData.getHostData()).c_str());
              }// isMaster
          }//FActGen...



    }




        GPUStatistics->calculate(MasterPopulation, true);
        if (IsMaster()){
              printf("--------------------------------------------------------------------------------------------------\n");
              printf("FinalBestIsland %d, FinalMaxFitness %6f, FinalMinFitness %6f, FinalAvgFitness %6f, FinalDiver %6f \n",
                        GPUStatistics->getBestIslandIdx(),
                        GPUStatistics->getMaxFitness(), GPUStatistics->getMinFitness(),
                        GPUStatistics->getAvgFitness(), GPUStatistics->getDivergence());

                //printf("%s\n", GPUStatistics->GetBestIndividualStr(GlobalData.HostData).c_str());
        }// is master

 checkAndReportCudaError(__FILE__,__LINE__);

}// end of RunEvolutionCycle
//------------------------------------------------------------------------------


/*
 * Migrate individuals between GPUs
 *
 */
void TGPU_Evolution::Migrate(){

    dim3 Blocks;
    dim3 Threads;

    // Communication consists of 4 messages
    static const int NumOfMessages = 4;

    MPI_Status  status [NumOfMessages];
    MPI_Request request[NumOfMessages];

    int Target;
    int Source;

    Target = (Params.getIslandIdx() + 1) % Params.getIslandCount(); // Send to the right
    if (Params.getIslandIdx() == 0) Source = Params.getIslandCount()-1;
    else Source = Params.getIslandIdx() - 1;                      // Send to the left


    // Receive immigrants
    MPI_Irecv(CPU_EmigrantsToReceive->HostData->Fitness   ,Params.getEmigrantCount()                           , MPI_FLOAT   , Source, MPI_TAG_DATA, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(CPU_EmigrantsToReceive->HostData->Population,Params.getEmigrantCount() * Params.getChromosomeSize(), MPI_UNSIGNED, Source, MPI_TAG_DATA, MPI_COMM_WORLD, &request[3]);



        Threads.x = WARP_SIZE;
        Threads.y = CHR_PER_BLOCK;
        Threads.z = 1;

        Blocks.x  = 1;
        Blocks.y  = (Params.getEmigrantCount() % CHR_PER_BLOCK  == 0) ?
                     Params.getEmigrantCount() / CHR_PER_BLOCK :
                     Params.getEmigrantCount() / CHR_PER_BLOCK  + 1;
        Blocks.z  = 1;


        // Fill emigrant population
      SelectEmigrantsKernel
                   <<<Blocks, Threads>>>
                   (MasterPopulation->DeviceData, GPU_EmigrantsToSend->DeviceData, GetSeed());


      GPU_EmigrantsToSend->CopyOut(CPU_EmigrantsToSend->HostData);


  // Send emigrants
  MPI_Isend(CPU_EmigrantsToSend->HostData->Fitness      ,Params.getEmigrantCount()                           , MPI_FLOAT   , Target, MPI_TAG_DATA, MPI_COMM_WORLD, &request[0]);
  MPI_Isend(CPU_EmigrantsToSend->HostData->Population   ,Params.getEmigrantCount() * Params.getChromosomeSize(), MPI_UNSIGNED, Target, MPI_TAG_DATA, MPI_COMM_WORLD, &request[1]);

  MPI_Waitall(NumOfMessages, request, status);

        GPU_EmigrantsToReceive->CopyIn(CPU_EmigrantsToReceive->HostData);

     //-- Fill emigrant population --//

        AcceptEmigrantsKernel
                     <<<Blocks, Threads>>>
                     (MasterPopulation->DeviceData, GPU_EmigrantsToReceive->DeviceData, GetSeed());


}// End of Migrate
//------------------------------------------------------------------------------