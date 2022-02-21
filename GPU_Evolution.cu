/* 
 * File:        GPU_Evolution.cu
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
 */

#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>

#include "GPU_Evolution.h"
#include "GPU_Statistics.h"
#include "CUDA_Kernels.h"
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
TGPU_Evolution::TGPU_Evolution(int argc, char **argv)  {
    
    MasterPopulation       = NULL;
    OffspringPopulation    = NULL;    
    
    GPU_EmigrantsToSend     = NULL;
    GPU_EmigrantsToReceive  = NULL;
    CPU_EmigrantsToSend     = NULL;
    CPU_EmigrantsToReceive  = NULL;
    
    GPUStatistics          = NULL;
    MigrationVectorLock    = NULL;
    
    // Create parameter class
    Params = TParameters::GetInstance();
    // Parse command line
    Params->LoadParametersFromCommandLine(argc,argv);    
        
    // Load data from disk
    GlobalData.LoadFromFile();
    
    // Create populations on GPU
    MasterPopulation       = new TGPU_Population(Params->PopulationSize(), Params->ChromosomeSize());    
    OffspringPopulation    = new TGPU_Population(Params->OffspringPopulationSize(), Params->ChromosomeSize());
    
    
    // Create buffers on GPU
    GPU_EmigrantsToSend     = new TGPU_Population(Params->EmigrantCount(), Params->ChromosomeSize());
    GPU_EmigrantsToReceive  = new TGPU_Population(Params->EmigrantCount(), Params->ChromosomeSize());
    // Create buffers on CPU
    CPU_EmigrantsToSend     = new TCPU_Population(Params->EmigrantCount(), Params->ChromosomeSize());
    CPU_EmigrantsToReceive  = new TCPU_Population(Params->EmigrantCount(), Params->ChromosomeSize());          
    
    // Create Vector lock
    MigrationVectorLock    = new TGPU_Vector_Lock(Params->PopulationSize());                
    
    // Create statistics
    GPUStatistics          = new TGPU_Statistics();
    
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
    
    
    if (MigrationVectorLock)    delete MigrationVectorLock;        
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
  
  
  FRandomSeed = (tp1.tv_sec / (Params->IslandIdx()+1)) * tp1.tv_usec;
  
};// end of InitSeed
//------------------------------------------------------------------------------
   

/*
 * Initialization
 */
void TGPU_Evolution::Initialize(){
    
        
    FActGeneration = 0;    
    
    // Store parameters on GPU and print them out
    Params->StoreParamsOnGPU();       
    Params->PrintAllParameters();
    
    
    
    //-- set elements count --//       
    int Elements = Params->ChromosomeSize() * Params->PopulationSize();
    
    
    //-- Initialize population --//
    FirstPopulationGenerationKernel
            <<<Params->GetGPU_SM_Count() * 2, BLOCK_SIZE>>>
            (MasterPopulation->DeviceData, GetSeed());

    dim3 Blocks; 
    dim3 Threads;
            
    Blocks.x = 1;    
    Blocks.y = (Params->PopulationSize() / (CHR_PER_BLOCK) +1);
    Blocks.z = 1;
    
    
    Threads.x = WARP_SIZE;
    Threads.y = CHR_PER_BLOCK;
    Threads.z = 1;
    
    
    CalculateKnapsackFintess
            <<<Blocks, Threads>>>
                (MasterPopulation->DeviceData, GlobalData.DeviceData);
    
    
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
    for (FActGeneration = 1; FActGeneration < Params->NumOfGenerations(); FActGeneration++) {
      
          //------------- Migration -----------//
          if (FActGeneration % Params->MigrationInterval() == 0) {              
              Migrate();
          }
        
        
          //-------------Selection -----------//
          Blocks.x = 1;    
          Blocks.y = (Params->OffspringPopulationSize() % (CHR_PER_BLOCK << 1)  == 0) ?
                            Params->OffspringPopulationSize() / (CHR_PER_BLOCK << 1)  :
                            Params->OffspringPopulationSize() / (CHR_PER_BLOCK << 1) + 1;

          Blocks.z = 1;

          GeneticManipulationKernel
                  <<<Blocks, Threads>>>
                  (MasterPopulation->DeviceData, OffspringPopulation->DeviceData, GetSeed());



          //----------- Evaluation ---------//

          Blocks.x = 1;    
          Blocks.y = (Params->OffspringPopulationSize() % (CHR_PER_BLOCK)  == 0) ?
                            Params->OffspringPopulationSize() / (CHR_PER_BLOCK)  :
                            Params->OffspringPopulationSize() / (CHR_PER_BLOCK) + 1;
          Blocks.z = 1;


          CalculateKnapsackFintess
                <<<Blocks, Threads>>>
                    (OffspringPopulation->DeviceData, GlobalData.DeviceData);


          //----------- Replacement ---------//


          Blocks.x = 1;    
          Blocks.y = (Params->PopulationSize() % (CHR_PER_BLOCK)  == 0) ?
                            Params->PopulationSize() / (CHR_PER_BLOCK)  :
                            Params->PopulationSize() / (CHR_PER_BLOCK) + 1;
          Blocks.z = 1;



          ReplacementKernel
                  <<<Blocks, Threads>>>
                  (MasterPopulation->DeviceData, OffspringPopulation->DeviceData, GetSeed());

          
          if (FActGeneration % Params->StatisticsInterval() == 0){
              GPUStatistics->Calculate(MasterPopulation, Params->GetPrintBest());
          
              
              if (IsMaster()) {                  
                    printf("Generation %6d, BestIsland %d, MaxFitness %6f, MinFitness %6f, AvgFitness %6f, Diver %6f \n", 
                        FActGeneration, GPUStatistics->GetBestIslandIdx(),
                        GPUStatistics->GetMaxFitness(), GPUStatistics->GetMinFitness(),
                        GPUStatistics->GetAvgFitness(), GPUStatistics->GetDivergence());
              
              
                  if (Params->GetPrintBest())  printf("%s\n", GPUStatistics->GetBestIndividualStr(GlobalData.HostData).c_str());
              }// isMaster    
          }//FActGen...
                  
              
                  
    }
        
    
    
                  
        GPUStatistics->Calculate(MasterPopulation, true);
        if (IsMaster()){         
              printf("--------------------------------------------------------------------------------------------------\n");   
              printf("FinalBestIsland %d, FinalMaxFitness %6f, FinalMinFitness %6f, FinalAvgFitness %6f, FinalDiver %6f \n", 
                        GPUStatistics->GetBestIslandIdx(),
                        GPUStatistics->GetMaxFitness(), GPUStatistics->GetMinFitness(),
                        GPUStatistics->GetAvgFitness(), GPUStatistics->GetDivergence());
                
                //printf("%s\n", GPUStatistics->GetBestIndividualStr(GlobalData.HostData).c_str());
        }// is master
              
   
    
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
    
    Target = (Params->IslandIdx() + 1) % Params->IslandCount(); // Send to the right
    if (Params->IslandIdx() == 0) Source = Params->IslandCount()-1;
    else Source = Params->IslandIdx() - 1;                      // Send to the left

              
    // Receive immigrants
    MPI_Irecv(CPU_EmigrantsToReceive->HostData->Fitness   ,Params->EmigrantCount()                           , MPI_FLOAT   , Source, MPI_TAG_DATA, MPI_COMM_WORLD, &request[2]);         
    MPI_Irecv(CPU_EmigrantsToReceive->HostData->Population,Params->EmigrantCount() * Params->ChromosomeSize(), MPI_UNSIGNED, Source, MPI_TAG_DATA, MPI_COMM_WORLD, &request[3]); 
    
    
        
        Threads.x = WARP_SIZE;
        Threads.y = CHR_PER_BLOCK;
        Threads.z = 1;    

        Blocks.x  = 1;    
        Blocks.y  = (Params->EmigrantCount() % CHR_PER_BLOCK  == 0) ?
                     Params->EmigrantCount() / CHR_PER_BLOCK :
                     Params->EmigrantCount() / CHR_PER_BLOCK  + 1;
        Blocks.z  = 1;


        // Fill emigrant population 
      SelectEmigrantsKernel
                   <<<Blocks, Threads>>>
                   (MasterPopulation->DeviceData, GPU_EmigrantsToSend->DeviceData, GetSeed());


      GPU_EmigrantsToSend->CopyOut(CPU_EmigrantsToSend->HostData);
      
  
  // Send emigrants    
  MPI_Isend(CPU_EmigrantsToSend->HostData->Fitness      ,Params->EmigrantCount()                           , MPI_FLOAT   , Target, MPI_TAG_DATA, MPI_COMM_WORLD, &request[0]);         
  MPI_Isend(CPU_EmigrantsToSend->HostData->Population   ,Params->EmigrantCount() * Params->ChromosomeSize(), MPI_UNSIGNED, Target, MPI_TAG_DATA, MPI_COMM_WORLD, &request[1]);         

  MPI_Waitall(NumOfMessages, request, status);

        GPU_EmigrantsToReceive->CopyIn(CPU_EmigrantsToReceive->HostData);
    
     //-- Fill emigrant population --//     
        
        AcceptEmigrantsKernel
                     <<<Blocks, Threads>>>
                     (MasterPopulation->DeviceData, GPU_EmigrantsToReceive->DeviceData, *MigrationVectorLock, GetSeed());

    
}// End of Migrate
//------------------------------------------------------------------------------