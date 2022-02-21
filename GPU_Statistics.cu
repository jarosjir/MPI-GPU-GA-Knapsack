/* 
 * File:        GPU_Statistics.cu
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
 * Comments:    Implementation file of the GA statistics
 *              This class maintains and collects GA statistics
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

#include <cutil_inline.h>
#include <sstream>
#include <malloc.h>
#include <mpi.h>

#include "GPU_Statistics.h"
#include "CUDA_Kernels.h"


//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//                       TGPU_Statistics Implementation                       //
//                              public methods                                //
//----------------------------------------------------------------------------//


/*
 * Constructor of the class
 * 
 */
TGPU_Statistics::TGPU_Statistics(){
       
    TParameters*  Params = TParameters::GetInstance();

    GlobalDerivedStat       = NULL;
    ReceiveStatDataBuffer   = NULL;
    ReceiveIndividualBuffer = NULL;
    
    //-- for MPI collection function --// 
    if (Params->IslandIdx() == 0) {
      GlobalDerivedStat          = (TDerivedStats *)       memalign(16, sizeof(TDerivedStats));          
      ReceiveStatDataBuffer      = (TStatDataToExchange *) memalign(16, sizeof(TStatDataToExchange) * Params->IslandCount());          
      ReceiveIndividualBuffer    = (TGene *)               memalign(16, sizeof(TGene)  * Params->ChromosomeSize()* Params->IslandCount());
    }
       
    // Allocate CUDA memory
    AllocateCudaMemory();
            
}// end of TGPU_Population
//------------------------------------------------------------------------------


/*
 * Destructor of the class
 * 
 */
TGPU_Statistics::~TGPU_Statistics(){
    
    //-- Free host memory --//
    if (GlobalDerivedStat){
        free(GlobalDerivedStat);       
        GlobalDerivedStat = NULL;
    }
    
    if (ReceiveStatDataBuffer){
        free(ReceiveStatDataBuffer);       
        ReceiveStatDataBuffer = NULL;
    }
    if (ReceiveIndividualBuffer){
        free(ReceiveIndividualBuffer);
        ReceiveIndividualBuffer = NULL;
    }
    
    FreeCudaMemory();
           
}// end of ~TGPU_Population
//------------------------------------------------------------------------------


/*
 * Print best individual as a string
 *
 * @param Global knapsack data
 * @retur Best individual in from of a sting 
 */
string TGPU_Statistics::GetBestIndividualStr(TKnapsackData * GlobalKnapsackData){

    stringstream  S; 
    
    TParameters * Params = TParameters::GetInstance();
    
 
    int  BlockCount    = GlobalKnapsackData->OriginalNumberOfItems / Params->IntBlockSize();
    bool IsNotFullBlock = (GlobalKnapsackData->OriginalNumberOfItems % Params->IntBlockSize()) != 0;
    
        // Convert by eight bits
    for (int BlockID=0; BlockID < BlockCount; BlockID++){
     
     for (int BitID = 0; BitID < Params->IntBlockSize() -1; BitID++ ) {         
         char c = ((LocalBestIndividual[BlockID] & (1 << BitID)) == 0) ? '0' : '1';         
         S << c;
         if (BitID % 8 ==7) S << " ";
     }    
         
     S << "\n";
     
   }
 
    // Convert the remainder
    if (IsNotFullBlock) { 
        int NumOfRestItems = GlobalKnapsackData->OriginalNumberOfItems  - (BlockCount * Params->IntBlockSize()); 
        for (int BitID = 0; BitID < NumOfRestItems; BitID++) {
             char c =  ((LocalBestIndividual[BlockCount] & (1 << BitID)) == 0) ? '0' : '1';
             S << c;
             if (BitID % 8 ==7) S << " ";
        }
    }      
 
    
 return S.str();   
}// end of GetBestIndividualStr
//------------------------------------------------------------------------------



/*
 * Calculate global statistics
 * 
 * @param Population - calculate over this population
 * @param PrintBest  - print best solution
 */
void TGPU_Statistics::Calculate(TGPU_Population * Population, bool PrintBest){
    
    TParameters *Params = TParameters::GetInstance();
    
    // Calculate local statistics
    CalculateLocalStats(Population,  PrintBest);
    
    
      
    //-- collect statistics data --//
    MPI_Gather(HostStatData         ,sizeof(TStatDataToExchange),MPI_BYTE,
               ReceiveStatDataBuffer,sizeof(TStatDataToExchange),MPI_BYTE, 0, MPI_COMM_WORLD);
    
    
    if (PrintBest) {
    
        //-- Collect Individuals --//
        MPI_Gather(LocalBestIndividual     ,Params->ChromosomeSize(), MPI_UNSIGNED,
                   ReceiveIndividualBuffer ,Params->ChromosomeSize(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    
    // only master calculates the global statistics
    if (Params->IslandIdx() == 0) CalculateGlobalStatistics(PrintBest);
   
}// end of Calculate
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                       TGPU_Statistics Implementation                       //
//                              protected methods                             //
//----------------------------------------------------------------------------//

/*
 * Allocate GPU memory
 */
void TGPU_Statistics::AllocateCudaMemory(){
    
    //------------ Host data ---------------//
    
    // Allocate Host basic structure 
    cutilSafeCall( 
            cudaHostAlloc((void**)&HostStatData,  sizeof(TStatDataToExchange)
                           ,cudaHostAllocDefault )
            );		
            
    
    // Allocate Host basic structure 
    cutilSafeCall( 
            cudaHostAlloc((void**)&LocalBestIndividual,  sizeof(TGene) * TParameters::GetInstance()->ChromosomeSize()                        
                           ,cudaHostAllocDefault)
            );		
    
    
    //------------ Device data ---------------//
        
    // Allocate data structure 
    cutilSafeCall( 
           cudaMalloc((void**)&LocalDeviceStatData,  sizeof(TStatDataToExchange))
           );		
       
    
    
}// end of AllocateMemory
//------------------------------------------------------------------------------

/*
 * Free GPU memory
 */
void TGPU_Statistics::FreeCudaMemory(){
       
    
    //-- Free CPU Best individual --//
    cutilSafeCall( 
            cudaFreeHost(LocalBestIndividual)
            );
    
    //-- Free structure --//
    cutilSafeCall( 
            cudaFreeHost(HostStatData)
            );
            
    
    //-- free whole structure --//
    cutilSafeCall( 
           cudaFree(LocalDeviceStatData)
           );  
    
}// end of FreeMemory
//------------------------------------------------------------------------------
    
    



/*
 * Initialize statistics before computation
 * 
 */
void TGPU_Statistics::InitStatistics(){
    
    
    HostStatData->MaxFitness  = TFitness(0);
    HostStatData->MinFitness  = TFitness(UINT_MAX);
    HostStatData->SumFitness  = 0.0f;
    HostStatData->Sum2Fitness = 0.0f;  
    HostStatData->IndexBest   = 0;        
                
    //-- Copy 4 statistics values --//
    cutilSafeCall( 
         cudaMemcpy(LocalDeviceStatData, HostStatData, sizeof(TStatDataToExchange), 
                    cudaMemcpyHostToDevice)
         );    
    
    
    
}// end of InitStatistics
//------------------------------------------------------------------------------


/*
 * Calculate local statistics and download them to the host
 * 
 * @param Population - calculate over this population
 * @param PrintBest  - print best solution
 */    
void TGPU_Statistics::CalculateLocalStats(TGPU_Population * Population, bool PrintBest){
    
    TGPU_Lock GPU_Lock;   
    
    // Initialize statistics
    InitStatistics();
    
    // Run the CUDA kernel to calculate statistics
    CalculateStatistics
            <<<TParameters::GetInstance()->GetGPU_SM_Count()*2, BLOCK_SIZE >>>
            (LocalDeviceStatData, Population->DeviceData, GPU_Lock);
    
    
    // Copy data down to host
    CopyOut(Population, PrintBest);
        
}// end of Calculate
//------------------------------------------------------------------------------

    
/*
 * Copy data from GPU Statistics structure to CPU  
 * 
 * @param Population - calculate over this population
 * @param PrintBest  - print best solution
 * 
 */      
void TGPU_Statistics::CopyOut(TGPU_Population * Population, bool PrintBest){
    
    TParameters * Params = TParameters::GetInstance(); 
    
    //-- Copy 4 statistics values --//
    cutilSafeCall( 
         cudaMemcpy(HostStatData, LocalDeviceStatData, sizeof(TStatDataToExchange), 
                    cudaMemcpyDeviceToHost)
         );    
    
    //-- Copy of chromosome --//
    if (PrintBest){
    
        Population->CopyOutIndividual(LocalBestIndividual, HostStatData->IndexBest);
    }
    
    
}// end of CopyOut
//------------------------------------------------------------------------------



/*
 * Summarize statistics Local statistics to global
 * 
 * @param PrintBest  - print best solution
 */
void TGPU_Statistics::CalculateGlobalStatistics(bool PrintBest){
 
 GlobalDerivedStat->IslandBestIdx = 0;
  
 HostStatData->MaxFitness  = ReceiveStatDataBuffer[0].MaxFitness;
 HostStatData->MinFitness  = ReceiveStatDataBuffer[0].MinFitness;
 HostStatData->SumFitness  = ReceiveStatDataBuffer[0].SumFitness;
 HostStatData->Sum2Fitness = ReceiveStatDataBuffer[0].Sum2Fitness;
 
 TParameters * Params = TParameters::GetInstance();
 
  // Numeric statistics 
  for (int i = 1 ;  i < Params->IslandCount(); i++){      
      

      if (HostStatData->MaxFitness < ReceiveStatDataBuffer[i].MaxFitness) {
          HostStatData->MaxFitness = ReceiveStatDataBuffer[i].MaxFitness;
          GlobalDerivedStat->IslandBestIdx = i;
      }
      
      if (HostStatData->MinFitness > ReceiveStatDataBuffer[i].MinFitness) {
          HostStatData->MinFitness = ReceiveStatDataBuffer[i].MinFitness;
      }
      
      HostStatData->SumFitness  +=  ReceiveStatDataBuffer[i].SumFitness;
      HostStatData->Sum2Fitness +=  ReceiveStatDataBuffer[i].Sum2Fitness;
      

  }
    
 
 GlobalDerivedStat->AvgFitness = HostStatData->SumFitness / (Params->PopulationSize() * Params->IslandCount());
 GlobalDerivedStat->Divergence = sqrtf(fabsf( (HostStatData->Sum2Fitness / (Params->PopulationSize() * Params->IslandCount()) -
                                         GlobalDerivedStat->AvgFitness * GlobalDerivedStat->AvgFitness))
         );
    
    
}// end of CalculateGlobalStatistics
//------------------------------------------------------------------------------