/* 
 * File:        Statistics.h
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
 * Comments:    Header file of the GA statistics
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

#ifndef STATISTICS_H
#define STATISTICS_H


#include "Parameters.h"
#include "Population.h"
#include "GlobalKnapsackData.h"




/*
 * Statistics Structure
 */
struct TStatDataToExchange{
  TFitness MinFitness;  // Minimum fitness value in population
  TFitness MaxFitness;  // Maximum fitness value in population      
  
  float    SumFitness;  // Sum of fitness values over an island
  float    Sum2Fitness; // Sum of fitness squares over an island                
  
  int      IndexBest;  // Index of the best solution
  
};//  TStatDataToExchange
//----------------------------------------------------------------------------

/*
 * Derived Statistics
 */
struct TDerivedStats{
    float   AvgFitness;     // Average individual
    float   Divergence;     // Divergence  
    int     IslandBestIdx;  // Index of the best island
};//TDerivedStats
//------------------------------------------------------------------------------



/*
 * GPU statistics class
 * 
 */
class TGPU_Statistics {
public:
    
    //-- Only master can read these values --//
    TFitness GetMaxFitness()           const {return HostStatData->MaxFitness;};
    TFitness GetMinFitness()           const {return HostStatData->MinFitness;};
    float    GetAvgFitness()           const {return GlobalDerivedStat->AvgFitness;};    
    float    GetDivergence()           const {return GlobalDerivedStat->Divergence;};
    int      GetBestIslandIdx()        const {return GlobalDerivedStat->IslandBestIdx;};
        
            
    // calculate global statistics
    void   Calculate              (TGPU_Population * Population, bool PrintBest);    
    // print best individual -- only on MASTER node --//
    string GetBestIndividualStr   (KnapsackData * GlobalKnapsackData);
    
    TGPU_Statistics();
    virtual ~TGPU_Statistics();

protected:
    
    TStatDataToExchange * LocalDeviceStatData;          // stat data on device
    TStatDataToExchange * HostStatData;                 // copy of stat data on host
    
    TStatDataToExchange * ReceiveStatDataBuffer;        // stat data from all nodex
    
    TDerivedStats       * GlobalDerivedStat;            // derived data from all nodes
        
    TGene               * LocalBestIndividual;          // host copy of the best solution / Global best
    TGene               * ReceiveIndividualBuffer;       // all the best solutions from all nodes
    
    // Memory Allocation
    void AllocateCudaMemory();
    void FreeCudaMemory();
    
    // Initialize statistics structure    
    void InitStatistics();
    // Calculate local statistics
    void CalculateLocalStats (TGPU_Population * Population, bool PrintBest);    
    
    // Calculate Global statistics
    void CalculateGlobalStatistics(bool PrintBest);
    
    // Copy statistics data from GPU memory down to host
    void CopyOut(TGPU_Population * Population, bool PrintBest); 
        
private:
         
   
    TGPU_Statistics(const TGPU_Population& orig);

};// end of TGPU_Statistics
//------------------------------------------------------------------------------


#endif	/* GPU_STATISTICS_H */

