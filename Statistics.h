/**
 * @file        Statistics.h
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
 * @brief       Header file of the GA statistics
 *              This class maintains and collects GA statistics
 *
 * @date        08 June      2012, 00:00 (created)
 *              28 March     2022, 11:02 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#ifndef STATISTICS_H
#define STATISTICS_H


#include "Parameters.h"
#include "Population.h"
#include "GlobalKnapsackData.h"


/**
 * @struct StatisticsData
 * @brief  Statistics Structure
 */
struct StatisticsData
{
  /// Minimum fitness value in population.
  Fitness minFitness   = Fitness(0);
  /// Maximum fitness value in population.
  Fitness maxFitness   = Fitness(0);

  /// Sum of fitness to calculate average.
  float    sumFitness  = 0.f;
  /// Sum of fitness squared to calculate divergence.
  float    sum2Fitness = 0.f;
  /// Which individual is the best.
  int      indexBest   = 0;
};//  StatisticsData
//----------------------------------------------------------------------------------------------------------------------

/**
 * @struct DerivedStats
 * @brief  Derived statistics.
 */
struct DerivedStats
{
  /// Average fitness value.
  float avgFitness    = 0.f;
  /// Divergence
  float divergence    = 0.f;
  // Index of the best island.
  int   bestIslandIdx = 0.f;
};// DerivedStats
//----------------------------------------------------------------------------------------------------------------------

/**
 * @class Statistics
 * @brief statistics class.
 */
class Statistics
{
  public:
    /// Constructor.
    Statistics();
    /// Copy constructor not allowed.
    Statistics(const Statistics&) = delete;
    /// Destructor.
    virtual ~Statistics();
    /// Assignment operator not allowed.
    Statistics& operator=(const Statistics&) = delete;


    /**
     * Calculate statistics
     * @param [in] population - Population to calculate statistics of.
     * @param [in] printBest  - do we need to download the best individual to print.
     */
    void   calculate(GPUPopulation* population,
                     bool           printBest);

    /**
     * Get best individual in text form.
     * @param  [in] globalKnapsackData  - Global knapsack data
     * @return String representation of the best individual.
     * @note only on root rank can call this routine.
     */
    std::string getBestIndividualStr(KnapsackData* globalKnapsackData) const;

    //-------------------------------------------- Getters for root rank ---------------------------------------------//
    /// Get minimum fitness.
    Fitness getMinFitness()     const {return mHostStatData->minFitness;};
    /// Get maximum fitness.
    Fitness getMaxFitness()     const {return mHostStatData->maxFitness;};
    /// Get average fitness.
    float    getAvgFitness()    const {return mGlobalDerivedStat->avgFitness;};
    /// Get divergence.
    float    getDivergence()    const {return mGlobalDerivedStat->divergence;};
    /// Get best Island id.
    int      getBestIslandIdx() const {return mGlobalDerivedStat->bestIslandIdx;};

  protected:
    /// Allocate memory on device side.
    void allocateCudaMemory();
    /// Free memory on device side.
    void freeCudaMemory();

    /// Initialize statistics structure.
    void initStatistics();

    /**
     * Calculate local statistics.
     * @param [in,out] population - Population to calculate the statistics on.
     * @param [in]     printBest  - Shall I print best solution?
     */
    void calculateLocalStats(GPUPopulation* population,
                             bool           printBest);

    /**
     * Calculate Global statistics.
     * @param [in] printBest - Shall I print best solution?
     */
    void calculateGlobalStatistics(bool printBest);

    /**
     * Copy statistics data from GPU memory down to host
     * @param [in] population - Population to take the best solution from.
     * @param [in] printBest  - Shall I print best solution?
     */
    void copyFromDevice(GPUPopulation* population,
                        bool           printBest);

    /// Statistics on the local device.
    StatisticsData* mLocalDeviceStatData;
    /// Copy of local statistics on the host side.
    StatisticsData* mHostStatData;

    /// Received statistics from all nodes, root rank only.
    StatisticsData* mReceiveStatDataBuffer;
    /// Global derived statistics, root rank only.
    DerivedStats*   mGlobalDerivedStat;

    // Host copy of the best solution / global best.
    Gene*           mLocalBestIndividual;
    // All the best solutions from all nodes, root rank only.
    Gene*           mReceiveIndividualBuffer;

  private:
};// end of Statistics
//----------------------------------------------------------------------------------------------------------------------


#endif	/* STATISTICS_H */

