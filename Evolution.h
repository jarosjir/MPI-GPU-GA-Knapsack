/**
 * @file        Evolution.h
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
 * @brief       Header file of the GA evolution
 *              This class controls the evolution process on multiple GPUs across many nodes.
 *
 * @date        08 June      2012, 00:00 (created)
 *              03 March     2022, 11:09 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "Parameters.h"

#include "Population.h"
#include "Statistics.h"
#include "GlobalKnapsackData.h"


/**
 * @class Evolution
 * @brief Class controlling the evolutionary process.
 */
class Evolution
{
  public:
    /// Class constructors
    Evolution() = delete;
    /**
     * Constructor commandline parameters.
     * @param [] argc - Argument counts
     * @param [] argv - Commandline argiments.
     */
    Evolution(int argc, char **argv);

    /// Copy constructor not allowed.
    Evolution(const Evolution&) = delete;
    /// Destructor.
    virtual ~Evolution();
    /// Assignment operator not allowed.
    Evolution& operator =(const Evolution&) = delete;

    /// Run evolution.
    void run();

    // Is this the master node?
    bool isMaster() const { return mParams.getIslandIdx() == 0; };


  protected:
    /// Initialize random seed.
    void         initRandomSeed();
    /// Get next random seed.
    unsigned int getRandomSeed()  { return mRandomSeed++; };

    /// Initialize evolution.
    void         initialize();
    /// Run evolution.
    void         runEvolutionCycle();
    /// Migrate.
    void         migrate();

    /// Parameters of evolution.
    Parameters&   mParams;
    /// Actual generation.
    int           mActGeneration;
    /// Random Seed.
    unsigned int  mRandomSeed;

    /// Master GA population.
    GPUPopulation*     mMasterPopulation;
    /// Population of offsprings.
    GPUPopulation*     mOffspringPopulation;

    /// Emigrants to send.
    GPUPopulation*     mDeviceEmigrantsToSend;
    /// Emigrants to receive.
    GPUPopulation*     mDeviceEmigrantsToReceive;

    /// Buffer for individuals to send.
    CPUPopulation*     mHostEmigrantsToSend;
    /// Buffer for individuals to receive.
    CPUPopulation*     mHostEmigrantsToReceive;

    /// Statistics over GA process.
    Statistics*        mStatistics;

    /// Global data of knapsack.
    GlobalKnapsackData mGlobalData;

    static constexpr int kMpiDataTag = 100;
};// end of Evolution
//----------------------------------------------------------------------------------------------------------------------

#endif	/* Evolution */
