/*
 * File:        Evolution.h
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
 * Comments:    Header file of the GA evolution
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
 * Created on 08 June     2012, 00:00 PM
 * Revised on 24 February 2022, 16:27 PM
 */
#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "Parameters.h"

#include "Population.h"
#include "Statistics.h"
#include "GlobalKnapsackData.h"


/*
 * GPU evolution process
 *
 */
class TGPU_Evolution{
public:
             // Class constructors
             TGPU_Evolution(int argc, char **argv);
             TGPU_Evolution() : Params(Parameters::getInstance()) {};
    virtual ~TGPU_Evolution();

    // Run evolution
    void     Run();

    // Is this the master node?
    bool     IsMaster() {return Params.getIslandIdx() == 0;};


protected:
    Parameters&   Params;               // Parameters of evolution
    int           FActGeneration;       // Actual generation
    unsigned int  FRandomSeed;          // Random Seed



    TGPU_Population*  MasterPopulation;        // Master GA population
    TGPU_Population*  OffspringPopulation;     // Population of offsprings

    TGPU_Population*  GPU_EmigrantsToSend;     // Emigrants to send
    TGPU_Population*  GPU_EmigrantsToReceive;  // Emigrants to receive

    TCPU_Population*  CPU_EmigrantsToSend;     // Buffer for individuals to send
    TCPU_Population*  CPU_EmigrantsToReceive;  // Buffer for individuals to receive

    Statistics*  GPUStatistics;           // Statistics over GA process

    GlobalKnapsackData GlobalData;            // Global data of knapsack


    void         InitSeed();
    unsigned int GetSeed()  {return FRandomSeed++;};

    // Initialize evolution
    void         Initialize();
    // Run evolution
    void         RunEvolutionCycle();

    // Migrate
    void         Migrate();

    TGPU_Evolution(const TGPU_Evolution& orig);
};

#endif	/* TGPU_EVOLUTION_H */

