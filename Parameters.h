/**
 * @file        Parameters.h
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
 * @brief       Header file of the parameter class.
 *              This class maintains all the parameters of evolution.
 *
 * @date        30 March     2012, 00:00 (created)
 *              11 April     2022, 21:00 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- CUDA constants ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Number of threads per block,
constexpr int BLOCK_SIZE = 256;
/// Warp size,
constexpr int WARP_SIZE  = 32;
/// Number of chromosomes per block - to correct functionality CHR_PER_BLOCK < WARP_SIZE.
constexpr int CHR_PER_BLOCK = (BLOCK_SIZE / WARP_SIZE);

/**
 * @struct EvolutionParameters
 * @brief  Parameters of the evolutionary process.
 */
struct EvolutionParameters
{
  /// Population size - number of chromosomes in the population.
  int populationSize;
  /// Offspring population size - number of newly generated chromosomes.
  int offspringPopulationSize;

  /// Length of binary chromosome in integer granularity (32b).
  int chromosomeSize;
  /// Total number of generations to evolve.
  int numOfGenerations;

  /// Crossover probability per individual (float number).
  float crossoverPst;
  /// Mutation probability per gene (float number).
  float mutationPst;
  /// Crossover rate converted to int for faster comparison and random number generation.
  unsigned int crossoverUintBoundary;
  /// Mutation rate converted to int for faster comparison and random number generation.
  unsigned int mutationUintBoundary;

  /// Number of migrating individuals between islands.
  int emigrantCount;
  /// Migration interval (how often to migrate).
  int migrationInterval;
  /// Index of the island.
  int islandIdx;
  /// Number of independent islands.
  int islandCount;
  /// How often to print statistics.
  int statisticsInterval;

  /// Size of int block (32 bin genes).
  int intBlockSize;
};// end of EvolutionParameters
//----------------------------------------------------------------------------------------------------------------------

/**
 * @class Parameters
 * @brief Singleton class with Parameters maintaining them in CPU and GPU constant memory.
 */
class Parameters
{
  public:
    /// Get instance of the singleton class.
    static Parameters& getInstance();

    /// Prevent copy-construction.
    Parameters(const Parameters&) = delete;

    /// Prevent assignment.
    Parameters& operator=(const Parameters&) = delete;


    // Destructor
    virtual ~Parameters()
    {
      sInstanceFlag = false;
    };

    /**
     * Load parameters from the commandline.
     * @param [in] argc
     * @param [in] argv
     */
    void parseCommandline(int argc, char **argv);

    /// Store GA parameters in GPU constant memory.
    void copyToDevice();


    //--------------------------------------------------- Getters ----------------------------------------------------//

    /// Get number of chromosomes in the population
    int  getPopulationSize()                const { return mEvolutionParameters.populationSize; };
    /// Get number of newly generated chromosomes.
    int  getOffspringPopulationSize()       const { return mEvolutionParameters.offspringPopulationSize; };
    /// Get size of the chromosome (including padding)
    int  getChromosomeSize()                const { return mEvolutionParameters.chromosomeSize; };
    /// Set size of the chromosome
    void setChromosomeSize(unsigned int value)    { mEvolutionParameters.chromosomeSize = value; };
    /// Get number of generations to evolve
    int  getNumOfGenerations()              const { return mEvolutionParameters.numOfGenerations; };

    /// Get crossover probability for two individuals.
    float        getCrossoverPst()          const { return mEvolutionParameters.crossoverPst; };
    /// Get per gene mutation probability.
    float        getMutationPst()           const { return mEvolutionParameters.mutationPst; };
    /// Get crossover probability in scaled to uint.
    unsigned int getCrossoverUintBoundary() const { return mEvolutionParameters.crossoverUintBoundary; };
    /// Get mutation probability in scaled to uint.
    unsigned int getMutationUintBoundary()  const { return mEvolutionParameters.mutationUintBoundary; };


    /// Get number of migrating individuals.
    int  getEmigrantCount()                 const { return mEvolutionParameters.emigrantCount; };
    /// How often to migrate individuals.
    int  getMigrationInterval()             const { return mEvolutionParameters.migrationInterval; };
    /// What  my island's Id.
    int  getIslandIdx()                     const { return mEvolutionParameters.islandIdx;};
    /// Get the number of islands.
    int  getIslandCount()                   const { return mEvolutionParameters.islandCount; };


      /// Get how often to print statistics.
    int  getStatisticsInterval()            const { return mEvolutionParameters.statisticsInterval; };

    /// Get the integer block size.
    int  getIntBlockSize()                  const {return mEvolutionParameters.intBlockSize;};

    /// Get filename with global data.
    std::string getBenchmarkFileName()      const {return mGlobalDataFileName;};


    /// Get number of SM processors on the GPU.
    int  getNumberOfDeviceSMs()            const {return mNumberOfDeviceSM;};
    /// Set GPU to computation.
    void setDevice();

    /// Print best solution?
    bool getPrintBest()                    const {return mPrintBest;};

    /// print parameters to stdout.
    void printOutAllParameters();

  private:
    /// Singleton constructor
    Parameters();

    /// print error message end exit if parameters are wrong.
    void printUsageAndExit();

    /// Structure with evolutionary parameters (host copy).
    EvolutionParameters mEvolutionParameters;
    /// Name of the file with global knapsack data
    std::string         mGlobalDataFileName;

    /// Number of SM processors on the device.
    int  mNumberOfDeviceSM;
    /// Id of the device being used by this island.
    int  mDeviceIdx;
    /// Shall we print the best solution?
    bool mPrintBest;

    /// Singleton static flag.
    static bool sInstanceFlag;
    /// Singleton static instance.
    static Parameters* sSingletonInstance;
};// end of Parameters
//----------------------------------------------------------------------------------------------------------------------


#endif	/* PARAMETERS_H */
