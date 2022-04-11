/**
 * @file        CUDAKernels.h
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
 * @brief       Header file of the GA evolution CUDA kernel
 *              This class controls the evolution process on a single GPU
 *
 * @date        08 June      2012, 00:00 (created)
 *              11 April     2022, 21:03 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#ifndef CUDA_KERNELS_H
#define	CUDA_KERNELS_H


#include "Population.h"
#include "Statistics.h"
#include "GlobalKnapsackData.h"

/**
 * Check and report CUDA errors.
 * @param [in] sourceFileName   - Source file where the error happened.
 * @param [in] sourceLineNumber - Line where the error happened.
 */
void checkAndReportCudaError(const char* sourceFileName,
                             const int   sourceLineNumber);


/**
 * Generate first population
 * @param [in, out] populationData - What population to generate.
 * @param [in]      randomSeed     - Random seed for Random123 generator.
 */
__global__ void cudaGenerateFirstPopulation(PopulationData* populationData,
                                            unsigned int    randomSeed);

/**
 * Genetic Manipulation (Selection, Crossover, Mutation).
 *
 * @param [in]  parentsData   - Parent population.
 * @param [out] offspringData - Offspring population.
 * @param [in]  randomSeed    - Random seed.
 *
 */
__global__ void cudaGeneticManipulation(const PopulationData* parentsData,
                                        PopulationData*       offspringData,
                                        unsigned int          randomSeed);



/**
 * Replacement kernel.
 *
 * @param [in]  parentsData   - Parent population.
 * @param [out] offspringData - Offspring population.
 * @param [in]  randomSeed    - Random seed.
 */
__global__ void cudaReplacement(const PopulationData* parentsData,
                                PopulationData*       offspringData,
                                unsigned int          randomSeed);


/**
 * Calculate statistics kernel.
 *
 * @param [out] statisticsData  - Statistical data.
 * @param [in]  populationData  - Population data.
 *
 */
__global__ void cudaCalculateStatistics(StatisticsData*       statisticsData,
                                        const PopulationData* populationData);

/**
 * Select individuals to migration.
 * @param [in]  parentsData     - Parent population.
 * @param [out] emigrantsToSend - Selected emigrants.
 * @param [in]  randomSeed      - Random seed.
 */
__global__ void cudaSelectEmigrants(const PopulationData* parentsData,
                                    PopulationData*       emigrantsToSend,
                                    unsigned int          randomSeed);


/**
 * Accept emigrants into parent population.
 * @param [in, out] parentsData       - Parent population.
 * @param [in]      emigrantsToAccept - Received emigrants.
 * @param [in]      randomSeed        - Random seed.
 */
 __global__ void cudaAcceptEmigrants(PopulationData*       parentsData,
                                     const PopulationData* emigrantsToAccept,
                                     unsigned int          randomSeed);

/**
 * Calculate Knapsack fitness.
 *
 * Each warp working with 1 32b gene. Different warps different individuals.
 *
 * @param [in,out] populationData - Population to be evaluated.
 * @param [in]     globalData     - Global knapsack data.
 *
 */
__global__ void cudaCalculateKnapsackFintess(PopulationData*     populationData,
                                             const KnapsackData* globalData);

#endif	/* CUDA_KERNELS_H */

