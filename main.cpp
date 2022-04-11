/**
 * @file        main.cpp
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
 * @brief       Efficient MPI island-based Multi-GPU implementation of the
 *              Genetic Algorithm, solving the Knapsack problem.
 *
 * @date        08 June      2012, 00:00 (created)
 *              11 April     2022, 17:51 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2022 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */


#include <mpi.h>
#include <cstdio>

#include "Evolution.h"
#include "Parameters.h"

/**
 * The main function
 */
int main(int argc, char **argv)
{
  // Initialize MPI
  MPI_Init(&argc, &argv);

  //Initialize Evolution
  Evolution evolution(argc,argv);

  // Start time
  double algorithmStartTime;
  MPI_Barrier(MPI_COMM_WORLD);
  algorithmStartTime = MPI_Wtime();

  // Run evolution process.
  evolution.run();

  // Run evolution.
  MPI_Barrier(MPI_COMM_WORLD);

  // Print execution time.
  double algorithmStopTime = MPI_Wtime();
  if (evolution.isMaster())
  {
    printf("Execution time: %0.3f s.\n",  algorithmStopTime - algorithmStartTime);
  }

  // Finalize.
  MPI_Finalize();
  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
