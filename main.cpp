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
 *              03 March     2022, 11:09 (revised)
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

/*
 * The main function
 */
int main(int argc, char **argv)
{
  // Initialize MPI
  MPI_Init(&argc, &argv);

  //Initialize Evolution
  Evolution evolution(argc,argv);

  // Start time
  double AlgorithmStartTime;
  MPI_Barrier(MPI_COMM_WORLD);
  AlgorithmStartTime = MPI_Wtime();

  // Run evolutin process.
  evolution.run();

  // Run evolution
  MPI_Barrier(MPI_COMM_WORLD);

  double AlgorithmStopTime = MPI_Wtime();
  if (evolution.isMaster())
  {
    printf("Execution time: %0.3f s.\n",  AlgorithmStopTime - AlgorithmStartTime);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
