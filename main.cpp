/* 
 * File:        main.cpp 
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
 * Comments:    Efficient MPI island-based Multi-GPU implementation of the 
 *              Genetic Algorithm, solving the Knapsack problem.
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

#include "Evolution.h"
#include "Parameters.h"



using namespace std;

/*
 * The main function
 */
int main(int argc, char **argv)
{

    // Initialize MPI
    MPI_Init(&argc, &argv);    
    
    //Initialize Evolution
    TGPU_Evolution GPU_Evolution(argc,argv);

    // Start time
    double AlgorithmStartTime;
    MPI_Barrier(MPI_COMM_WORLD);
    AlgorithmStartTime = MPI_Wtime();
        
    //
    GPU_Evolution.Run();
        
    // Run evolution
    MPI_Barrier(MPI_COMM_WORLD);
    double AlgorithmStopTime = MPI_Wtime();    
    if (GPU_Evolution.IsMaster()) printf("Execution time: %0.3f s.\n",  AlgorithmStopTime - AlgorithmStartTime);        
        
    MPI_Finalize();
    return 0;
}
