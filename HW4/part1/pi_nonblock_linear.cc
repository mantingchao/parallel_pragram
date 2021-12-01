#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int source;
    int dest;
    int tag = 0;

    long long int number_of_tosses = tosses / world_size;
    long long int number_in_circle = 0;
    long long int total = 0;
    float x = 0.0f, y = 0.0f;
    unsigned int seed = time(NULL) + world_rank;

    MPI_Request reqs[world_size - 1];
    MPI_Status status[world_size - 1];

    for (long long int toss = 0; toss < number_of_tosses; toss++)
    {
        x = rand_r(&seed) / ((float)RAND_MAX);
        y = rand_r(&seed) / ((float)RAND_MAX);
        if (x * x + y * y <= 1.0f)
            number_in_circle++;
    }

    if (world_rank > 0)
    {
        // TODO: MPI workers
        dest = 0;
        MPI_Isend(&number_in_circle, 1, MPI_LONG_LONG_INT, dest, tag, MPI_COMM_WORLD, &reqs[world_rank]);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        total = number_in_circle;
        for (source = 1; source < world_size; source++)
        {
            MPI_Irecv(&number_in_circle, 1, MPI_LONG_LONG_INT, source, tag, MPI_COMM_WORLD, &reqs[source - 1]);
            total += number_in_circle;
        }

        MPI_Waitall(world_size - 1, reqs, status);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * total / ((float)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
