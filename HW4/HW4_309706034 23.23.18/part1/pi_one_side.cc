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

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int number_of_tosses = tosses / world_size;
    long long int number_in_circle = 0;
    long long int total = 0;
    float x = 0.0f, y = 0.0f;
    unsigned int seed = time(NULL) + world_rank;

    for (long long int toss = 0; toss < number_of_tosses; toss++)
    {
        x = rand_r(&seed) / ((float)RAND_MAX);
        y = rand_r(&seed) / ((float)RAND_MAX);
        if (x * x + y * y <= 1.0f)
            number_in_circle++;
    }

    if (world_rank == 0)
    {
        // Master
        MPI_Alloc_mem(sizeof(long long int), MPI_INFO_NULL, &total);
        total = number_in_circle;
        MPI_Win_create(&total, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&number_in_circle, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
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