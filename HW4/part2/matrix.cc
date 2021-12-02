#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    int *ptr;
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0)
    {
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
    }
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *a_mat_ptr = (int *)malloc(sizeof(int) * (*n_ptr) * (*m_ptr));
    *b_mat_ptr = (int *)malloc(sizeof(int) * (*m_ptr) * (*l_ptr));

    if (world_rank == 0)
    {

        for (int i = 0; i < *n_ptr; i++)
        {
            for (int j = 0; j < *m_ptr; j++)
            {
                ptr = *a_mat_ptr + i * (*m_ptr) + j;
                scanf("%d", ptr);
            }
        }

        for (int i = 0; i < *m_ptr; i++)
        {
            for (int j = 0; j < *l_ptr; j++)
            {
                ptr = *b_mat_ptr + i * (*l_ptr) + j;
                scanf("%d", ptr);
            }
        }
    }
    MPI_Bcast(*a_mat_ptr, (*n_ptr) * (*m_ptr), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0, MPI_COMM_WORLD);
}

// Just matrix multiplication (your should output the result in this function)
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int *c_mat, *ans;
    c_mat = (int *)calloc(n * l, sizeof(int));
    ans = (int *)calloc(n * l, sizeof(int));

    int start = (n * l / world_size) * world_rank;
    int end;
    if (world_rank == world_size - 1)
    {
        end = n * l;
    }
    else
    {
        end = start + (n * l / world_size);
    }

    int b_idx, c_idx, sum;
    int a_idx = start;
    for (int i = start; i < end; i++)
    {
        int c_y = i % l;
        int c_x = i / l;
        int sum = 0;
        c_idx = i * l;
        for (int j = 0; j < m; j++)
        {
            sum += a_mat[c_x * m + j] * b_mat[j * l + c_y];
        }
        c_mat[i] = sum;
    }

    MPI_Reduce(c_mat, ans, n * l, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            c_idx = i * l;
            for (int j = 0; j < l; j++)
            {
                printf("%d ", ans[c_idx]);
                c_idx++;
            }
            printf("\n");
        }
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0)
    {
        free(a_mat);
        free(b_mat);
    }
}