#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
bool top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, vertex_set *bu_frontier)
{
    bool change = false;
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < frontier->count; i++)
    {
        int node = frontier->vertices[i];
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];
        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {

            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
#pragma omp critical
                {
                    int index = new_frontier->count++;
                    new_frontier->vertices[index] = outgoing;
                    bu_frontier->vertices[outgoing] = 1;
                    change = true;
                }
            }
        }
    }
    return change;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set list3;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set_init(&list3, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
    vertex_set *bu_frontier = &list3;

    // initialize all nodes to NOT_VISITED
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    bool change = true;
    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        change = top_down_step(graph, frontier, new_frontier, sol->distances, bu_frontier);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

bool bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    bool change = false;
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < g->num_nodes; i++)
    {
        if (distances[i] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];
                if (frontier->vertices[incoming] == 1)
                {
                    distances[i] = distances[incoming] + 1;
                    new_frontier->vertices[i] = 1;
                    change = true;
                    break;
                }
            }
        }
    }
    return change;
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    vertex_set list1;
    vertex_set list2;
    vertex_set list3;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set_init(&list3, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;
    vertex_set *td_frontier = &list3;

// initialize all nodes to NOT_VISITED
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;
    bool change = true;
    while (change)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        change = bottom_up_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set list3;
    vertex_set list4;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set_init(&list3, graph->num_nodes);
    vertex_set_init(&list4, graph->num_nodes);

    vertex_set *td_frontier = &list1;
    vertex_set *td_new_frontier = &list2;
    vertex_set *bu_frontier = &list3;
    vertex_set *bu_new_frontier = &list4;

#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // top down
    td_frontier->vertices[td_frontier->count++] = ROOT_NODE_ID;
    // bottom up
    bu_frontier->vertices[ROOT_NODE_ID] = 1;

    sol->distances[ROOT_NODE_ID] = 0;
    bool change = true;
    bool flag = false;
    while (change)
    {
        if (!flag && (float)td_frontier->count < 100000)
        {
            // top down
            change = top_down_step(graph, td_frontier, td_new_frontier, sol->distances, bu_frontier);
        }
        else
        {
            // bottom up
            change = bottom_up_step(graph, bu_frontier, bu_new_frontier, sol->distances);
            flag = true;
            vertex_set *tmp2 = bu_frontier;
            bu_frontier = bu_new_frontier;
            bu_new_frontier = tmp2;
        }
        vertex_set *tmp1 = td_frontier;
        td_frontier = td_new_frontier;
        td_new_frontier = tmp1;
    }
}
