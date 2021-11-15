#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double no_out = 0.0;
#pragma omp parallel for reduction(+ \
                                   : no_out)
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
    if (outgoing_size(g, i) == 0)
    {
      no_out += solution[i];
    }
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:
  */

  // initialization: see example code above
  bool converged = false;
  double *score_new = (double *)calloc(numNodes, sizeof(double));
  // score_old[vi] = 1 / numNodes;
  double sum_v_out, global_diff = 0.0;
  while (!converged)
  {
#pragma omp parallel for private(sum_v_out)
    for (int i = 0; i < numNodes; i++)
    {
      // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
      score_new[i] = 0.0;
      sum_v_out = 0.0;
      // num_out = 0.0;
      const Vertex *start = incoming_begin(g, i);
      const Vertex *end = incoming_end(g, i);
      for (const Vertex *vi = start; vi != end; vi++)
      {
        // num_out = outgoing_size(g, *vi);
        sum_v_out += solution[*vi] / outgoing_size(g, *vi);
      }
      score_new[i] = (damping * sum_v_out) + (1.0 - damping) / numNodes;
      score_new[i] += damping * no_out / numNodes;
    }

    // compute score_new[vi] for all nodes vi:
    // score_new[vi] = sum over all nodes vj reachable from incoming edges{score_old[vj] / number of edges leaving vj}

    // compute how much per-node scores have changed
    // quit once algorithm has converged
    global_diff = 0.0;
    no_out = 0.0;
#pragma omp parallel for reduction(+ \
                                   : global_diff, no_out)
    for (int i = 0; i < numNodes; i++)
    {
      global_diff += fabs(score_new[i] - solution[i]);
      solution[i] = score_new[i];
      if (outgoing_size(g, i) == 0)
      {
        no_out += score_new[i];
      }
    }

    converged = (global_diff < convergence);
  }
}
