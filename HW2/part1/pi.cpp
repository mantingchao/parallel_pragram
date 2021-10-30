#include <stdio.h>
#include <time.h>

static long num_steps = 1e9;

int main()
{

    double x, pi, sum = 0.0;
    double step = 1.0 / num_steps;
    for (int i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    printf("%.10lf\n", pi);
    printf("runtime: %f  sec \n", (double)clock() / CLOCKS_PER_SEC);
}