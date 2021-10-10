#include <iostream>
#include "test.h"
#include "fasttime.h"

double test3(double *__restrict a, int N)
{
  __builtin_assume(N == 1024);
  a = (double *)__builtin_assume_aligned(a, 16);

  int count = 50;
  double elapsedf = 0;
  double b = 0;
  for (int t = 0; t < count; t++)
  {
    b = 0;

    fasttime_t time1 = gettime();
    for (int i = 0; i < I; i++)
    {
      for (int j = 0; j < N; j++)
      {
        b += a[j];
      }
    }
    fasttime_t time2 = gettime();

    elapsedf += tdiff(time1, time2);
  }
  elapsedf /= count;
  std::cout << "Elapsed execution time of the loop in test3():\n"
            << elapsedf << "sec (N: " << N << ", I: " << I << ")\n";

  return b;
}