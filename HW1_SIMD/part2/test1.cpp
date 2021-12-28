#include <iostream>
#include "test.h"
#include "fasttime.h"

void test1(float *__restrict a, float *__restrict b, float *__restrict c, int N)
{
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);

  int count = 50;
  double elapsedf = 0;
  for (int t = 0; t < count; t++)
  {
    fasttime_t time1 = gettime();

    for (int i = 0; i < I; i++)
    {
      for (int j = 0; j < N; j++)
      {
        c[j] = a[j] + b[j];
      }
    }
    fasttime_t time2 = gettime();

    elapsedf += tdiff(time1, time2);
  }

  elapsedf /= count;

  std::cout << "Elapsed execution time of the loop in test1():\n"
            << elapsedf << "sec (N: " << N << ", I: " << I << ")\n";
}