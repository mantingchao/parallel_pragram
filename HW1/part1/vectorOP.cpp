#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f); // set zero = (type: __pp_vec_float)[0., 0., 0., 0.]
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  float one = 1.0;
  float nine = 9.999999f;
  __pp_vec_float vec1, result;
  __pp_vec_int vec2, count;
  __pp_mask maskAll, expZeroMask, valZeroMask, overMask, countMask;

  __pp_vec_int allZeros = _pp_vset_int(0);
  __pp_vec_int allOnes = _pp_vset_int(1);
  __pp_vec_float oneFloat = _pp_vset_float(1.f);
  __pp_vec_float UBMask = _pp_vset_float(9.999999f);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = ((i + VECTOR_WIDTH) > N) ? _pp_init_ones(N - i) : _pp_init_ones();
    expZeroMask = _pp_init_ones(0);

    _pp_vload_float(vec1, values + i, maskAll);  // x = values[i];
    _pp_vload_int(vec2, exponents + i, maskAll); // y = exponents[i];

    _pp_veq_int(expZeroMask, vec2, allZeros, maskAll); // if (y == 0)
    _pp_vset_float(result, 1.f, expZeroMask);          // output[i] = 1.f;

    valZeroMask = _pp_mask_not(expZeroMask);
    valZeroMask = _pp_mask_and(maskAll, valZeroMask);
    _pp_vmove_float(result, vec1, valZeroMask); // float result = x;

    _pp_vsub_int(count, vec2, allOnes, valZeroMask); // int count = y - 1;
    _pp_vgt_int(countMask, count, allZeros, valZeroMask);
    // count = _pp_cntbits(countMask);

    while (_pp_cntbits(countMask))
    {
      // printf("ck");
      _pp_vmult_float(result, result, vec1, countMask); // result *= x;
      _pp_vsub_int(count, count, allOnes, valZeroMask); // count--;
      _pp_vgt_int(countMask, count, allZeros, valZeroMask);
      // count = _pp_cntbits(countMask);
    }
    printf("ck");
    _pp_vgt_float(overMask, result, UBMask, valZeroMask); // if (result > 9.999999f)
    _pp_vset_float(result, nine, overMask);               // result = 9.999999f;

    _pp_vstore_float(output + i, result, maskAll); // output[i] = result;
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float vec;
  __pp_mask maskAll;
  maskAll = _pp_init_ones();
  float sum = 0;
  float *output = new float[VECTOR_WIDTH];
  int size = VECTOR_WIDTH;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    size = VECTOR_WIDTH;
    _pp_vload_float(vec, values + i, maskAll); // sum += values[i];
    while (size > 1)
    {
      _pp_hadd_float(vec, vec);
      _pp_interleave_float(vec, vec);
      size /= 2;
    }
    _pp_vstore_float(output, vec, maskAll);
    sum += output[0];
  }

  return sum;
}
