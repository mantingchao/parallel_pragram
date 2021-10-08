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
  __pp_vec_float vec1, result;
  __pp_vec_int vec2, count;
  __pp_mask maskAll, expZeroMask, valMask, overMask, countMask;

  __pp_vec_int allZeros = _pp_vset_int(0);
  __pp_vec_int allOnes = _pp_vset_int(1);
  __pp_vec_float UBMask = _pp_vset_float(9.999999f);
  __pp_vec_float oneFloat = _pp_vset_float(1.f);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones(N - i);

    // load data
    _pp_vload_float(vec1, values + i, maskAll);  // x = values[i];
    _pp_vload_int(vec2, exponents + i, maskAll); // y = exponents[i];

    // find exponent = 0 in the vector
    _pp_veq_int(expZeroMask, vec2, allZeros, maskAll); // if (y == 0)
    // _pp_vset_float(result, 1.0, expZeroMask);          // output[i] = 1.f;
    _pp_vstore_float(output + i, oneFloat, expZeroMask);

    // else (exponent != 0)
    valMask = _pp_mask_not(expZeroMask);
    valMask = _pp_mask_and(maskAll, valMask);
    _pp_vmove_float(result, vec1, valMask); // float result = x;

    // update valMask
    _pp_vsub_int(count, vec2, allOnes, valMask); // int count = y - 1;
    _pp_vgt_int(countMask, count, allZeros, valMask);

    // compute count
    while (_pp_cntbits(countMask) > 0)
    {
      _pp_vmult_float(result, result, vec1, countMask); // result *= x;
      _pp_vsub_int(count, count, allOnes, valMask);     // count--;
      _pp_vgt_int(countMask, count, allZeros, valMask);
    }

    // if (result > 9.999999f)
    _pp_vgt_float(overMask, result, UBMask, valMask);
    _pp_vset_float(result, 9.999999f, overMask); // result = 9.999999f;

    _pp_vstore_float(output + i, result, valMask); // output[i] = result;
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
  float sum = 0.f;
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
