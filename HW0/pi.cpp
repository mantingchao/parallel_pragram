#include <iostream>
using namespace std;

#define number_of_tosses 1000000

int main()
{
    int number_in_circle = 0;
    for (int toss = 0; toss < number_of_tosses; toss++)
    {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            number_in_circle++;
    }
    double pi_estimate = 4.0 * number_in_circle / number_of_tosses;
    cout << pi_estimate << endl;

    return 0;
}