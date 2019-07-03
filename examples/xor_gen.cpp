#include <iostream>
#include <iomanip>
#include <random>

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cout << "Usage: xor_gen N\n";
    return 0;
  }
  std::random_device rd;
  std::uniform_real_distribution<double> d(0., 1.);
  int N = atoi(argv[1]);
  for (int i = 0; i < N; ++i) {
    double inputs[2];
    double output;
    inputs[0] = d(rd);
    inputs[1] = d(rd);
    output = d(rd);
    std::cout << std::fixed << std::setprecision(20)
	      << inputs[0] << " " << inputs[1] << "\n"
	      << output << "\n";
  }
}
