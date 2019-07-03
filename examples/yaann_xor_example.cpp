#include <iostream>
#include <random>
#include <memory>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "yaann.h"
#include "yaann_addons.h"

int main(int argc, char* argv[])
{
  std::unique_ptr<AI::YAAnnReport> rep0 = std::make_unique<AI::YAAnnReportStdFile>();
  AI::YAAnn ann0(new AI::YAAnnRandNormDistr, new AI::YAAnnActFunTanh, 1, 1, 3, 1);
  ann0.saveToFile(rep0.get(), argv[1]);
  ann0.saveMdlToFile(rep0.get(), argv[2]);  
  ann0.saveToFile(rep0.get(), argv[3]);
  ann0.saveMdlToFile(rep0.get(), argv[4]);
}
