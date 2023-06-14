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
    AI::YAAnn ann0(new AI::YAAnnRandNormDistr, new AI::YAAnnActFunSig, 8, 2, 4, 1, true);
    rep0->saveToFile(ann0, "ann0.dot");
}
