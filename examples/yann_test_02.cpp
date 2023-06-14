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
    FILE *fp = fopen("1.bin", "rb");
    double *arr = (double*)malloc(784*sizeof(double));
    fread(arr, sizeof(double), 784, fp);
    fclose(fp);
    std::unique_ptr<AI::YAAnnReport> rep0 = std::make_unique<AI::YAAnnReportStdFile>();
    AI::YAAnn ann0(new AI::YAAnnRandNormDistr, new AI::YAAnnActFunSig, 784, 1, 256, 784, true);
    double *oarr = (double*)malloc(784*sizeof(double));
    int MAX_ITERS = 1500;
    for (int i = 0; i < MAX_ITERS; ++i)
        ann0.iterTraining(784,784, arr, arr, 0.1, 0.001, rep0.get());
    rep0->saveMdlToFile(ann0, argv[1]);
    free(oarr);
    free(arr);
}
