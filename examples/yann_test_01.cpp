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

    rep0->readMdlFromFile(ann0, argv[1]);
    double *oarr = (double*)malloc(784*sizeof(double));

/*
    int MAX_ITERS = 500;
    for (int i = 0; i < MAX_ITERS; ++i)
        ann0.iterTraining(784,784, arr, arr, 0.1, 0.001, rep0.get());
    rep0->saveMdlToFile(ann0, "ann_02.mdl");
*/
    ann0.setInputs(arr, 784);
    ann0.forwardProp();

    int n = 0;
    double mse = 0.;
    for (int i = 0; i < ann0.getOutputLayer()->neuronsNumber(); ++i) {
        if (!ann0.getOutputLayer()->getNeuron(i)->isBiased()) {
            oarr[i] = ann0.getOutputLayer()->getNeuron(i)->getOutput();
            mse += pow(oarr[i]-arr[i],2);
            ++n;
        }
    }
    mse /= n;

    printf("%f\n", mse);

    fp = fopen(argv[2], "wb");
    fwrite(oarr, sizeof(double), 784, fp);
    fclose(fp);
    free(oarr);
    free(arr);
}
