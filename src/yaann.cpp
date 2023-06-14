#include "yaann.h"

#include <omp.h>

AI::YAAnn::YAAnn(
        int inputLayerSize,
        int hiddenLayersNumber,
        int hiddenLayerSize,
        int outputLayerSize,
        bool standardBiasedNeurons,
        AI::YAAnn::rnnType_t rnnType)
    :
    mHiddenLayersNumber(0),
    mHiddenLayers(NULL),
    mLayerConnSize(0),
    mLayerConn(NULL)
{
    buildModel(inputLayerSize, hiddenLayersNumber, hiddenLayerSize, outputLayerSize,
            standardBiasedNeurons, rnnType);
}

void AI::YAAnn::destroyModel()
{
    delete [] mLayerConn;
    delete [] mHiddenLayers;
    mLayerConnSize = 0;
    mHiddenLayersNumber = 0;
    mHiddenLayers = NULL;
    mLayerConn = NULL;
}

void AI::YAAnn::buildModel(int inputLayerSize, int hiddenLayersNumber, int hiddenLayerSize, int outputLayerSize, bool standardBiasedNeurons, AI::YAAnn::rnnType_t rnnType)
{
    if (hiddenLayersNumber > 0) {
        mLayerConnSize = hiddenLayersNumber+1;
    } else if (hiddenLayersNumber == 0) {
        mLayerConnSize = 1;
    }
    if (rnnType != YAAnn::rnnType_t::NO_RNN_TYPE) {
        mLayerConnSize+=2;
    }
    mLayerConn = new YAAnnLayerConnect[mLayerConnSize];
    if (standardBiasedNeurons) {
        inputLayerSize++;
        hiddenLayerSize++;
    }
    mInputLayer.buildLayer(YAAnnLayer::layerType_t::INPUT, inputLayerSize);
    if (standardBiasedNeurons) {
       mInputLayer[inputLayerSize - 1].setIsBiased(true);
       mInputLayer[inputLayerSize - 1].setOutput(1.0);
    }
    mOutputLayer.buildLayer(YAAnnLayer::layerType_t::OUTPUT, outputLayerSize);
    mHiddenLayersNumber = hiddenLayersNumber;
    if (hiddenLayersNumber > 0) {
        mHiddenLayers = new YAAnnLayer[hiddenLayersNumber];
        for (int i = 0; i < hiddenLayersNumber; ++i) {
            mHiddenLayers[i].buildLayer(YAAnnLayer::layerType_t::HIDDEN, hiddenLayerSize);
            if (standardBiasedNeurons) {
                mHiddenLayers[i][hiddenLayerSize - 1].setIsBiased(true);
                mHiddenLayers[i][hiddenLayerSize - 1].setOutput(1.0);
            }
            if (i == 0) {
                mLayerConn[0].connect(&mInputLayer, &mHiddenLayers[0]);
            } else {
                mLayerConn[i].connect(&mHiddenLayers[i-1], &mHiddenLayers[i]);
            }
        }
        if (rnnType != YAAnn::rnnType_t::NO_RNN_TYPE)
            mLayerConn[mLayerConnSize-3].connect(&mHiddenLayers[mHiddenLayersNumber-1],
                    &mOutputLayer);
        else
            mLayerConn[mLayerConnSize-1].connect(&mHiddenLayers[mHiddenLayersNumber-1],
                    &mOutputLayer);
    } else {
        mLayerConn[0].connect(&mInputLayer, &mOutputLayer);
    }

    switch (rnnType) {
    case ELMAN_RNN_TYPE:
        {
            if (mHiddenLayersNumber > 0) {
                mContextLayer.buildLayer(YAAnnLayer::layerType_t::CONTEXT, mHiddenLayers[0].neuronsNumber());
                mLayerConn[mLayerConnSize-2].connect(&mHiddenLayers[mHiddenLayersNumber-1], &mContextLayer);
                mLayerConn[mLayerConnSize-1].connect(&mContextLayer, &mHiddenLayers[0]);
            } else {
                mContextLayer.buildLayer(YAAnnLayer::layerType_t::CONTEXT, mInputLayer.neuronsNumber());
                mLayerConn[mLayerConnSize-2].connect(&mInputLayer, &mContextLayer);
                mLayerConn[mLayerConnSize-1].connect(&mContextLayer, &mInputLayer);
            }
        }
        break;
    case JORDAN_RNN_TYPE:
        {
            mLayerConn[mLayerConnSize-2].connect(&mOutputLayer, &mContextLayer);
            mContextLayer.buildLayer(YAAnnLayer::layerType_t::CONTEXT, mOutputLayer.neuronsNumber());
            if (mHiddenLayersNumber > 0) {
                mLayerConn[mLayerConnSize-1].connect(&mContextLayer, &mHiddenLayers[0]);
            } else {
                mLayerConn[mLayerConnSize-1].connect(&mContextLayer, &mInputLayer);
            }
        }
        break;
    default:
        break;
    }
    for (int i = 0; i < mLayerConnSize; ++i) {
        mLayerConn[i].randWeights();
    }
}

AI::YAAnn::~YAAnn()
{
    destroyModel();
}

