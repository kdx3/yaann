#include "yaann.h"

AI::YAAnnLayerConnect::YAAnnLayerConnect(
        AI::YAAnnLayer* srcLayer,
        AI::YAAnnLayer* dstLayer,
        AI::YAAnnActFun* actFun,
        AI::YAAnnRand* rand):
    mSrcLayer(srcLayer),
    mDstLayer(dstLayer),
    mActFun(actFun),
    mRand(rand),
    mAxons(NULL),
    mAxonsNumber(0)
{
    connect(srcLayer, dstLayer);
}

AI::YAAnnLayerConnect::~YAAnnLayerConnect()
{
    delete [] mAxons;
    delete mRand;
    delete mActFun;
}

int AI::YAAnnLayerConnect::axonsNumber() const
{
    return mAxonsNumber;
}

const AI::YAAnnAxon& AI::YAAnnLayerConnect::operator[](int id) const
{
    return mAxons[id];
}

AI::YAAnnAxon& AI::YAAnnLayerConnect::operator[](int id)
{
    return mAxons[id];
}

AI::YAAnnActFun* AI::YAAnnLayerConnect::getActFun() const
{
    return mActFun;
}

void AI::YAAnnLayerConnect::setActFun(AI::YAAnnActFun* actFun)
{
    delete mActFun;
    mActFun = actFun;
}

AI::YAAnnRand* AI::YAAnnLayerConnect::getRand() const
{
    return mRand;
}

void AI::YAAnnLayerConnect::setRand(AI::YAAnnRand* rand)
{
    delete mRand;
    mRand = rand;
}

void AI::YAAnnLayerConnect::connect(AI::YAAnnLayer* srcLayer, AI::YAAnnLayer* dstLayer)
{
    if (!srcLayer || !dstLayer)
        return;
    delete [] mAxons;
    mAxonsNumber = srcLayer->neuronsNumber() * dstLayer->neuronsNumber();
    mAxons = new YAAnnAxon[mAxonsNumber];
}

void AI::YAAnnLayerConnect::forward(bool delta)
{
    if (!mSrcLayer || !mDstLayer || !mActFun)
        return;
    {
        for (int i = 0; i < mDstLayer->neuronsNumber(); ++i) {
            if (!(*mDstLayer)[i].isBiased()) {
                double sum = 0.;
                for (int j = 0; j < mSrcLayer->neuronsNumber(); ++j) {
                    if ((*mSrcLayer)[j].isBiased()) {
                        sum += (*mSrcLayer)[j].getOutput();
                    } else {
                        sum += (*mSrcLayer)[j].getOutput() * mAxons[j+i*mSrcLayer->neuronsNumber()].getWeight();
                    }
                }
                (*mDstLayer)[i].setInput(sum);
                (*mDstLayer)[i].setOutput(mActFun->f(sum));
                if (mDstLayer->getLayerType() == YAAnnLayer::OUTPUT) {
                    (*mDstLayer)[i].setError((*mDstLayer)[i].getOutput() - (*mDstLayer)[i].getIdealOutput());
                    if (delta)
                        (*mDstLayer)[i].setDelta(-(*mDstLayer)[i].getError() * mActFun->fPrime((*mDstLayer)[i].getInput()));
                }
            }
        }
    }
}

void AI::YAAnnLayerConnect::backward()
{
    if (!mSrcLayer || !mDstLayer || !mActFun || mSrcLayer->getLayerType() == YAAnnLayer::INPUT)
        return;
    {
        for (int i = 0; i < mSrcLayer->neuronsNumber(); ++i) {
            double delta = 0.;
            for (int j = 0; j < mDstLayer->neuronsNumber(); ++j) {
                if (!(*mDstLayer)[j].isBiased()) {
                    delta += (*mDstLayer)[j].getDelta() * mAxons[j*mSrcLayer->neuronsNumber() + i].getWeight();
                }
            }
            (*mSrcLayer)[i].setDelta(delta*mActFun->fPrime((*mSrcLayer)[i].getInput()));
        }
    }
}

void AI::YAAnnLayerConnect::randWeights()
{
    if (mRand == NULL)
        return;
    double ni = .0;
    ni = sqrt(1.0 / mAxonsNumber);
    for (int i = 0; i < mAxonsNumber; ++i) {
      mAxons[i].setWeight(mRand->rand(.0, 1.) * ni);
    }
}

void AI::YAAnnLayerConnect::updateWeights(double epsilon, double alpha)
{
    for (int i = 0; i < mAxonsNumber; ++i) {
        mAxons[i].updateWeight(epsilon, alpha);
    }
}
