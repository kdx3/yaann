#include "yaann.h"

AI::YAAnnLayer::YAAnnLayer(
			   layerType_t layerType,		   
			   int neuronsNumber,
			   YAAnnLayer* nextLayer,
			   YAAnnLayer* prevLayer,
			   YAAnnRand* rand)
  : mLayerType(layerType),
    mNeuronsNumber(neuronsNumber),
    mNeurons(NULL),
    mNextLayer(nextLayer),
    mPrevLayer(prevLayer),
    mRand(rand)
{
  if (neuronsNumber > 0) {
    mNeurons = new YAAnnNeuron[neuronsNumber];
  }
}

AI::YAAnnLayer::~YAAnnLayer()
{
  delete [] mNeurons;
}

AI::YAAnnLayer::layerType_t AI::YAAnnLayer::getLayerType() const
{
  return mLayerType;
}

int AI::YAAnnLayer::neuronsNumber() const
{
  return mNeuronsNumber;
}

int AI::YAAnnLayer::biasedNeuronsNumber() const
{
  //TODO: don't count increase during layer modify
  int ret = 0;
  for (int i = 0; i < neuronsNumber(); ++i) {
    if (getNeuron(i)->isBiased())
      ++ret;
  }
  return ret;
}

AI::YAAnnLayer* AI::YAAnnLayer::getNextLayer() const
{
  return mNextLayer;
}

AI::YAAnnLayer* AI::YAAnnLayer::getPrevLayer() const
{
  return mPrevLayer;
}

AI::YAAnnNeuron* AI::YAAnnLayer::getNeuron(int index) const
{
  if (index >= 0 && index < mNeuronsNumber)
    return &mNeurons[index];
  return NULL;
}

AI::YAAnnRand* AI::YAAnnLayer::getRand() const
{
  return mRand;
}

void AI::YAAnnLayer::setLayerType(layerType_t layerType)
{
  mLayerType = layerType;
}

void AI::YAAnnLayer::setNextLayer(YAAnnLayer* nextLayer)
{
  mNextLayer = nextLayer;
}

void AI::YAAnnLayer::setPrevLayer(YAAnnLayer* prevLayer)
{
  mPrevLayer = prevLayer;
}

void AI::YAAnnLayer::setRand(YAAnnRand* rand)
{
  mRand = rand;
}

void AI::YAAnnLayer::connectTo(YAAnnLayer* destLayer)
{
  if (destLayer != NULL && mNextLayer == destLayer) {
    return;
  }
  for (int i = 0; i < destLayer->neuronsNumber(); ++i) {
    for (int j = 0; j < mNeuronsNumber; ++j) {
      if (destLayer->getNeuron(i)->isBiased())
          continue;
      mNeurons[j].connectTo(destLayer->getNeuron(i));
    }
  }
  setNextLayer(destLayer);
  destLayer->setPrevLayer(this);
}

void AI::YAAnnLayer::disconnectFrom(YAAnnLayer* destLayer)
{
  if (destLayer != NULL && mNextLayer != destLayer) {
    return;
  }
  for (int i = 0; i < destLayer->neuronsNumber(); ++i) {
    for (int j = 0; j < mNeuronsNumber; ++j) {
      mNeurons[j].disconnectFrom(destLayer->getNeuron(i));
    }
  }
  setNextLayer(NULL);
  destLayer->setPrevLayer(NULL);
}

void AI::YAAnnLayer::disconnectFromNeuron(YAAnnNeuron* neuron)
{
  for (int i = 0; i < neuronsNumber(); ++i) {
    mNeurons[i].disconnectFrom(neuron);
  }
}

void AI::YAAnnLayer::setBiasedNeuron(int id, double output)
{
  if (id >= 0 && id < neuronsNumber()) {
    mNeurons[id].setIsBiased(true);
    mNeurons[id].setOutput(output);
  }    
}

void AI::YAAnnLayer::randWeights()
{
  if (mRand == NULL)
    return;  
  double ni = .0;
  ni = sqrt(2.0 / mNeuronsNumber);
  for (int i = 0; i < mNeuronsNumber; ++i) {
    for (int j = 0; j < mNeurons[i].axonsNumber(); ++j) {
      mNeurons[i].getAxon(j)->setWeight(mRand->rand(.0, 1.) * ni);
    }
  }
}

void AI::YAAnnLayer::rebuildLayer(int newNeuronsNumber)
{
  if (newNeuronsNumber > 0 && newNeuronsNumber != mNeuronsNumber) {
      delete [] mNeurons;
      mNeurons = new YAAnnNeuron[newNeuronsNumber];
      mNeuronsNumber = newNeuronsNumber;
  }
}

void AI::YAAnnLayer::setActFunTo(int neuronIndex, YAAnnActFun* actFun)
{
  if (neuronIndex >= 0 && neuronIndex < mNeuronsNumber) {
    mNeurons[neuronIndex].setActFun(actFun);
  }
}

void AI::YAAnnLayer::setActFunToAllNeurons(YAAnnActFun* actFun)
{
  for (int i = 0; i < mNeuronsNumber; ++i) {
    setActFunTo(i, actFun);
  }
}
