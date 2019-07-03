#include "yaann.h"

AI::YAAnn::YAAnn(
		 YAAnnRand* rand,
		 YAAnnActFun* actFun,
		 int inputLayerSize,
		 int hiddenLayersNumber,
		 int hiddenLayerSize,
		 int outputLayerSize,
		 bool standardBiasedNeurons)
  : mRand(rand),
    mActFun(actFun),
    mInputLayer(NULL),
    mHiddenLayersNumber(0),
    mHiddenLayers(NULL),
    mOutputLayer(NULL),
    mEpsilon(.0),
    mAlpha(.0),
    mMSE(.0)
{
  setModel(inputLayerSize, hiddenLayersNumber, hiddenLayerSize, outputLayerSize,
	   standardBiasedNeurons);
}

void AI::YAAnn::destroyModel()
{
    delete mOutputLayer;
    delete [] mHiddenLayers;
    delete mInputLayer;
}

void AI::YAAnn::setModel(int inputLayerSize, int hiddenLayersNumber, int hiddenLayerSize, int outputLayerSize, bool standardBiasedNeurons)
{
    mInputLayer = (new YAAnnLayer(YAAnnLayer::layerType_t::INPUT, inputLayerSize));
    /*
    if (standardBiasedNeurons) {
      mInputLayer->getNeuron(inputLayerSize - 1)->setIsBiased(true);
      mInputLayer->getNeuron(inputLayerSize - 1)->setOutput(1.0);
    }
    */
    mHiddenLayersNumber = (hiddenLayersNumber);
    mHiddenLayers = (new YAAnnLayer[hiddenLayersNumber]);
    mOutputLayer = (new YAAnnLayer(YAAnnLayer::layerType_t::OUTPUT, outputLayerSize));
    mInputLayer->setRand(mRand);
    mOutputLayer->setRand(mRand);
    if (hiddenLayersNumber > 0) {
        for (int i = 0; i < hiddenLayersNumber; ++i) {
            mHiddenLayers[i].setLayerType(YAAnnLayer::layerType_t::HIDDEN);
            mHiddenLayers[i].rebuildLayer(hiddenLayerSize);
            mHiddenLayers[i].setActFunToAllNeurons(mActFun);
            mHiddenLayers[i].setRand(mRand);
	    mHiddenLayers[i].getNeuron(hiddenLayerSize - 1)->setIsBiased(true);
	    mHiddenLayers[i].getNeuron(hiddenLayerSize - 1)->setOutput(1.0);	    
            if (i > 0) {
                mHiddenLayers[i - 1].connectTo(&mHiddenLayers[i]);
            }
        }
        mInputLayer->connectTo(&mHiddenLayers[0]);
        mHiddenLayers[hiddenLayersNumber - 1].connectTo(mOutputLayer);
    } else {
        mInputLayer->connectTo(mOutputLayer);
    }
    mInputLayer->setActFunToAllNeurons(mActFun);
    mOutputLayer->setActFunToAllNeurons(mActFun);
    mInputLayer->randWeights();
    for (int i = 0; i < hiddenLayersNumber; ++i) {
        mHiddenLayers[i].randWeights();
    }
}

AI::YAAnn::~YAAnn()
{
    destroyModel();
    delete mActFun;
    delete mRand;
}

AI::YAAnnRand* AI::YAAnn::getRand() const
{
  return mRand;
}

AI::YAAnnActFun* AI::YAAnn::getActFun() const
{
  return mActFun;
}

AI::YAAnnLayer* AI::YAAnn::getInputLayer() const
{
  return mInputLayer;
}

AI::YAAnnLayer* AI::YAAnn::getOutputLayer() const
{
  return mOutputLayer;
}

AI::YAAnnLayer* AI::YAAnn::getHiddenLayer(int index) const
{
  if (index >= 0 && index < mHiddenLayersNumber)
    return &mHiddenLayers[index];
  return NULL;
}

double AI::YAAnn::getEpsilon() const
{
  return mEpsilon;
}

double AI::YAAnn::getAlpha() const
{
  return mAlpha;
}

double AI::YAAnn::getMSE() const
{
  return mMSE;
}

void AI::YAAnn::setRand(YAAnnRand* rand)
{
  delete mRand;
  mRand = rand;
}

void AI::YAAnn::setActFun(YAAnnActFun* actFun)
{
  delete mActFun;
  mActFun = actFun;
}

void AI::YAAnn::setEpsilon(double epsilon)
{
  mEpsilon = epsilon;
}

void AI::YAAnn::setAlpha(double alpha)
{
  mAlpha = alpha;
}

void AI::YAAnn::setMSE(double MSE)
{
  mMSE = MSE;
}

int AI::YAAnn::hiddenLayersNumber() const
{
  return mHiddenLayersNumber;
}

void AI::YAAnn::setInputs(double* inputs, int inputsSize)
{
  if (mInputLayer != NULL && inputsSize <= mInputLayer->neuronsNumber()) {
    for (int i = 0; i < inputsSize; ++i) {
      if (!mInputLayer->getNeuron(i)->isBiased())
          mInputLayer->getNeuron(i)->setOutput(inputs[i]);
    }
  }
}

void AI::YAAnn::setIdealOutputs(double* idealOutputs, int idealOutputsSize)
{
  if (mOutputLayer != NULL && idealOutputsSize <= mOutputLayer->neuronsNumber()) {
    for (int i = 0; i < idealOutputsSize; ++i) {
      if (!mOutputLayer->getNeuron(i)->isBiased())
	mOutputLayer->getNeuron(i)->setIdealOutput(idealOutputs[i]);
    }
  }
}

void AI::YAAnn::forwardProp()
{
  YAAnnLayer* currentLayer
    = mInputLayer;
  while (currentLayer != mOutputLayer) {
    for (int i = 0; i < currentLayer->getNextLayer()->neuronsNumber(); ++i) {
      if (currentLayer->getNextLayer()->getNeuron(i)->isBiased())
          continue;
      double sum = .0;
      for (int j = 0; j < currentLayer->neuronsNumber(); ++j) {
          sum += currentLayer->getNeuron(j)->calcOutputToDestNeuron(
                      currentLayer->getNextLayer()->getNeuron(i));
      }
      currentLayer->getNextLayer()->getNeuron(i)->setSumToInput(sum);
    }   
    currentLayer = currentLayer->getNextLayer();
  } 
}

void AI::YAAnn::calcMSE()
{
    int n = 0;
    double MSE = .0;
    for (int i = 0; i < mOutputLayer->neuronsNumber(); ++i) {
      if (!mOutputLayer->getNeuron(i)->isBiased()) {
        ++n;
        mOutputLayer->getNeuron(i)->setError(mOutputLayer->getNeuron(i)->getOutput() - mOutputLayer->getNeuron(i)->getIdealOutput());
        MSE += pow(mOutputLayer->getNeuron(i)->getError(), 2.);
      }
    }
    mMSE = MSE / n;
}

void AI::YAAnn::backwardProp()
{
  YAAnnLayer* currentLayer = mOutputLayer;
  while (currentLayer != mInputLayer) {
    for (int i = 0; i < currentLayer->neuronsNumber(); ++i) {
      if (currentLayer == mOutputLayer) {
          if (!currentLayer->getNeuron(i)->isBiased())
              currentLayer->getNeuron(i)->setDelta(-currentLayer->getNeuron(i)->getError() *
                                                   currentLayer->getNeuron(i)->getActFun()->fPrime(
                                                       currentLayer->getNeuron(i)->getInput()));
      } else {
          if (currentLayer != mInputLayer) {
              if (!currentLayer->getNeuron(i)->isBiased())
              {
                  double delta = .0;
                  for (int j = 0; j < currentLayer->getNeuron(i)->axonsNumber(); ++j) {
                      delta += currentLayer->getNeuron(i)->getAxon(j)->getWeight() *
                              currentLayer->getNeuron(i)->getAxon(j)->getDestNeuron()->getDelta();
                  }
                  delta *= currentLayer->getNeuron(i)->getActFun()->fPrime(currentLayer->getNeuron(i)->getInput());
                  currentLayer->getNeuron(i)->setDelta(delta);
              }
          }
          for (int j = 0; j < currentLayer->getNeuron(i)->axonsNumber(); ++j) {
              currentLayer->getNeuron(i)->getAxon(j)->setGrad(currentLayer->getNeuron(i)->getOutput() * currentLayer->getNeuron(i)->getAxon(j)->getDestNeuron()->getDelta());
          }
      }
    }
    currentLayer = currentLayer->getPrevLayer();
  }
}

void AI::YAAnn::updateWeights()
{
  YAAnnLayer* currentLayer
    = mInputLayer;
  while (currentLayer != mOutputLayer) {
    for (int i = 0; i < currentLayer->neuronsNumber(); ++i) {
      for (int j = 0; j < currentLayer->getNeuron(i)->axonsNumber(); ++j) {
    	currentLayer->getNeuron(i)->getAxon(j)->updateWeight(mEpsilon, mAlpha);
      }
    }
    currentLayer = currentLayer->getNextLayer();
  }
}

void AI::YAAnn::batchTraining(
			      int epochs,
			      int step,
			      int inputVectorSize,
			      int outputVectorSize,
			      int N,
			      double** inputs,
			      double** outputs,
			      double alpha,
			      double epsilon,
			      YAAnnReport* pR
			      )
{
  if (mInputLayer == NULL || mOutputLayer == NULL)
    return;
  if (inputVectorSize != mInputLayer->neuronsNumber() -
      mInputLayer->biasedNeuronsNumber()
      || outputVectorSize != mOutputLayer->neuronsNumber()
      - mOutputLayer->biasedNeuronsNumber()
      )
    return;
  if (N <= 0)
    return;

  setAlpha(alpha);
  setEpsilon(epsilon);
  
  int n = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    setInputs(inputs[n], inputVectorSize);
    setIdealOutputs(outputs[n], outputVectorSize);
    ++n;
    if (n == N) {
      n = 0;
    }
    forwardProp();
    backwardProp();
    updateWeights();
    pR->report(*this);

    /*
    if (epoch % step == 0) {
      pR->report(*this);      
      updateWeights();
      }*/
  }
}

void AI::YAAnn::iterTraining(
        int inputVectorSize,
        int outputVectorSize,
        double* inputVector,
        double* outputVector,
        double alpha,
        double epsilon,
        YAAnnReport* pR)
{
    if (mInputLayer == NULL || mOutputLayer == NULL)
      return;
    if (inputVectorSize != mInputLayer->neuronsNumber() -
        mInputLayer->biasedNeuronsNumber()
        || outputVectorSize != mOutputLayer->neuronsNumber()
        - mOutputLayer->biasedNeuronsNumber()
        )
      return;
    setAlpha(alpha);
    setEpsilon(epsilon);
    setInputs(inputVector, inputVectorSize);
    setIdealOutputs(outputVector, outputVectorSize);
    forwardProp();
    calcMSE();
    backwardProp();
    updateWeights();
    if (pR != NULL)
        pR->report(*this);
}

void AI::YAAnn::saveToFile(YAAnnReport* report, const char* fileName)
{
  report->saveToFile(*this, fileName);
}

void AI::YAAnn::saveMdlToFile(YAAnnReport* report, const char* fileName)
{
  report->saveMdlToFile(*this, fileName);
}

void AI::YAAnn::readMdlFromFile(YAAnnReport* report, const char* fileName)
{
  report->readMdlFromFile(*this, fileName);
}
