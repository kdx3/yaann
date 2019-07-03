#include "yaann.h"

#include <string.h>

AI::YAAnnNeuron::YAAnnNeuron(
			     YAAnnActFun* actFun,
			     double input,
			     double output,
			     double delta,
			     double error,
			     double idealOutput,
			     bool isBiased)
  :
      mActFun(actFun),
      mInput(input),
      mOutput(output),
      mDelta(delta),
      mError(error),
      mIdealOutput(idealOutput),
      mIsBiased(isBiased),
      mAxonsNumber(0),
      mAxons(NULL),
      mInputsNumber(0)
{ /*ctor*/ }

AI::YAAnnNeuron::~YAAnnNeuron()
{ delete [] mAxons; }

AI::YAAnnActFun* AI::YAAnnNeuron::getActFun() const
{
  return mActFun;
}

double AI::YAAnnNeuron::getInput() const
{
  return mInput;
}

double AI::YAAnnNeuron::getOutput() const
{
  return mOutput;
}

double AI::YAAnnNeuron::getDelta() const
{
  return mDelta;
}

double AI::YAAnnNeuron::getIdealOutput() const
{
  return mIdealOutput;
}

double AI::YAAnnNeuron::getError() const
{
  return mError;
}

int AI::YAAnnNeuron::axonsNumber() const
{
  return mAxonsNumber;
}

AI::YAAnnAxon* AI::YAAnnNeuron::getAxon(int index) const
{
  if (mAxonsNumber > 0 && index < mAxonsNumber)
    return &mAxons[index];
  return NULL;
}

int AI::YAAnnNeuron::findAxonByDestNeuron(YAAnnNeuron* destNeuron) const
{
  int ret = -1;
  for (int i = 0; i < mAxonsNumber; ++i) {
    if (mAxons[i].getDestNeuron() == destNeuron) {
      return i;
    }
  }
  return ret;
}

double AI::YAAnnNeuron::calcOutputToDestNeuron(YAAnnNeuron* destNeuron) const
{
  int destNeuronAxonIndex = findAxonByDestNeuron(destNeuron);
  if (destNeuronAxonIndex >= 0 && destNeuronAxonIndex < mAxonsNumber) {
    return getOutput() * getAxon(destNeuronAxonIndex)->getWeight();
  }
  return .0;
}

void AI::YAAnnNeuron::setSumToInput(double sum)
{
  if (mIsBiased)
    return;
  setInput(sum);
  if (mActFun != NULL) {
    setOutput(mActFun->f(sum));
  }
}

int AI::YAAnnNeuron::findFreeAxon() const
{
  int ret = -1;
  for (int i = 0; i < mAxonsNumber; ++i) {
    if (mAxons[i].getDestNeuron() == NULL) {
      return i;
    }
  }
  return ret;
}

bool AI::YAAnnNeuron::isBiased() const
{
  return mIsBiased;
}

void AI::YAAnnNeuron::setActFun(YAAnnActFun* actFun)
{
  mActFun = actFun;
}

void AI::YAAnnNeuron::setInput(double input)
{
  mInput = input;
}

void AI::YAAnnNeuron::setOutput(double output)
{
  mOutput = output;
}

void AI::YAAnnNeuron::setDelta(double delta)
{
  mDelta = delta;
}

void AI::YAAnnNeuron::setError(double error)
{
  mError = error;
}

void AI::YAAnnNeuron::setIdealOutput(double idealOutput)
{
  mIdealOutput = idealOutput;
}

void AI::YAAnnNeuron::setIsBiased(bool isBiased)
{
  mIsBiased = isBiased;
}

void AI::YAAnnNeuron::connectTo(YAAnnNeuron* destNeuron)
{
  int axonIndex = findAxonByDestNeuron(destNeuron);
  if (axonIndex >= 0 && axonIndex < mAxonsNumber) {
    return;
  }
  axonIndex = findFreeAxon();
  if (axonIndex >= 0 && axonIndex < mAxonsNumber) {
    mAxons[axonIndex].setDestNeuron(destNeuron);
    destNeuron->setInputsNumber(destNeuron->getInputsNumber() + 1);
    return;
  }
  YAAnnAxon* tmpPtr = mAxons;
  mAxons = new YAAnnAxon[mAxonsNumber + 1];  
  if (mAxonsNumber > 0 && tmpPtr != NULL) {
      memcpy(mAxons, tmpPtr, sizeof(YAAnnAxon) * mAxonsNumber);
      delete [] tmpPtr;
  }
  mAxons[mAxonsNumber].setDestNeuron(destNeuron);
  destNeuron->setInputsNumber(destNeuron->getInputsNumber() + 1);
  mAxonsNumber++;
}

void AI::YAAnnNeuron::disconnectFrom(YAAnnNeuron* destNeuron)
{
  int axonIndex = findAxonByDestNeuron(destNeuron);
  destNeuron->setInputsNumber(destNeuron->getInputsNumber() - 1);
  if (axonIndex >= 0 && axonIndex < mAxonsNumber) {
    mAxons[axonIndex].setDestNeuron(NULL);
    mAxons[axonIndex].setWeight(.0);
    mAxons[axonIndex].setGrad(.0);
  }
}

void AI::YAAnnNeuron::setInputsNumber(int inputsNumber)
{
  mInputsNumber = inputsNumber;
}

int AI::YAAnnNeuron::getInputsNumber() const
{
  return mInputsNumber;
}
