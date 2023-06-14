#include "yaann.h"

#include <string.h>

AI::YAAnnNeuron::YAAnnNeuron(
        double input,
        double output,
        double delta,
        double error,
        double idealOutput,
        bool isBiased)
    :
        mInput(input),
        mOutput(output),
        mDelta(delta),
        mError(error),
        mIdealOutput(idealOutput),
        mIsBiased(isBiased)
{ /*ctor*/ }

AI::YAAnnNeuron::~YAAnnNeuron()
{ /*dtor*/ }


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

bool AI::YAAnnNeuron::isBiased() const
{
    return mIsBiased;
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

