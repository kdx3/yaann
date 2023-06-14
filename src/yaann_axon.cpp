#include "yaann.h"

AI::YAAnnAxon::YAAnnAxon(
        double weight,
        double grad,
        double prevWeightDelta
) :
    mWeight(weight),
    mGrad(grad),
    mPrevWeightDelta(prevWeightDelta)
{ /*ctor*/ }

double AI::YAAnnAxon::getWeight() const
{
    return mWeight;
}

double AI::YAAnnAxon::getGrad() const
{
    return mGrad;
}

double AI::YAAnnAxon::getPrevWeightDelta() const
{
    return mPrevWeightDelta;
}

void AI::YAAnnAxon::setWeight(double weight)
{
    mWeight = weight;
}

void AI::YAAnnAxon::setGrad(double grad)
{
    mGrad = grad;
}

void AI::YAAnnAxon::incGrad(double grad)
{
    mGrad += grad;
}

void AI::YAAnnAxon::setPrevWeightDelta(double prevWeightDelta)
{
    mPrevWeightDelta = prevWeightDelta;
}

void AI::YAAnnAxon::updateWeight(double epsilon, double alpha)
{
    double weightDelta = epsilon * mGrad + alpha * mPrevWeightDelta;
    mWeight += weightDelta;
    mPrevWeightDelta = weightDelta;
}
