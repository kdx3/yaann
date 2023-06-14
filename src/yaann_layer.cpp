#include "yaann.h"

AI::YAAnnLayer::YAAnnLayer(
        layerType_t layerType,
        int neuronsNumber
    ) :
    mLayerType(layerType_t::UNKNOWN),
    mNeuronsNumber(0),
    mNeurons(NULL)
{
    buildLayer(layerType, neuronsNumber);
}

AI::YAAnnLayer::~YAAnnLayer()
{
    destroyLayer();
}

AI::YAAnnLayer::layerType_t AI::YAAnnLayer::getLayerType() const
{
    return mLayerType;
}

void AI::YAAnnLayer::buildLayer(layerType_t layerType, int neuronsNumber)
{
    destroyLayer();
    mNeuronsNumber = neuronsNumber;
    mLayerType = layerType;
    mNeurons = new YAAnnNeuron[neuronsNumber];
}

void AI::YAAnnLayer::destroyLayer()
{
    delete [] mNeurons;
    mNeuronsNumber = 0;
    mLayerType = layerType_t::UNKNOWN;
    mNeurons = NULL;

}

int AI::YAAnnLayer::neuronsNumber() const
{
    return mNeuronsNumber;
}

void AI::YAAnnLayer::setLayerType(layerType_t layerType)
{
    mLayerType = layerType;
}

void AI::YAAnnLayer::setBiasedNeuron(int id, double output)
{
    mNeurons[id].setIsBiased(true);
    mNeurons[id].setOutput(output);
}

const AI::YAAnnNeuron& AI::YAAnnLayer::operator[](int id) const
{
    return mNeurons[id];
}

AI::YAAnnNeuron& AI::YAAnnLayer::operator[](int id)
{
    return mNeurons[id];
}
