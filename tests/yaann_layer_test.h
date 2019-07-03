#include "common.h"

using namespace AI;

TEST(YAAnnLayer, TDD)
{
  YAAnnLayer layer0(YAAnnLayer::layerType_t::INPUT, 1);
  ASSERT_EQ(YAAnnLayer::layerType_t::INPUT, layer0.getLayerType());
  layer0.setLayerType(YAAnnLayer::layerType_t::OUTPUT);
  ASSERT_EQ(YAAnnLayer::layerType_t::OUTPUT, layer0.getLayerType()); 
  ASSERT_EQ(1, layer0.neuronsNumber());  
  YAAnnLayer layer1(YAAnnLayer::layerType_t::OUTPUT, 1);
  layer0.connectTo(&layer1);
  ASSERT_EQ(1, layer0.getNeuron(0)->axonsNumber());
  ASSERT_EQ(layer1.getNeuron(0), layer0.getNeuron(0)->getAxon(0)->getDestNeuron());
  ASSERT_EQ(&layer1, layer0.getNextLayer());
  ASSERT_EQ(&layer0, layer1.getPrevLayer());
  layer0.disconnectFrom(&layer1);
  ASSERT_EQ(1, layer0.neuronsNumber());
  ASSERT_EQ(1, layer0.getNeuron(0)->axonsNumber());
  ASSERT_EQ(NULL, layer0.getNeuron(0)->getAxon(0)->getDestNeuron());
  ASSERT_EQ(NULL, layer0.getNextLayer());
  ASSERT_EQ(NULL, layer1.getPrevLayer());
  YAAnnLayer layer2(YAAnnLayer::layerType_t::HIDDEN);
  layer0.setLayerType(YAAnnLayer::layerType_t::INPUT);
  layer0.rebuildLayer(2);
  ASSERT_EQ(2, layer0.neuronsNumber());
  layer1.setLayerType(YAAnnLayer::layerType_t::HIDDEN);
  layer1.rebuildLayer(2);
  layer2.setLayerType(YAAnnLayer::layerType_t::OUTPUT);
  layer2.rebuildLayer(2);
  layer0.connectTo(&layer1);
  layer1.connectTo(&layer2);
  ASSERT_EQ(layer1.getNeuron(0), layer0.getNeuron(0)->getAxon(0)->getDestNeuron());
  ASSERT_EQ(layer1.getNeuron(1), layer0.getNeuron(0)->getAxon(1)->getDestNeuron());
  ASSERT_EQ(layer1.getNeuron(0), layer0.getNeuron(1)->getAxon(0)->getDestNeuron());
  ASSERT_EQ(layer1.getNeuron(1), layer0.getNeuron(1)->getAxon(1)->getDestNeuron());
  ASSERT_EQ(layer2.getNeuron(0), layer1.getNeuron(0)->getAxon(0)->getDestNeuron());
  ASSERT_EQ(layer2.getNeuron(1), layer1.getNeuron(0)->getAxon(1)->getDestNeuron());
  ASSERT_EQ(layer2.getNeuron(0), layer1.getNeuron(1)->getAxon(0)->getDestNeuron());
  ASSERT_EQ(layer2.getNeuron(1), layer1.getNeuron(1)->getAxon(1)->getDestNeuron());
}

