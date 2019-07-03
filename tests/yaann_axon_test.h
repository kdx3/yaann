#include "common.h"

using namespace AI;

TEST(YAAnnAxon, TDD)
{
  YAAnnAxon axon0;
  ASSERT_EQ(NULL, axon0.getDestNeuron());
  ASSERT_DOUBLE_EQ(.0, axon0.getWeight());
  ASSERT_DOUBLE_EQ(.0, axon0.getGrad());
  YAAnnNeuron neuron0;
  axon0.setDestNeuron(&neuron0);
  ASSERT_EQ(&neuron0, axon0.getDestNeuron());
  axon0.setWeight(1.);
  ASSERT_DOUBLE_EQ(1., axon0.getWeight());
  axon0.setGrad(1.);
  ASSERT_DOUBLE_EQ(1., axon0.getGrad());
}
