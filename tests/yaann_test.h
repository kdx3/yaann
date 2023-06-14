#include "common.h"

using namespace AI;

TEST(YAAnn, TDD)
{
  YAAnn ann0(new YAAnnRandStub0, new YAAnnActFunSig, 3, 1, 3, 1);
  //  ASSERT_NE(std::nullptr, ann0.getInputLayer());
  //  ASSERT_NE(std::nullptr, ann0.getOutputLayer());
  ASSERT_EQ(1, ann0.hiddenLayersNumber());
  //  ASSERT_NE(std::nullptr, ann0.getHiddenLayer(0));
  ASSERT_EQ(3, ann0.getInputLayer()->neuronsNumber());
  ASSERT_EQ(3, ann0.getHiddenLayer(0)->neuronsNumber());
  ASSERT_EQ(1, ann0.getOutputLayer()->neuronsNumber());
  ASSERT_DOUBLE_EQ(.0, ann0.getEpsilon());
  ASSERT_DOUBLE_EQ(.0, ann0.getAlpha());
  ASSERT_DOUBLE_EQ(.0, ann0.getMSE());
  double inputs0[2] = {1.0, 0.0};
  ann0.setInputs(inputs0, 2);
  double idealOutputs0[1] = {1.};
  ann0.setIdealOutputs(idealOutputs0, 1);
  ASSERT_DOUBLE_EQ(1.0, ann0.getOutputLayer()->getNeuron(0)->getIdealOutput());
  ASSERT_DOUBLE_EQ(1.0, ann0.getInputLayer()->getNeuron(0)->getOutput());
  ASSERT_DOUBLE_EQ(0.0, ann0.getInputLayer()->getNeuron(1)->getOutput());
  ann0.getInputLayer()->getNeuron(0)->getAxon(0)->setWeight(-.07);
  ann0.getInputLayer()->getNeuron(0)->getAxon(1)->setWeight(.94);
  ann0.getInputLayer()->getNeuron(1)->getAxon(0)->setWeight(.22);
  ann0.getInputLayer()->getNeuron(1)->getAxon(1)->setWeight(.46);
  ann0.getInputLayer()->getNeuron(2)->setIsBiased(true);
  ann0.getInputLayer()->getNeuron(2)->setOutput(1.);
  ann0.getInputLayer()->getNeuron(2)->getAxon(0)->setWeight(-.46);
  ann0.getInputLayer()->getNeuron(2)->getAxon(1)->setWeight(.10);
  ann0.getHiddenLayer(0)->getNeuron(0)->getAxon(0)->setWeight(-.22);
  ann0.getHiddenLayer(0)->getNeuron(1)->getAxon(0)->setWeight(.58);
  ann0.getHiddenLayer(0)->getNeuron(2)->setIsBiased(true);
  ann0.getHiddenLayer(0)->getNeuron(2)->setOutput(1.0);
  ann0.getHiddenLayer(0)->getNeuron(2)->getAxon(0)->setWeight(.78);
  ann0.forwardProp();

  EXPECT_NEAR(-.53, ann0.getHiddenLayer(0)->getNeuron(0)->getInput(), .01); 
  EXPECT_NEAR(1.05, ann0.getHiddenLayer(0)->getNeuron(1)->getInput(), .02);
  EXPECT_NEAR(0.37, ann0.getHiddenLayer(0)->getNeuron(0)->getOutput(), .01);
  EXPECT_NEAR(0.74, ann0.getHiddenLayer(0)->getNeuron(1)->getOutput(), .01);
  EXPECT_NEAR(1.13, ann0.getOutputLayer()->getNeuron(0)->getInput(), .01);
  EXPECT_NEAR(.75, ann0.getOutputLayer()->getNeuron(0)->getOutput(), .01);

  ASSERT_NEAR(.06, ann0.getMSE(), .06);

  ann0.setEpsilon(.7);
  ann0.setAlpha(.3);
  ann0.backwardProp();

  EXPECT_DOUBLE_EQ(.0, ann0.getInputLayer()->getNeuron(0)->getDelta());
  EXPECT_DOUBLE_EQ(.0, ann0.getInputLayer()->getNeuron(1)->getDelta());
  EXPECT_DOUBLE_EQ(.0, ann0.getInputLayer()->getNeuron(2)->getDelta());
  EXPECT_NEAR(-.0025, ann0.getHiddenLayer(0)->getNeuron(0)->getDelta(), .1);
  EXPECT_NEAR(.0055, ann0.getHiddenLayer(0)->getNeuron(1)->getDelta(), .1);
  EXPECT_DOUBLE_EQ(.0, ann0.getHiddenLayer(0)->getNeuron(2)->getDelta());
  EXPECT_NEAR(.045, ann0.getOutputLayer()->getNeuron(0)->getDelta(), .1);

  EXPECT_NEAR(1. * -.0025, ann0.getInputLayer()->getNeuron(0)->getAxon(0)->getGrad(), .1);
  EXPECT_NEAR(1. * .0055, ann0.getInputLayer()->getNeuron(0)->getAxon(1)->getGrad(), .1);

  EXPECT_NEAR(0. * -.0025, ann0.getInputLayer()->getNeuron(1)->getAxon(0)->getGrad(), .1);
  EXPECT_NEAR(0. * .0055, ann0.getInputLayer()->getNeuron(1)->getAxon(1)->getGrad(), .1);

  EXPECT_NEAR(1. * -.0025, ann0.getInputLayer()->getNeuron(2)->getAxon(0)->getGrad(), .1);
  EXPECT_NEAR(1. * .0055, ann0.getInputLayer()->getNeuron(2)->getAxon(1)->getGrad(), .1);

  EXPECT_NEAR(.37 * .045, ann0.getHiddenLayer(0)->getNeuron(0)->getAxon(0)->getGrad(), .1);
  EXPECT_NEAR(.74 * .045, ann0.getHiddenLayer(0)->getNeuron(1)->getAxon(0)->getGrad(), .1);
  EXPECT_NEAR(1. * .045, ann0.getHiddenLayer(0)->getNeuron(2)->getAxon(0)->getGrad(), .1);

  ann0.updateWeights();

  EXPECT_NEAR(.7 * -.0025 + .3 * .0, ann0.getInputLayer()->getNeuron(0)->getAxon(0)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * -.0025 + .3 * .0 - .07, ann0.getInputLayer()->getNeuron(0)->getAxon(0)->getWeight(), .1);
  EXPECT_NEAR(.7 * .0055 + .3 * .0, ann0.getInputLayer()->getNeuron(0)->getAxon(1)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .0055 + .3 * .0 + .94, ann0.getInputLayer()->getNeuron(0)->getAxon(1)->getWeight(), .1);

  EXPECT_NEAR(.7 * .0 + .3 * .0, ann0.getInputLayer()->getNeuron(1)->getAxon(0)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .0 + .3 * .0 + .22, ann0.getInputLayer()->getNeuron(1)->getAxon(0)->getWeight(), .1);
  EXPECT_NEAR(.7 * .0 + .3 * .0, ann0.getInputLayer()->getNeuron(1)->getAxon(1)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .0 + .3 * .0 + .46, ann0.getInputLayer()->getNeuron(1)->getAxon(1)->getWeight(), .1);

  EXPECT_NEAR(.7 * .0 + .3 * .0, ann0.getInputLayer()->getNeuron(1)->getAxon(0)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .0 + .3 * .0 + .22, ann0.getInputLayer()->getNeuron(1)->getAxon(0)->getWeight(), .1);
  EXPECT_NEAR(.7 * .0 + .3 * .0, ann0.getInputLayer()->getNeuron(1)->getAxon(1)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .0 + .3 * .0 + .46, ann0.getInputLayer()->getNeuron(1)->getAxon(1)->getWeight(), .1);

  EXPECT_NEAR(.7 * -.0025 + .3 * .0, ann0.getInputLayer()->getNeuron(2)->getAxon(0)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * -.0025 + .3 * .0 + -.46, ann0.getInputLayer()->getNeuron(2)->getAxon(0)->getWeight(), .1);
  EXPECT_NEAR(.7 * .0055 + .3 * .0, ann0.getInputLayer()->getNeuron(2)->getAxon(1)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .0055 + .3 * .0 + .10, ann0.getInputLayer()->getNeuron(2)->getAxon(1)->getWeight(), .1);

  EXPECT_NEAR(.7 * .016 + .3 * .0, ann0.getHiddenLayer(0)->getNeuron(0)->getAxon(0)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .016 + .3 * .0 - .22, ann0.getHiddenLayer(0)->getNeuron(0)->getAxon(0)->getWeight(), .1);

  EXPECT_NEAR(.7 * .033 + .3 * .0, ann0.getHiddenLayer(0)->getNeuron(1)->getAxon(0)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .033 + .3 * .0 + .58, ann0.getHiddenLayer(0)->getNeuron(1)->getAxon(0)->getWeight(), .1);

  EXPECT_NEAR(.7 * .045 + .3 * .0, ann0.getHiddenLayer(0)->getNeuron(2)->getAxon(0)->getPrevWeightDelta(), .1);
  EXPECT_NEAR(.7 * .045 + .3 * .0 + .78, ann0.getHiddenLayer(0)->getNeuron(2)->getAxon(0)->getWeight(), .1);


}
