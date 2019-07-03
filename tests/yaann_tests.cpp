#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "yaann_neuron_test.h"
#include "yaann_axon_test.h"
#include "yaann_layer_test.h"
#include "yaann_test.h"

int main(int argc, char *argv[])
{
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
