#ifndef __COMMON_H
#define __COMMON_H

#include "yaann.h"
#include <memory>
#include <cstddef>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace AI {
  
  class YAAnnActFunStub0 : public YAAnnActFun
  {
  public:
    double f(double)
    { return .0; }
    double fPrime(double)
    { return .0; }
  };
  
  class YAAnnActFunStub1 : public YAAnnActFun
  {
  public:
    double f(double)
    { return 1.; }
    double fPrime(double)
    { return 1.; }
    const char* name()
    { return "stub1"; }
  };

  class YAAnnRandStub0 : public YAAnnRand
  {
  public:
    double rand(double, double)
    { return 0.; }
    const char* name()
    { return "stub0"; }
  };
  
}


#endif
