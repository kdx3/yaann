#ifndef YAANN_ADDONS_H
#define YAANN_ADDONS_H

#include "yaann.h"
#include <random>

namespace AI {
class YAAnnRandNormDistr : public YAAnnRand
{
public:
  YAAnnRandNormDistr() : YAAnnRand(), mRandDev()
  {/*ctor*/}
  double rand(double m, double sd) override;
  const char* name() override;
private:
  std::random_device mRandDev;
};

class YAAnnReportStdFile : public YAAnnReport
{
public:
  YAAnnReportStdFile();
  void report(const YAAnn& ann) override;
  void saveToFile(const YAAnn& ann, const char* fileName) override;
  void saveMdlToFile(const YAAnn& yaann, const char* fileName) override;
  void readMdlFromFile(YAAnn& yaann, const char* fileName) override;
};
}

#endif // YAANN_ADDONS_H
