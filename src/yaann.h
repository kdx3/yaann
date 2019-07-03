/**

 */
#ifndef __YAANN_H
#define __YAANN_H

#include <stddef.h>
#include <math.h>

namespace AI
{

  class YAAnnRand
  {
  public:
    virtual double rand(double, double) = 0;
    virtual const char* name() = 0;
    virtual ~YAAnnRand()
    { /*dtor*/ }
  };
  
  class YAAnnActFun
  {
  public:
    virtual double f(double) = 0;
    virtual double fPrime(double) = 0;
    virtual const char* name() = 0;
    virtual ~YAAnnActFun()
    { /*dtor*/ }
  };

  class YAAnnActFunSig : public YAAnnActFun
  {
  public:
    double f(double x) override
    {
      return 1. / ( 1. + exp(-x) );
    }
    double fPrime(double x) override
    {
      return exp(x) / pow( exp(x) + 1., 2. );
    }
    const char* name() override
    { return "sig"; }
  };

  class YAAnnActFunHyp : public YAAnnActFun
  {
  public:
    double f(double x) override
    {
      return ( exp(2. * x) - 1 ) / ( exp(2. * x) + 1 );
    }
    double fPrime(double x) override
    {
      return 4 * exp(2. * x) / pow( exp(2. * x) + 1, 2. );
    }
    const char* name() override
    { return "hyp"; }
  };

  class YAAnnActFunTanh : public YAAnnActFun
  {
  public:
      double f(double x) override
      {
          return ( (exp(x) - exp(-x)) / (exp(x) + exp(-x)));
      }
      double fPrime(double x) override
      {
          return 1. - pow( f(x), 2. );
      }
      const char* name() override
      { return "tanh"; }
  };

  class YAAnnActFunATan : public YAAnnActFun
  {
      double f(double x) override
      {
          return atan(x);
      }
      double fPrime(double x) override
      {
          return 1. / ( pow(x, 2.) + 1. );
      }
      const char* name() override
      { return "atan"; }
  };

  class YAAnnActFunLeakyReLU : public YAAnnActFun
  {
  public:
      double f(double x) override;
      double fPrime(double x) override;
      const char* name() override;
  };

  class YAAnnActFunReLU : public YAAnnActFun
  {
  public:
      double f(double x) override;
      double fPrime(double x) override;
      const char* name() override;
  };


  class YAAnnAxon;
  
  class YAAnnNeuron
  {
  public:
    explicit YAAnnNeuron(
   		        YAAnnActFun* = NULL,
			double = .0,
			double = .0,
			double = .0,
			double = .0,
			double  = .0,
			bool = false
			 );
    ~YAAnnNeuron();
    YAAnnActFun* getActFun() const;    
    double getInput() const;
    double getOutput() const;
    double getDelta() const;
    double getError() const;
    double getIdealOutput() const;
    int getInputsNumber() const;
    bool isBiased() const;
    int axonsNumber() const;
    YAAnnAxon* getAxon(int) const;
    int findAxonByDestNeuron(YAAnnNeuron*) const;
    double calcOutputToDestNeuron(YAAnnNeuron*) const;
    void setSumToInput(double);
    int findFreeAxon() const;
    void setActFun(YAAnnActFun*);
    void setInput(double input);
    void setOutput(double output);
    void setDelta(double delta);
    void setError(double error);
    void setIdealOutput(double idealOutput);
    void setIsBiased(bool);
    void setInputsNumber(int);
    void connectTo(YAAnnNeuron*);
    void disconnectFrom(YAAnnNeuron*);    
  private:
    YAAnnActFun* mActFun;
    double mInput;
    double mOutput;
    double mDelta;
    double mError;
    double mIdealOutput;
    bool mIsBiased;
    int mAxonsNumber;
    YAAnnAxon* mAxons;
    int mInputsNumber;
  };

  class YAAnnAxon
  {
  public:
    explicit YAAnnAxon(YAAnnNeuron* = NULL, double = .0, double = .0, double = .0);
    YAAnnNeuron* getDestNeuron() const;
    double getWeight() const;
    double getGrad() const;
    double getPrevWeightDelta() const;
    void setDestNeuron(YAAnnNeuron*);
    void setWeight(double);
    void setGrad(double);
    void setPrevWeightDelta(double);
    void incGrad(double);
    void updateWeight(double, double);
  private:
    YAAnnNeuron* mDestNeuron;
    double mWeight;    
    double mGrad;
    double mPrevWeightDelta;
  };

  
  class YAAnnLayer
  {
  public:
    enum layerType_t {UNKNOWN = -1, INPUT, OUTPUT, HIDDEN};
    YAAnnLayer(
	       layerType_t = UNKNOWN,
	       int = 0,
	       YAAnnLayer* = NULL,
	       YAAnnLayer* = NULL,
	       YAAnnRand* = NULL
	       );
    ~YAAnnLayer();
    layerType_t getLayerType() const;
    int neuronsNumber() const;
    int biasedNeuronsNumber() const;
    YAAnnLayer* getNextLayer() const;
    YAAnnLayer* getPrevLayer() const;
    YAAnnNeuron* getNeuron(int) const;
    YAAnnRand* getRand() const;
    void setLayerType(layerType_t);
    void setNextLayer(YAAnnLayer*);
    void setPrevLayer(YAAnnLayer*);
    void connectTo(YAAnnLayer*);
    void disconnectFrom(YAAnnLayer*);
    void disconnectFromNeuron(YAAnnNeuron*);
    void setBiasedNeuron(int, double = 1.0);
    void randWeights();
    void rebuildLayer(int);
    void setActFunTo(int, YAAnnActFun*);
    void setActFunToAllNeurons(YAAnnActFun*);
    void setRand(YAAnnRand*);    
  private:
    layerType_t mLayerType;
    int mNeuronsNumber;
    YAAnnNeuron* mNeurons;
    YAAnnLayer* mNextLayer;
    YAAnnLayer* mPrevLayer;
    YAAnnRand* mRand;
  };

  class YAAnnReport;

  class YAAnn
  {
  public:
    YAAnn(YAAnnRand*,YAAnnActFun*, int, int, int, int, bool = true);
    ~YAAnn();
    YAAnnRand* getRand() const;
    YAAnnActFun* getActFun() const;
    YAAnnLayer* getInputLayer() const;
    YAAnnLayer* getOutputLayer() const;
    YAAnnLayer* getHiddenLayer(int) const;
    double getEpsilon() const;
    double getAlpha() const;
    double getMSE() const;
    void setRand(YAAnnRand*);
    void setActFun(YAAnnActFun*);
    void setEpsilon(double);
    void setAlpha(double);
    void setMSE(double);
    int hiddenLayersNumber() const;
    void setInputs(double*, int);
    void setIdealOutputs(double*, int);
    void forwardProp();
    void backwardProp();
    void updateWeights();
    void batchTraining(int, int, int, int, int, double**, double**, double, double, YAAnnReport*);
    void iterTraining(int, int, double*, double*, double, double, YAAnnReport*);
    void saveToFile(YAAnnReport*, const char*);
    void saveMdlToFile(YAAnnReport*, const char*);
    void readMdlFromFile(YAAnnReport*, const char*);
    void destroyModel();
    void setModel(int, int, int, int, bool);
    void calcMSE();
  private:
    YAAnnRand* mRand;
    YAAnnActFun* mActFun;
    YAAnnLayer* mInputLayer;
    int mHiddenLayersNumber;
    YAAnnLayer* mHiddenLayers;
    YAAnnLayer* mOutputLayer;
    double mEpsilon;
    double mAlpha;
    double mMSE;
  };

  class YAAnnReport
  {
  public:
    virtual ~YAAnnReport()
    { /*dtor*/ }
    virtual void report(const YAAnn&) = 0;
    virtual void saveToFile(const YAAnn&, const char*) = 0;
    virtual void saveMdlToFile(const YAAnn&, const char*) = 0;
    virtual void readMdlFromFile(YAAnn&, const char*) = 0;
  };

}

#endif
