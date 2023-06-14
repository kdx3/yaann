/**

*/
#ifndef __YAANN_H
#define __YAANN_H

#include <stddef.h>
#include <math.h>
#include <random>

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


    class YAAnnRandNormDistr : public YAAnnRand
    {
        public:
            YAAnnRandNormDistr()
                :
                    YAAnnRand(),
                    mRandDev()
            { /*ctor*/ }
            double rand(double m, double sd) override
            {
                std::normal_distribution<double> distr(m, sd);
                return distr(mRandDev);
            }
            const char* name() override
            {
                return "norm";
            }
        private:
            std::random_device mRandDev;
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
                    double input = .0,
                    double output = .0,
                    double delta = .0,
                    double error = .0,
                    double idealOutput = .0,
                    bool isBiased = false
                    );
            ~YAAnnNeuron();
            double getInput() const;
            double getOutput() const;
            double getDelta() const;
            double getError() const;
            double getIdealOutput() const;
            bool isBiased() const;
            void setInput(double input);
            void setOutput(double output);
            void setIdealOutput(double idealOutput);
            void setDelta(double delta);
            void setError(double error);
            void setIsBiased(bool);
        private:
            double mInput;
            double mOutput;
            double mDelta;
            double mError;
            double mIdealOutput;
            bool mIsBiased;
    };


    class YAAnnAxon
    {
        public:
            explicit YAAnnAxon(
                    double weight = .0,
                    double grad = .0,
                    double prevWeightDelta = .0);
            double getWeight() const;
            double getGrad() const;
            double getPrevWeightDelta() const;
            void setWeight(double);
            void setGrad(double);
            void setPrevWeightDelta(double);
            void incGrad(double);
            void updateWeight(double, double);
        private:
            double mWeight;
            double mGrad;
            double mPrevWeightDelta;
    };


    class YAAnnLayer
    {
        public:
            enum layerType_t {UNKNOWN=-1,INPUT, OUTPUT, HIDDEN, CONTEXT};
            YAAnnLayer(
                    layerType_t = UNKNOWN,
                    int neuronsNumber = 0);
            ~YAAnnLayer();
            layerType_t getLayerType() const;
            void buildLayer(layerType_t layerType, int neuronsNumber);
            void destroyLayer();
            int neuronsNumber() const;
            void setLayerType(layerType_t);
            void setBiasedNeuron(int, double = 1.0);
            const YAAnnNeuron& operator[](int) const;
            YAAnnNeuron& operator[](int);
        private:
            layerType_t mLayerType;
            int mNeuronsNumber;
            YAAnnNeuron* mNeurons;
    };


    class YAAnnLayerConnect
    {
        public:
            YAAnnLayerConnect(
                    YAAnnLayer* srcLayer = NULL,
                    YAAnnLayer* dstLayer = NULL,
                    YAAnnActFun* actFun = new YAAnnActFunSig,
                    YAAnnRand* randFun = new YAAnnRandNormDistr);
            ~YAAnnLayerConnect();
            int axonsNumber() const;
            const YAAnnAxon& operator[](int) const;
            YAAnnAxon& operator[](int);
            YAAnnActFun* getActFun() const;
            void setActFun(YAAnnActFun*);
            YAAnnRand* getRand() const;
            void setRand(YAAnnRand*);
            void connect(YAAnnLayer* srcLayer, YAAnnLayer* dstLayer);
            void forward(bool delta);
            void backward();
            void randWeights();
            void updateWeights(double epsilon, double alpha);
        private:
            YAAnnLayer* mSrcLayer;
            YAAnnLayer* mDstLayer;
            YAAnnActFun* mActFun;
            YAAnnRand* mRand;
            YAAnnAxon* mAxons;
            int mAxonsNumber;
    };


    class YAAnn
    {
        public:
            enum rnnType_t{ NO_RNN_TYPE = 0, ELMAN_RNN_TYPE = 1, JORDAN_RNN_TYPE = 2 };
            YAAnn(
                    int inputLayerSize,
                    int hiddenLayersNumber,
                    int hiddenLayerSize,
                    int outputLayerSize,
                    bool standardBiasedNeurons = true,
                    rnnType_t rnnType = NO_RNN_TYPE
            );
            ~YAAnn();
            void destroyModel();
            void buildModel(int, int, int, int, bool, rnnType_t);
        private:
            YAAnnLayer mInputLayer;
            int mHiddenLayersNumber;
            YAAnnLayer* mHiddenLayers;
            YAAnnLayer mOutputLayer;
            rnnType_t mRnnType;
            YAAnnLayer mContextLayer;
            int mLayerConnSize;
            YAAnnLayerConnect* mLayerConn;
    };


    class YAAnnMdlStorage
    {
        public:
            void saveMdlToDot(const YAAnn&, const char*);
            void saveMdlToFile(const YAAnn&, const char*);
            void readMdlFromFile(YAAnn&, const char*);
    };


}

#endif
