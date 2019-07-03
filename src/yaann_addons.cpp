#include "yaann_addons.h"

#include <fstream>
#include <iostream>
#include <iomanip>

const char* AI::YAAnnRandNormDistr::name()
{
    return "norm";
}

double AI::YAAnnRandNormDistr::rand(double m, double sd)
{
  std::normal_distribution<double> distr(m, sd);
  return distr(mRandDev);
}

void AI::YAAnnReportStdFile::report(const YAAnn& ann)
{
  std::cout << "MSE: " << std::fixed << std::setprecision(10)
	    << ann.getMSE() << "\n";
}

AI::YAAnnReportStdFile::YAAnnReportStdFile()
    : YAAnnReport ()
{ /*ctor*/ }

void AI::YAAnnReportStdFile::saveToFile(const YAAnn& yaann, const char* fileName)
{
    char labelName = 'u';
    std::ofstream ofs(fileName);
    if (ofs.is_open()) {
      ofs << "digraph YAANN {\n";
      ofs << "\trankdir=\"LR\";\n";
      AI::YAAnnLayer* currentLayer
    = yaann.getInputLayer();
      int k = 0;
      while (currentLayer != NULL) {
    for (int i = 0; i < currentLayer->neuronsNumber(); ++i) {
      const char* nodeName;
      const char* fillColor;
      switch (currentLayer->getLayerType()) {
      case AI::YAAnnLayer::layerType_t::INPUT:
        nodeName = "I_";
        fillColor = "#ff0000";
    labelName = 'i';
        break;
      case AI::YAAnnLayer::layerType_t::OUTPUT:
        nodeName = "O_";
        fillColor = "#00ff00";
    labelName = 'o';
        break;
      case AI::YAAnnLayer::layerType_t::HIDDEN:
        nodeName = "H_";
        fillColor = "#0000ff";
    labelName = 'h';
        break;
      default:
        ofs << "\tUNKNOWN_" << i;
        break;
      }

      if (currentLayer->getNeuron(i)->isBiased()) {
        labelName = 'b';
    fillColor = "#ffffff";
      }

      if (currentLayer->getLayerType() == AI::YAAnnLayer::layerType_t::HIDDEN) {
    if (currentLayer->getNeuron(i)->isBiased()) {
      ofs << "\t" << nodeName << k << "_" << i << " [label=\"" << labelName
	//<< "|" << currentLayer->getNeuron(i)->getOutput() << "|"
          <<"\", style=\"filled\", fillcolor=\""<< fillColor <<"\"];\n";

    } else {
      ofs << "\t" << nodeName << k << "_" << i << " [label=<" << labelName << "<SUB>"
          << k << i
          <<"</SUB>>, style=\"filled\", fillcolor=\""<< fillColor <<"\"];\n";
    }
      }
      else {
    if (currentLayer->getNeuron(i)->isBiased()) {
      ofs << "\t" << nodeName << i << " [label=\"" << labelName
	//<< "|" << currentLayer->getNeuron(i)->getOutput() << "|"
	  << "\", style=\"filled\", fillcolor=\""<< fillColor <<"\"];\n";
    } else {
      ofs << "\t" << nodeName << i << " [label=<" << labelName << "<SUB>"
          << i << "</SUB>>, style=\"filled\", fillcolor=\""<< fillColor <<"\"];\n";

    }
      }
      for (int j = 0; j < currentLayer->getNeuron(i)->axonsNumber();
           ++j) {
        switch (currentLayer->getLayerType()) {
        case AI::YAAnnLayer::layerType_t::INPUT:
          ofs << "\tI_" << i;
          break;
        case AI::YAAnnLayer::layerType_t::OUTPUT:
          ofs << "\tO_" << i;
          break;
        case AI::YAAnnLayer::layerType_t::HIDDEN:
          ofs << "\tH_" << k << "_" << i;
          break;
        default:
          ofs << "\tUNKNOWN_" << i;
          break;
        }
        if (currentLayer->getNextLayer() != NULL) {
          switch (currentLayer->getNextLayer()->getLayerType()) {
          case AI::YAAnnLayer::layerType_t::INPUT:
        ofs << " -> I_" << j;
        break;
          case AI::YAAnnLayer::layerType_t::OUTPUT:
        ofs << " -> O_" << j;
        break;
          case AI::YAAnnLayer::layerType_t::HIDDEN:
        {
          if (currentLayer->getLayerType() == AI::YAAnnLayer::layerType_t::HIDDEN)
            ofs << " -> H_" << k + 1 << "_" << j;
          else
            ofs << " -> H_" << k << "_" << j;
        }

        break;
          default:
        ofs << " -> UNKNOWN_" << j;
        break;
          }
        }
        ofs << "[label = \"" << currentLayer->getNeuron(i)->getAxon(j)->getWeight()  << "\"];\n";
      }
    }
    if (currentLayer->getLayerType() == AI::YAAnnLayer::layerType_t::HIDDEN)
      ++k;
    currentLayer = currentLayer->getNextLayer();
      }
      ofs << "}\n";
      ofs.close();
    }

}

void AI::YAAnnReportStdFile::saveMdlToFile(const YAAnn& yaann, const char* fileName)
{
    std::ofstream ofs(fileName);
    if (ofs.is_open()) {
        ofs << yaann.getActFun()->name() << " " << yaann.getRand()->name() << "\n"
            << yaann.getInputLayer()->neuronsNumber()
            << " "
            << yaann.hiddenLayersNumber()
            << " "
            << yaann.getHiddenLayer(0)->neuronsNumber()
            << " "
            << yaann.getOutputLayer()->neuronsNumber()
            << "\n";
        AI::YAAnnLayer* currLayer = yaann.getInputLayer();
        while (currLayer != yaann.getOutputLayer()) {
            for (int i = 0; i < currLayer->neuronsNumber(); ++i) {
                ofs << currLayer->getNeuron(i)->isBiased() << " "
                    << currLayer->getNeuron(i)->getOutput() << " ";
                for (int j = 0; j < currLayer->getNeuron(i)->axonsNumber(); ++j) {
                    ofs << currLayer->getNeuron(i)->getAxon(j)->getWeight() << " ";
                }
                ofs << "\n";
            }
            currLayer = currLayer->getNextLayer();
        }
        ofs.close();
    }
}

void AI::YAAnnReportStdFile::readMdlFromFile(YAAnn& yaann, const char* fileName)
{
    std::ifstream ifs(fileName);
    if (ifs.is_open()) {
        int inputLayerSize;
        int hiddenLayersNumber;
        int hiddenLayerSize;
        int outputLayerSize;
        std::string actFuncName;
        std::string randName;
        ifs >> actFuncName >> randName
                >> inputLayerSize
                >> hiddenLayersNumber
                >> hiddenLayerSize
                >> outputLayerSize;
        yaann.destroyModel();
        if (randName == "norm") {
            yaann.setRand(new YAAnnRandNormDistr);
        }
        if (actFuncName == "sig") {
            yaann.setActFun(new YAAnnActFunSig);
        } else if (actFuncName == "hyp") {
            yaann.setActFun(new YAAnnActFunHyp);
        }
        yaann.setModel(inputLayerSize, hiddenLayersNumber, hiddenLayerSize, outputLayerSize, false);
        AI::YAAnnLayer* currLayer = yaann.getInputLayer();
        while (currLayer != yaann.getOutputLayer()) {
            for (int i = 0; i < currLayer->neuronsNumber(); ++i) {
                bool isBiased;
                double output;
                ifs >> isBiased >> output;
                currLayer->getNeuron(i)->setIsBiased(isBiased);
                currLayer->getNeuron(i)->setOutput(output);
		if (isBiased && currLayer->getPrevLayer() != NULL) {		  
		  currLayer->getPrevLayer()->disconnectFromNeuron(currLayer->getNeuron(i));		  
		}
                for (int j = 0; j < currLayer->getNeuron(i)->axonsNumber(); ++j) {
                    double weight = .0;
                    ifs >> weight;
                    currLayer->getNeuron(i)->getAxon(j)->setWeight(weight);
                }
            }
            currLayer = currLayer->getNextLayer();
        }
        ifs.close();
    }
}
