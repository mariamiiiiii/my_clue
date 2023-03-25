#ifndef CLUEAlgo_h
#define CLUEAlgo_h

// C/C++ headers
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "LayerTiles.h"
#include "Points.h"

class CLUEAlgo {
 public:
  CLUEAlgo(float dc, float rhoc, float outlierDeltaFactor, bool verbose) {
    dc_ = dc;
    rhoc_ = rhoc;
    outlierDeltaFactor_ = outlierDeltaFactor;
    verbose_ = verbose;
  }
  ~CLUEAlgo() {}

  // public variables
  float dc_, rhoc_, outlierDeltaFactor_;
  bool verbose_;

  Points points_;

  int setPoints(int n, float* x, float* y, int* layer, float* weight) {
    points_.clear();
    // input variables
    for (int i = 0; i < n; ++i) {
      points_.x.push_back(x[i]);
      points_.y.push_back(y[i]);
      points_.layer.push_back(layer[i]);
      points_.weight.push_back(weight[i]);
    }

    points_.n = points_.x.size();

    // result variables
    points_.rho.resize(points_.n, 0);
    points_.delta.resize(points_.n, std::numeric_limits<float>::max());
    points_.nearestHigher.resize(points_.n, -1);
    points_.followers.resize(points_.n);
    points_.clusterIndex.resize(points_.n, -1);
    points_.isSeed.resize(points_.n, 0);

    return points_.n;
  }

  void clearPoints() { points_.clear(); }

  void makeClusters();

  void infoSeeds();
  void infoHits();

  void verboseResults(const std::string& outputFileName = "cout",
      const unsigned nVerbose = -1) const {

    if (!verbose_)
      return;

    unsigned int to_print = (nVerbose == -1 ? points_.n : nVerbose);

    std::string header("index, x, y, layer, weight, rho, delta, nh, isSeed, clusterId\n");
    std::string s;
    char buffer[100];
    for (unsigned i = 0; i < to_print; i++) {
      snprintf(buffer, 100, "%d, %5.3f, %5.3f, %d, %5.3f, %5.3f, %5.3g, %d, %d, %d\n",
          i, points_.x[i], points_.y[i], points_.layer[i], points_.weight[i],
          points_.rho[i], points_.delta[i], points_.nearestHigher[i],
          points_.isSeed[i], points_.clusterIndex[i]);
      s += buffer;
    }

    if (outputFileName == "cout")  // verbose to screen
      std::cout << header << s << std::endl;
    else {  // verbose to file
      std::ofstream outfile(outputFileName);
      if (outfile.is_open()) {
        outfile << header;
        outfile << s;
        outfile.close();
      } else {
        std::cerr << "Error: Unable to open file " << outputFileName << std::endl;
      }
    }
  }

 private:
  // private member methods
  void prepareDataStructures(std::array<LayerTiles, NLAYERS>&);
  void calculateLocalDensity(std::array<LayerTiles, NLAYERS>&);
  void calculateDistanceToHigher(std::array<LayerTiles, NLAYERS>&);
  void findAndAssignClusters();
  inline float distance(int, int) const;
};

#endif
