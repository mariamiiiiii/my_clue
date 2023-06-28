#ifndef CLUEAlgo_h
#define CLUEAlgo_h

// C/C++ headers
#include <chrono>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "Points.h"
#include "Tiles.h"

// The type T is used to pass the number of bins in each dimension and the
// allowed ranges spanned. Anchillary quantitied, like the inverse of the bin
// width should also be provided. Code will not compile if any such information
// is missing.
template <typename T, int NLAYERS>
class CLUEAlgo {
 public:
  CLUEAlgo(float dc, float kappa, float outlierDeltaFactor, bool verbose) {
    dc_ = dc;
    kappa_ = kappa;
    outlierDeltaFactor_ = outlierDeltaFactor;
    verbose_ = verbose;
    numberOfClusters_ = 0;
  }
  CLUEAlgo(){}
  ~CLUEAlgo() {}

  // public variables
  float dc_, kappa_, outlierDeltaFactor_;
  bool verbose_;

  int numberOfClusters_;
  int getNumberOfClusters(){return numberOfClusters_;}

  Points points_;

  int copyPoints(int n, float* x, float* y, int* layer, float* weight) {
    points_.clear();
    // Reserve, first
    points_.n = n;
    points_.x.resize(n);
    points_.y.resize(n);
    points_.layer.resize(n);
    points_.weight.resize(n);
    // Copy, next
    std::copy(x, x + n, std::begin(points_.x));
    std::copy(y, y + n, std::begin(points_.y));
    std::copy(layer, layer + n, std::begin(points_.layer));
    std::copy(weight, weight + n, std::begin(points_.weight));

    points_.p_x = points_.x.data();
    points_.p_y = points_.y.data();
    points_.p_layer = points_.layer.data();
    points_.p_weight = points_.weight.data();

    resizeOutputContainers();

    return points_.n;
  }
  int copyPoints(int n, float* x, float* y, int* layer, float* weight, float* sigmaNoise) {
    points_.clear();
    // Reserve, first
    points_.n = n;
    points_.x.resize(n);
    points_.y.resize(n);
    points_.layer.resize(n);
    points_.weight.resize(n);
    points_.sigmaNoise.resize(n);
    // Copy, next
    std::copy(x, x + n, std::begin(points_.x));
    std::copy(y, y + n, std::begin(points_.y));
    std::copy(layer, layer + n, std::begin(points_.layer));
    std::copy(weight, weight + n, std::begin(points_.weight));
    std::copy(sigmaNoise, sigmaNoise + n, std::begin(points_.sigmaNoise));

    points_.p_x = points_.x.data();
    points_.p_y = points_.y.data();
    points_.p_layer = points_.layer.data();
    points_.p_weight = points_.weight.data();
    points_.p_sigmaNoise = points_.sigmaNoise.data();

    resizeOutputContainers();

    return points_.n;
  }

  int copyPoints(const std::vector<float>& x, const std::vector<float>& y,
                 const std::vector<int>& layer,
                 const std::vector<float>& weight
                 ) {
    points_.clear();
    auto const n = x.size();
    points_.n = n;
    // input variables
    points_.x.resize(n);
    std::copy(std::begin(x), std::end(x), std::begin(points_.x));

    points_.y.resize(n);
    std::copy(std::begin(y), std::end(y), std::begin(points_.y));

    points_.layer.resize(n);
    std::copy(std::begin(layer), std::end(layer), std::begin(points_.layer));

    points_.weight.resize(n);
    std::copy(std::begin(weight), std::end(weight), std::begin(points_.weight));


    points_.p_x = points_.x.data();
    points_.p_y = points_.y.data();
    points_.p_layer = points_.layer.data();
    points_.p_weight = points_.weight.data();

    resizeOutputContainers();

    return points_.n;
  }
  int copyPoints(const std::vector<float>& x, const std::vector<float>& y,
                 const std::vector<int>& layer,
                 const std::vector<float>& weight,
                 const std::vector<float>& sigmaNoise
                 ) {
    points_.clear();
    auto const n = x.size();
    points_.n = n;
    // input variables
    points_.x.resize(n);
    std::copy(std::begin(x), std::end(x), std::begin(points_.x));

    points_.y.resize(n);
    std::copy(std::begin(y), std::end(y), std::begin(points_.y));

    points_.layer.resize(n);
    std::copy(std::begin(layer), std::end(layer), std::begin(points_.layer));

    points_.weight.resize(n);
    std::copy(std::begin(weight), std::end(weight), std::begin(points_.weight));

    points_.sigmaNoise.resize(n);
    std::copy(std::begin(sigmaNoise), std::end(sigmaNoise), std::begin(points_.sigmaNoise));

    points_.p_x = points_.x.data();
    points_.p_y = points_.y.data();
    points_.p_layer = points_.layer.data();
    points_.p_weight = points_.weight.data();
    points_.p_sigmaNoise = points_.sigmaNoise.data();

    resizeOutputContainers();

    return points_.n;
  }

  int setPoints(int n, const float* x, const float* y, const int* layer,
                const float* weight) {
    points_.clear();
    points_.n = n;
    points_.p_x = x;
    points_.p_y = y;
    points_.p_layer = layer;
    points_.p_weight = weight;

    resizeOutputContainers();

    return n;
  }
  int setPoints(int n, const float* x, const float* y, const int* layer,
                const float* weight, const float* sigmaNoise) {
    points_.clear();
    points_.n = n;
    points_.p_x = x;
    points_.p_y = y;
    points_.p_layer = layer;
    points_.p_weight = weight;
    points_.p_sigmaNoise = sigmaNoise;

    resizeOutputContainers();

    return n;
  }

  int setPoints(std::vector<float>& x, std::vector<float>& y,
                std::vector<int>& layer, std::vector<float>& weight) {
    // std::cout << "STANDALONE: start setPosition" << std::endl;
    points_.clear();
    auto const n = x.size();
    points_.n = n;
    // input variables
    points_.n = n;
    points_.x.swap(x);
    points_.y.swap(y);
    points_.layer.swap(layer);
    points_.weight.swap(weight);

    points_.p_x = points_.x.data();
    points_.p_y = points_.y.data();
    points_.p_layer = points_.layer.data();
    points_.p_weight = points_.weight.data();

    resizeOutputContainers();

    // std::cout << "STANDALONE: end setPosition" << std::endl;
    return points_.n;

  }
  int setPoints(std::vector<float>& x, std::vector<float>& y,
                std::vector<int>& layer, std::vector<float>& weight, std::vector<float>& sigmaNoise) {
    // std::cout << "STANDALONE: start setPosition" << std::endl;
    points_.clear();
    auto const n = x.size();
    points_.n = n;
    // input variables
    points_.n = n;
    points_.x.swap(x);
    points_.y.swap(y);
    points_.layer.swap(layer);
    points_.weight.swap(weight);
    points_.sigmaNoise.swap(sigmaNoise);

    points_.p_x = points_.x.data();
    points_.p_y = points_.y.data();
    points_.p_layer = points_.layer.data();
    points_.p_weight = points_.weight.data();
    points_.p_sigmaNoise = points_.sigmaNoise.data();

    resizeOutputContainers();

    // std::cout << "STANDALONE: end setPosition" << std::endl;
    return points_.n;

  }

  int getPoints(std::vector<float>& delta, std::vector<int>& nearestHigher,
                std::vector<int>& clusterIndex, std::vector<float>& rho, std::vector<uint8_t>& isSeed, std::vector<int>& layer, std::vector<float>& weight) {

    delta.swap(points_.delta);
    nearestHigher.swap(points_.nearestHigher);
    clusterIndex.swap(points_.clusterIndex);
    rho.swap(points_.rho);
    isSeed.swap(points_.isSeed);
    layer.swap(points_.layer);
    weight.swap(points_.weight);
    return points_.n;
  }

  void clearPoints() { points_.clear(); }

  void makeClusters();

  void infoSeeds();
  void infoHits();

  void verboseResults(const std::string& outputFileName = "cout",
                      const unsigned nVerbose = -1) const;

 private:
  // private member methods
  void prepareDataStructures(std::array<Tiles<T>, NLAYERS>&);
  void calculateLocalDensity(std::array<Tiles<T>, NLAYERS>&);
  void calculateDistanceToHigher(std::array<Tiles<T>, NLAYERS>&);
  void findAndAssignClusters();
  inline float distance(int, int) const;
  void resizeOutputContainers();
};

template <typename T, int NLAYERS>
void CLUEAlgo<T, NLAYERS>::resizeOutputContainers() {
  // result variables
  points_.rho.resize(points_.n, 0);
  points_.delta.resize(points_.n, std::numeric_limits<float>::max());
  points_.nearestHigher.resize(points_.n, -1);
  points_.followers.resize(points_.n);
  points_.clusterIndex.resize(points_.n, -1);
  points_.isSeed.resize(points_.n, 0);
}

template <typename T, int NLAYERS>
void CLUEAlgo<T, NLAYERS>::verboseResults(
    const std::string& outputFileName /* "cout" */,
    const unsigned nVerbose /* -1 */) const {
  if (!verbose_) return;

  unsigned int to_print = (nVerbose == -1 ? points_.n : nVerbose);

  std::string header(
      "index, x, y, layer, weight, rho, delta, nh, isSeed, clusterId\n");
  std::string s;
  char buffer[100];
  for (unsigned i = 0; i < to_print; i++) {
    snprintf(
        buffer, 100, "%d, %5.3f, %5.3f, %d, %5.3f, %5.3f, %5.3g, %d, %d, %d\n",
        i, points_.p_x[i], points_.p_y[i], points_.p_layer[i],
        points_.p_weight[i], points_.rho[i], points_.delta[i],
        points_.nearestHigher[i], points_.isSeed[i], points_.clusterIndex[i]);
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

template <typename T, int NLAYERS>
void CLUEAlgo<T, NLAYERS>::makeClusters() {
  std::array<Tiles<T>, NLAYERS> allLayerTiles;
  // std::cout << "STANDALONE: start makeClusters.." << std::endl;
  // start clustering
  auto start = std::chrono::high_resolution_clock::now();

  // std::cout << "STANDALONE: before prepare" << std::endl;
  prepareDataStructures(allLayerTiles);
  // std::cout << "STANDALONE: after prepare datastructures  makeClusters" << std::endl;
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "--- prepareDataStructures:     " << elapsed.count() * 1000
            << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  calculateLocalDensity(allLayerTiles);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- calculateLocalDensity:     " << elapsed.count() * 1000
            << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  calculateDistanceToHigher(allLayerTiles);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- calculateDistanceToHigher: " << elapsed.count() * 1000
            << " ms\n";


  findAndAssignClusters();
  // std::cout << "STANDALONE: end makeClusters" << std::endl;
}

template <typename T, int NLAYERS>
void CLUEAlgo<T, NLAYERS>::prepareDataStructures(
    std::array<Tiles<T>, NLAYERS>& allLayerTiles) {
  for (int i = 0; i < points_.n; i++) {
    // push index of points into tiles
    allLayerTiles[points_.p_layer[i]].fill(points_.p_x[i], points_.p_y[i], i);
  }
}

template <typename T, int NLAYERS>
void CLUEAlgo<T, NLAYERS>::calculateLocalDensity(
    std::array<Tiles<T>, NLAYERS>& allLayerTiles) {
  // loop over all points
  for (int i = 0; i < points_.n; i++) {
    Tiles<T>& lt = allLayerTiles[points_.p_layer[i]];

    // get search box
    std::array<int, 4> search_box =
        lt.searchBox(points_.p_x[i] - dc_, points_.p_x[i] + dc_,
                     points_.p_y[i] - dc_, points_.p_y[i] + dc_);

    // loop over bins in the search box
    for (int xBin = search_box[0]; xBin < search_box[1] + 1; ++xBin) {
      for (int yBin = search_box[2]; yBin < search_box[3] + 1; ++yBin) {
        // get the id of this bin
        int binId = lt.getGlobalBinByBin(xBin, yBin);
        // get the size of this bin
        int binSize = lt[binId].size();

        // iterate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = lt[binId][binIter];
          // query N_{dc_}(i)
          float dist_ij = distance(i, j);
          if (dist_ij <= dc_) {
            // sum weights within N_{dc_}(i)
            points_.rho[i] += (i == j ? 1.f : 0.5f) * points_.p_weight[j];
          }
        }  // end of interate inside this bin
      }
    }  // end of loop over bins in search box
  }    // end of loop over points
}

template <typename T, int NLAYERS>
void CLUEAlgo<T, NLAYERS>::calculateDistanceToHigher(
    std::array<Tiles<T>, NLAYERS>& allLayerTiles) {
  // loop over all points
  float dm = outlierDeltaFactor_ * dc_;
  for (int i = 0; i < points_.n; i++) {
    // default values of delta and nearest higher for i
    float delta_i = std::numeric_limits<float>::max();
    int nearestHigher_i = -1;
    float xi = points_.p_x[i];
    float yi = points_.p_y[i];
    float rho_i = points_.rho[i];

    // get search box
    Tiles<T>& lt = allLayerTiles[points_.p_layer[i]];
    std::array<int, 4> search_box =
        lt.searchBox(xi - dm, xi + dm, yi - dm, yi + dm);

    // loop over all bins in the search box
    for (int xBin = search_box[0]; xBin < search_box[1] + 1; ++xBin) {
      for (int yBin = search_box[2]; yBin < search_box[3] + 1; ++yBin) {
        // get the id of this bin
        int binId = lt.getGlobalBinByBin(xBin, yBin);
        // get the size of this bin
        int binSize = lt[binId].size();

        // interate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = lt[binId][binIter];
          // query N'_{dm}(i)
          bool foundHigher = (points_.rho[j] > rho_i);
          // in the rare case where rho is the same, use detid
          foundHigher = foundHigher || ((points_.rho[j] == rho_i) && (j > i));
          float dist_ij = distance(i, j);
          if (foundHigher && dist_ij <= dm) {  // definition of N'_{dm}(i)
            // find the nearest point within N'_{dm}(i)
            if (dist_ij < delta_i) {
              // update delta_i and nearestHigher_i
              delta_i = dist_ij;
              nearestHigher_i = j;
            }
          }
        }  // end of interate inside this bin
      }
    }  // end of loop over bins in search box

    points_.delta[i] = delta_i;
    points_.nearestHigher[i] = nearestHigher_i;
  }  // end of loop over points
}

template <typename T, int NLAYERS>
void CLUEAlgo<T, NLAYERS>::findAndAssignClusters() {
  auto start = std::chrono::high_resolution_clock::now();

  int nClusters = 0;

  // find cluster seeds and outlier
  std::vector<int> localStack;
  // loop over all points
  std::cout << "Number of points: " << points_.n << std::endl;
  for (int i = 0; i < points_.n; i++) {
    // initialize clusterIndex
    points_.clusterIndex[i] = -1;

    float deltai = points_.delta[i];
    float rhoi = points_.rho[i];

    // determine seed or outlier
    float rhoc = points_.sigmaNoise[i] * kappa_;
    bool isSeed = (deltai > dc_) and (rhoi >= rhoc);
    bool isOutlier = (deltai > outlierDeltaFactor_ * dc_) and (rhoi < rhoc);
    if (isSeed) {
      // set isSeed as 1
      points_.isSeed[i] = 1;
      // set cluster id
      points_.clusterIndex[i] = nClusters;
      // increment number of clusters
      nClusters++;
      // add seed into local stack
      localStack.push_back(i);
    } else if (!isOutlier) {
      // register as follower at its nearest higher
      points_.followers[points_.nearestHigher[i]].push_back(i);
    }
  }
  numberOfClusters_ = nClusters;

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "--- findSeedAndFollowers:      " << elapsed.count() * 1000
            << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  // expend clusters from seeds
  while (!localStack.empty()) {
    int i = localStack.back();
    auto& followers = points_.followers[i];
    localStack.pop_back();

    // loop over followers
    for (int j : followers) {
      // pass id from i to a i's follower
      points_.clusterIndex[j] = points_.clusterIndex[i];
      // push this follower to localStack
      localStack.push_back(j);
    }
  }
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- assignClusters:            " << elapsed.count() * 1000
            << " ms\n";
}

template <typename T, int NLAYERS>
inline float CLUEAlgo<T, NLAYERS>::distance(int i, int j) const {
  // 2-d distance on the layer

  // assert(points_.p_layer[i] == points_.p_layer[j]);
  const float dx = points_.p_x[i] - points_.p_x[j];
  const float dy = points_.p_y[i] - points_.p_y[j];
  return std::sqrt(dx * dx + dy * dy);
}

#endif
