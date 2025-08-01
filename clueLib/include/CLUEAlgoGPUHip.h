// -------- 1. bring in the HIP runtime --------------------------------------
// -------- 2. map the CUDA symbols used in the code to HIP ------------------
// -------- 3. keep the existing error macro but rename it -------------------
// -------- 4. pull in the body that still uses CUDA names -------------------

#ifndef CLUEAlgoGPUHip_h
#define CLUEAlgoGPUHip_h
#include <math.h>

#include <iostream>
#include <limits>

// GPU Add
#include <hip/hip_runtime.h>
// for timing
#include <chrono>
#include <ctime>

#include "CLUEAlgo.h"
#include "TilesGPU.h"

#define CHECK_HIP_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file, const int line)
{
    if (err != hipSuccess)
    {
        std::cerr << "HIP Runtime Error at: " << file << ":" << line << "\n"
                  << hipGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

class WorkDivByPoints;
class WorkDivByTile;

static const int maxNSeeds = 100000;
static const int maxNFollowers = 32;
static const int localStackSizePerSeed = 32;

struct PointsPtr {

  int n;

  float *x;
  float *y;
  int *layer;
  float *weight;
  float *sigmaNoise;

  float *rho;
  float *delta;
  unsigned int *nearestHigher;
  int *clusterIndex;
  uint8_t *isSeed;
};

// The type T is used to pass the number of bins in each dimension and the
// allowed ranges spanned. Anchillary quantitied, like the inverse of the bin
// width should also be provided. Code will not compile if any such information
// is missing.
template <typename T, int NLAYERS, typename W=WorkDivByPoints>
class CLUEAlgoGPU : public CLUEAlgo<T, NLAYERS> {
  // inherit from CLUEAlgo

 public:
  // constructor
  CLUEAlgoGPU(float dc, float rhoc, float outlierDeltaFactor, bool verbose,
      int n, float* x, float* y, int* layer, float* weight, 
    float* rho, float* delta, unsigned int* nearestHigher, int* clusterIndex, uint8_t* isSeed, bool useAbsoluteSigma=false)
      : CLUEAlgo<T, NLAYERS>(dc, rhoc, outlierDeltaFactor, verbose, useAbsoluteSigma) {
    init_device(n, x, y, layer, weight, rho, delta, nearestHigher, clusterIndex, isSeed);
  }
  // destructor
  ~CLUEAlgoGPU() { free_device(); }

  void copy_todevice() {
    int gpuId;
    CHECK_HIP_ERROR(hipGetDevice(&gpuId));

    CHECK_HIP_ERROR(
      hipMemPrefetchAsync(points_gpu.x, sizeof(float) * points_gpu.n, gpuId, stream_));
    CHECK_HIP_ERROR(
      hipMemPrefetchAsync(points_gpu.y, sizeof(float) * points_gpu.n, gpuId, stream_));
    CHECK_HIP_ERROR(         
      hipMemPrefetchAsync(points_gpu.layer, sizeof(int) * points_gpu.n, gpuId, stream_));
    CHECK_HIP_ERROR(
      hipMemPrefetchAsync(points_gpu.weight, sizeof(float) * points_gpu.n, gpuId, stream_)); 
    if (useAbsoluteSigma_)
      CHECK_HIP_ERROR(
        hipMemPrefetchAsync(points_gpu.sigmaNoise, sizeof(float) * points_gpu.n, gpuId, stream_));
  } 

  void copy_tohost() {
    //prefetch just for the output
    CHECK_HIP_ERROR(
      hipMemPrefetchAsync(points_gpu.rho, sizeof(float) * points_gpu.n, hipCpuDeviceId, stream_));
    CHECK_HIP_ERROR(
      hipMemPrefetchAsync(points_gpu.delta, sizeof(float) * points_gpu.n, hipCpuDeviceId, stream_));
    CHECK_HIP_ERROR(         
      hipMemPrefetchAsync(points_gpu.nearestHigher, sizeof(int) * points_gpu.n, hipCpuDeviceId, stream_));
    CHECK_HIP_ERROR(
      hipMemPrefetchAsync(points_gpu.clusterIndex, sizeof(int) * points_gpu.n, hipCpuDeviceId, stream_)); 
    CHECK_HIP_ERROR(
      hipMemPrefetchAsync(points_gpu.isSeed, sizeof(uint8_t) * points_gpu.n, hipCpuDeviceId, stream_));
  }


  // public methods
  void makeClusters();  // overwrite base class

  void Sync();

  // Bring base-class public variables into the scope of this template derived
  // class
  using CLUEAlgo<T, NLAYERS>::dc_;
  using CLUEAlgo<T, NLAYERS>::rhoc_;
  using CLUEAlgo<T, NLAYERS>::kappa_;
  using CLUEAlgo<T, NLAYERS>::outlierDeltaFactor_;
  using CLUEAlgo<T, NLAYERS>::verbose_;
  using CLUEAlgo<T, NLAYERS>::points_;
  using CLUEAlgo<T, NLAYERS>::useAbsoluteSigma_;

 private:
  // private variables
  
  hipStream_t stream_;
  // algorithm internal variables
  PointsPtr points_gpu;
  TilesGPU<T> *d_hist;
  GPU::VecArray<int, maxNSeeds> *d_seeds;
  GPU::VecArray<int, maxNFollowers> *d_followers;

  // private methods
  void init_device(int n, float* x, float* y, int* layer, float* weight,
        float* rho, float* delta, unsigned int* nearestHigher, int* clusterIndex, uint8_t* isSeed) {
    // Create our own hip stream
    CHECK_HIP_ERROR(hipStreamCreate(&stream_));
    // Allocate memory
    //unsigned int reserve = 10000;

    points_gpu.x = x; 
    points_gpu.y = y;
    points_gpu.layer = layer;
    points_gpu.weight = weight;
    points_gpu.n = n;

    points_gpu.rho = rho; 
    points_gpu.delta = delta;
    points_gpu.nearestHigher = nearestHigher;
    points_gpu.clusterIndex = clusterIndex;
    points_gpu.isSeed = isSeed; //for output

    // algorithm internal variables
    CHECK_HIP_ERROR(
      hipMallocAsync(&d_hist, sizeof(TilesGPU<T>) * NLAYERS, stream_));
    CHECK_HIP_ERROR(hipMallocAsync(
        &d_seeds, sizeof(GPU::VecArray<int, maxNSeeds>), stream_));
    CHECK_HIP_ERROR(hipMallocAsync(
        &d_followers, sizeof(GPU::VecArray<int, maxNFollowers>) * points_gpu.n,
        stream_));
  }

  void free_device() {
    // algorithm internal variables
    CHECK_HIP_ERROR(hipFreeAsync(d_hist, stream_));
    CHECK_HIP_ERROR(hipFreeAsync(d_seeds, stream_));
    CHECK_HIP_ERROR(hipFreeAsync(d_followers, stream_));

    CHECK_HIP_ERROR(hipStreamDestroy(stream_));
  }


  void clear_internal_buffers() {
// result variables
    CHECK_HIP_ERROR(hipMemsetAsync(points_gpu.rho, 0x00,
                                     sizeof(float) * points_gpu.n, stream_));
    CHECK_HIP_ERROR(hipMemsetAsync(points_gpu.delta, 0x00,
                                     sizeof(float) * points_gpu.n, stream_));
    CHECK_HIP_ERROR(hipMemsetAsync(points_gpu.nearestHigher, 0x00,
                                     sizeof(int) * points_gpu.n, stream_));
    CHECK_HIP_ERROR(hipMemsetAsync(points_gpu.clusterIndex, 0x00,
                                     sizeof(int) * points_gpu.n, stream_));
    CHECK_HIP_ERROR(hipMemsetAsync(points_gpu.isSeed, 0x00,
                                     sizeof(uint8_t) * points_gpu.n, stream_));
    //algorithm internal variables
  CHECK_HIP_ERROR(
    hipMemsetAsync(d_hist, 0x00, sizeof(TilesGPU<T>) * NLAYERS, stream_));
  CHECK_HIP_ERROR(
    hipMemsetAsync(d_seeds, 0x00, sizeof(GPU::VecArray<int, maxNSeeds>), stream_));
  CHECK_HIP_ERROR(
    hipMemsetAsync(d_followers, 0x00, sizeof(GPU::VecArray<int, maxNFollowers>) * points_gpu.n, stream_));
  }

};

template <typename T>
__global__ void kernel_compute_histogram(TilesGPU<T> *d_hist,
                                         const PointsPtr points_gpu,
                                         int numberOfPoints) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numberOfPoints) {
    // push index of points into tiles
    d_hist[points_gpu.layer[i]].fill(points_gpu.x[i], points_gpu.y[i], i);
  }
}  // kernel kernel_compute_histogram

template <typename T>
__global__ void kernel_calculate_densityTile(TilesGPU<T> *d_hist,
                                         PointsPtr points_gpu, float dc,
                                         int numberOfPoints) {
  int layeri = blockIdx.y;
  int globalBinOnLayer = blockIdx.x;
  int bin = threadIdx.x;

  if (bin < d_hist[layeri][globalBinOnLayer].size()) {
    int i = d_hist[layeri][globalBinOnLayer][bin];
    float rhoi{0.};
    int layeri = points_gpu.layer[i];
    float xi = points_gpu.x[i];
    float yi = points_gpu.y[i];

    // get search box
    int4 search_box =
        d_hist[layeri].searchBox(xi - dc, xi + dc, yi - dc, yi + dc);

    // loop over bins in the search box
    for (int xBin = search_box.x; xBin < search_box.y + 1; ++xBin) {
      for (int yBin = search_box.z; yBin < search_box.w + 1; ++yBin) {
        // get the id of this bin
        int binId = d_hist[layeri].getGlobalBinByBin(xBin, yBin);
        // get the size of this bin
        int binSize = d_hist[layeri][binId].size();

        // interate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = d_hist[layeri][binId][binIter];
          // query N_{dc_}(i)
          float xj = points_gpu.x[j];
          float yj = points_gpu.y[j];
          float dist_ij_2 =
              ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
          rhoi += (dist_ij_2 <= dc*dc) * (i == j ? 1.f : 0.5f) * points_gpu.weight[j];
          /*
          if (dist_ij_2 <= dc*dc) {
            // sum weights within N_{dc_}(i)
            rhoi += (i == j ? 1.f : 0.5f) * points_gpu.weight[j];
          }
          */
        }  // end of interate inside this bin
      }
    }  // end of loop over bins in search box
    points_gpu.rho[i] = rhoi;
  }
}  // kernel kernel_calculate_density

template <typename T>
__global__ void kernel_calculate_density(TilesGPU<T> *d_hist,
                                         PointsPtr points_gpu, float dc,
                                         int numberOfPoints) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numberOfPoints) {
    float rhoi{0.};
    int layeri = points_gpu.layer[i];
    float xi = points_gpu.x[i];
    float yi = points_gpu.y[i];

    // get search box
    int4 search_box =
        d_hist[layeri].searchBox(xi - dc, xi + dc, yi - dc, yi + dc);

    // loop over bins in the search box
    for (int xBin = search_box.x; xBin < search_box.y + 1; ++xBin) {
      for (int yBin = search_box.z; yBin < search_box.w + 1; ++yBin) {
        // get the id of this bin
        int binId = d_hist[layeri].getGlobalBinByBin(xBin, yBin);
        // get the size of this bin
        int binSize = d_hist[layeri][binId].size();

        // interate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = d_hist[layeri][binId][binIter];
          // query N_{dc_}(i)
          float xj = points_gpu.x[j];
          float yj = points_gpu.y[j];
          float dist_ij_2 =
              ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
          rhoi += (dist_ij_2 <= dc*dc) * (i == j ? 1.f : 0.5f) * points_gpu.weight[j];
          /*
          if (dist_ij_2 <= dc*dc) {
            // sum weights within N_{dc_}(i)
            rhoi += (i == j ? 1.f : 0.5f) * points_gpu.weight[j];
          }
          */
        }  // end of interate inside this bin
      }
    }  // end of loop over bins in search box
    points_gpu.rho[i] = rhoi;
  }
}  // kernel kernel_calculate_density

template <typename T>
__global__ void kernel_calculate_distanceToHigher(TilesGPU<T> *d_hist,
                                                  PointsPtr points_gpu,
                                                  float outlierDeltaFactor,
                                                  float dc, int numberOfPoints) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  const float dm = outlierDeltaFactor * dc;

  if (i < numberOfPoints) {
    int layeri = points_gpu.layer[i];

    float deltai = std::numeric_limits<float>::max();
    int nearestHigheri = -1;
    float xi = points_gpu.x[i];
    float yi = points_gpu.y[i];
    float rhoi = points_gpu.rho[i];

    // get search box
    int4 search_box =
        d_hist[layeri].searchBox(xi - dm, xi + dm, yi - dm, yi + dm);

    // loop over all bins in the search box
    for (int xBin = search_box.x; xBin < search_box.y + 1; ++xBin) {
      for (int yBin = search_box.z; yBin < search_box.w + 1; ++yBin) {
        // get the id of this bin
        int binId = d_hist[layeri].getGlobalBinByBin(xBin, yBin);
        // get the size of this bin
        int binSize = d_hist[layeri][binId].size();

        // interate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = d_hist[layeri][binId][binIter];
          // query N'_{dm}(i)
          float xj = points_gpu.x[j];
          float yj = points_gpu.y[j];
          float dist_ij_2 =
              ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
          bool foundHigher = (points_gpu.rho[j] > rhoi);
          // in the rare case where rho is the same, use detid
          foundHigher = foundHigher || ((points_gpu.rho[j] == rhoi) && (j > i));
          if (foundHigher && dist_ij_2 <= dm*dm) {  // definition of N'_{dm}(i)
            // find the nearest point within N'_{dm}(i)
            if (dist_ij_2 < deltai) {
              // update deltai and nearestHigheri
              deltai = dist_ij_2;
              nearestHigheri = j;
            }
          }
        }  // end of interate inside this bin
      }
    }  // end of loop over bins in search box
    points_gpu.delta[i] = std::sqrt(deltai);
    points_gpu.nearestHigher[i] = nearestHigheri;
  }
}  // kernel kernel_calculate_distanceToHigher

template <typename T>
__global__ void kernel_calculate_distanceToHigherTile(TilesGPU<T> *d_hist,
                                                  PointsPtr points_gpu,
                                                  float outlierDeltaFactor,
                                                  float dc, int numberOfPoints) {
  int layeri = blockIdx.y;
  int globalBinOnLayer = blockIdx.x;
  int bin = threadIdx.x;

  if (bin < d_hist[layeri][globalBinOnLayer].size()) {
    const float dm = outlierDeltaFactor * dc;

    int i = d_hist[layeri][globalBinOnLayer][bin];
    float deltai = std::numeric_limits<float>::max();
    int nearestHigheri = -1;
    float xi = points_gpu.x[i];
    float yi = points_gpu.y[i];
    float rhoi = points_gpu.rho[i];

    // get search box
    int4 search_box =
        d_hist[layeri].searchBox(xi - dm, xi + dm, yi - dm, yi + dm);

    // loop over all bins in the search box
    for (int xBin = search_box.x; xBin < search_box.y + 1; ++xBin) {
      for (int yBin = search_box.z; yBin < search_box.w + 1; ++yBin) {
        // get the id of this bin
        int binId = d_hist[layeri].getGlobalBinByBin(xBin, yBin);
        // get the size of this bin
        int binSize = d_hist[layeri][binId].size();

        // interate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = d_hist[layeri][binId][binIter];
          // query N'_{dm}(i)
          float xj = points_gpu.x[j];
          float yj = points_gpu.y[j];
          float dist_ij_2 =
              ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
          bool foundHigher = (points_gpu.rho[j] > rhoi);
          // in the rare case where rho is the same, use detid
          foundHigher = foundHigher || ((points_gpu.rho[j] == rhoi) && (j > i));
          if (foundHigher && dist_ij_2 <= dm*dm) {  // definition of N'_{dm}(i)
            // find the nearest point within N'_{dm}(i)
            if (dist_ij_2 < deltai) {
              // update deltai and nearestHigheri
              deltai = dist_ij_2;
              nearestHigheri = j;
            }
          }
        }  // end of interate inside this bin
      }
    }  // end of loop over bins in search box
    points_gpu.delta[i] = std::sqrt(deltai);
    points_gpu.nearestHigher[i] = nearestHigheri;
  }
}  // kernel kernel_calculate_distanceToHigher

__global__ void kernel_find_clusters(
    GPU::VecArray<int, maxNSeeds> *d_seeds,
    GPU::VecArray<int, maxNFollowers> *d_followers, PointsPtr points_gpu,
    float outlierDeltaFactor, float dc, float rhoc, int numberOfPoints) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numberOfPoints) {
    // initialize clusterIndex
    points_gpu.clusterIndex[i] = -1;
    // determine seed or outlier
    float deltai = points_gpu.delta[i];
    float rhoi = points_gpu.rho[i];
    bool isSeed = (deltai > dc) && (rhoi >= rhoc);
    bool isOutlier = (deltai > outlierDeltaFactor * dc) && (rhoi < rhoc);

    if (isSeed) {
      // set isSeed as 1
      points_gpu.isSeed[i] = 1;
      d_seeds[0].push_back(i);  // head of d_seeds
    } else {
      if (!isOutlier) {
        assert(points_gpu.nearestHigher[i] < numberOfPoints);
        // register as follower at its nearest higher
        d_followers[points_gpu.nearestHigher[i]].push_back(i);
      }
    }
  }
}  // kernel kernel_find_clusters 

__global__ void kernel_find_clusters_kappa(
    GPU::VecArray<int, maxNSeeds> *d_seeds,
    GPU::VecArray<int, maxNFollowers> *d_followers, PointsPtr points_gpu,
    float outlierDeltaFactor, float dc, float kappa, int numberOfPoints) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numberOfPoints) {
    // initialize clusterIndex
    points_gpu.clusterIndex[i] = -1;
    // determine seed or outlier
    float deltai = points_gpu.delta[i];
    float rhoi = points_gpu.rho[i];
    float rhoc = points_gpu.sigmaNoise[i] * kappa;
    bool isSeed = (deltai > dc) && (rhoi >= rhoc);
    bool isOutlier = (deltai > outlierDeltaFactor * dc) && (rhoi < rhoc);

    if (isSeed) {
      // set isSeed as 1
      points_gpu.isSeed[i] = 1;
      d_seeds[0].push_back(i);  // head of d_seeds
    } else {
      if (!isOutlier) {
        assert(points_gpu.nearestHigher[i] < numberOfPoints);
        // register as follower at its nearest higher
        d_followers[points_gpu.nearestHigher[i]].push_back(i);
      }
    }
  }
}  // kernel kernel_find_clusters

__global__ void kernel_assign_clusters(
    const GPU::VecArray<int, maxNSeeds> *d_seeds,
    const GPU::VecArray<int, maxNFollowers> *d_followers, PointsPtr points_gpu,
    int numberOfPoints) {
  int idxCls = blockIdx.x * blockDim.x + threadIdx.x;
  const auto &seeds = d_seeds[0];
  const auto nSeeds = seeds.size();
  if (idxCls < nSeeds) {
    int localStack[localStackSizePerSeed] = {-1};
    int localStackSize = 0;

    // asgine cluster to seed[idxCls]
    int idxThisSeed = seeds[idxCls];
    points_gpu.clusterIndex[idxThisSeed] = idxCls;
    // push_back idThisSeed to localStack
    localStack[localStackSize] = idxThisSeed;
    localStackSize++;
    // process all elements in localStack
    while (localStackSize > 0) {
      // get last element of localStack
      int idxEndOflocalStack = localStack[localStackSize - 1];

      int temp_clusterIndex = points_gpu.clusterIndex[idxEndOflocalStack];
      // pop_back last element of localStack
      localStack[localStackSize - 1] = -1;
      localStackSize--;

      // loop over followers of last element of localStack
      for (int j : d_followers[idxEndOflocalStack]) {
        // // pass id to follower
        points_gpu.clusterIndex[j] = temp_clusterIndex;
        // push_back follower to localStack
        localStack[localStackSize] = j;
        localStackSize++;
      }
    }
  }
}  // kernel kernel_assign_clusters

template <typename T, int NLAYERS, typename W>
void CLUEAlgoGPU<T, NLAYERS, W>::makeClusters() {
  //copy_todevice();
  clear_internal_buffers();

  ////////////////////////////////////////////
  // calculate rho, delta and find seeds
  // 1 point per thread
  ////////////////////////////////////////////
  const dim3 blockSize(64, 1, 1);
  const dim3 gridSize(ceil(points_.n / static_cast<float>(blockSize.x)), 1, 1);
  kernel_compute_histogram<T>
      <<<gridSize, blockSize, 0, stream_>>>(d_hist, points_gpu, points_.n);

  if constexpr (std::is_same_v<W, WorkDivByPoints>) {
    kernel_calculate_density<T>
      <<<gridSize, blockSize, 0, stream_>>>(d_hist, points_gpu, dc_, points_.n);
    kernel_calculate_distanceToHigher<T><<<gridSize, blockSize, 0, stream_>>>(
        d_hist, points_gpu, outlierDeltaFactor_, dc_, points_.n);
  }

  if constexpr (std::is_same_v<W, WorkDivByTile>) {
    const dim3 gridSizeTile(T::nTiles, NLAYERS, 1);
    const dim3 blockSizeTile(T::maxTileDepth, 1, 1);
    kernel_calculate_densityTile<T>
      <<<gridSizeTile, blockSizeTile, 0, stream_>>>(d_hist, points_gpu, dc_, points_.n);
    kernel_calculate_distanceToHigherTile<T><<<gridSizeTile, blockSizeTile, 0, stream_>>>(
        d_hist, points_gpu, outlierDeltaFactor_, dc_, points_.n);
  }

  if (!useAbsoluteSigma_) {
    kernel_find_clusters<<<gridSize, blockSize, 0, stream_>>>(d_seeds, d_followers, points_gpu,
                                                  outlierDeltaFactor_, dc_, rhoc_,
                                                  points_.n);
  } else {
    kernel_find_clusters_kappa<<<gridSize, blockSize, 0, stream_>>>(d_seeds, d_followers, points_gpu,
                                                        outlierDeltaFactor_, dc_, kappa_,
                                                        points_.n);
  }

  ////////////////////////////////////////////
  // assign clusters
  // 1 point per seeds
  ////////////////////////////////////////////
  const dim3 gridSize_nseeds(ceil(maxNSeeds / 1024.0), 1, 1);
  kernel_assign_clusters<<<gridSize_nseeds, blockSize, 0, stream_>>>(
      d_seeds, d_followers, points_gpu, points_.n);

  //copy_tohost();
  ///
  }


template <typename T, int NLAYERS, typename W>
void CLUEAlgoGPU<T, NLAYERS, W>::Sync() {
  CHECK_HIP_ERROR(hipStreamSynchronize(stream_));
}


#endif
