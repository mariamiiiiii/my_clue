#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <string>

#include "CLUEAlgo.h"
#include "CUDAEssentials.h"

#if defined(USE_ALPAKA)
#include "CLUEAlgoAlpaka.h"
#else
#include "CLUEAlgoGPU.h"
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#include <tbb/global_control.h>
#endif

#define NLAYERS 100

using namespace std;

void exclude_stats_outliers(std::vector<float> &v) {
  if (v.size() == 1)
    return;
  float mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  float sum_sq_diff =
      std::accumulate(v.begin(), v.end(), 0.0, [mean](float acc, float x) {
        return acc + (x - mean) * (x - mean);
      });
  float stddev = std::sqrt(sum_sq_diff / (v.size() - 1));
  std::cout << "Sigma cut outliers: " << stddev << std::endl;
  float z_score_threshold = 3.0;
  v.erase(std::remove_if(v.begin(), v.end(),
                         [mean, stddev, z_score_threshold](float x) {
                           float z_score = std::abs(x - mean) / stddev;
                           return z_score > z_score_threshold;
                         }),
          v.end());
}

pair<float, float> stats(const std::vector<float> &v) {
  float m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  float sum = std::accumulate(v.begin(), v.end(), 0.0, [m](float acc, float x) {
    return acc + (x - m) * (x - m);
  });
  auto den = v.size() > 1 ? (v.size() - 1) : v.size();
  return {m, std::sqrt(sum / den)};
}

void printTimingReport(std::vector<float> &vals, int repeats,
                       std::vector<std::pair<std::string, float>> &timings,
                       const std::string label = "SUMMARY ") {
  int precision = 2;
  float mean = 0.f;
  float sigma = 0.f;
  exclude_stats_outliers(vals);
  tie(mean, sigma) = stats(vals);
  std::cout << label << " 1 outliers(" << repeats << "/" << vals.size() << ") "
            << std::fixed << std::setprecision(precision) << mean << " +/- "
            << sigma << " [ms]" << std::endl;
  exclude_stats_outliers(vals);
  tie(mean, sigma) = stats(vals);
  std::cout << label << " 2 outliers(" << repeats << "/" << vals.size() << ") "
            << std::fixed << std::setprecision(precision) << mean << " +/- "
            << sigma << " [ms]" << std::endl;

  if (label == "SUMMARY WorkDivByPoints submission times:") {
    timings.emplace_back("kernelSubmissionMean", mean);
  }     
  else if (label == "SUMMARY WorkDivByPoints execution times:") {
    timings.emplace_back("kernelExecutionMean", mean);
  }
}

//another function same but for output, also allocate output with correct size

void allocateInputData(float* &x, float* &y, int* &layer, float* &weight, int capacity, bool use_accelerator) {

  if (use_accelerator) {
    #if !defined(USE_ALPAKA) 
      // Allocate CUDA-managed memory
      CHECK_CUDA_ERROR(cudaMallocManaged(&x, capacity * sizeof(float)));
      CHECK_CUDA_ERROR(cudaMallocManaged(&y, capacity * sizeof(float)));
      CHECK_CUDA_ERROR(cudaMallocManaged(&layer, capacity * sizeof(int)));
      CHECK_CUDA_ERROR(cudaMallocManaged(&weight, capacity * sizeof(float)));
    #endif
  }
  else {
    x = new float[capacity];
    y = new float[capacity];
    layer = new int[capacity];
    weight = new float[capacity];
  }
}

void readDataFromFile(const std::string &inputFileName, float* x, float* y, int* layer, float* weight, int capacity, int &size) {

  int i = 0;

  // make dummy layers
  for (int l = 0; l < NLAYERS; l++) {
    // open csv file
    std::ifstream iFile(inputFileName);
    std::string value = "";
    // Iterate through each line and split the content using delimeter
    while (getline(iFile, value, ',')) {
      if(i < capacity) {
        x[i] = std::stof(value);
        getline(iFile, value, ',');
        y[i] = std::stof(value);
        getline(iFile, value, ',');
        layer[i] = std::stoi(value) + l;
        getline(iFile, value);
        weight[i] = std::stof(value);
      }
      else{
        std::cerr << "Error: Capacity exceeded (" << capacity << "). Exiting..." << std::endl;
        exit(EXIT_FAILURE);
      }
      i++;
    }
    iFile.close();
  }
  size = i;
}

void allocateOutputData (float* &rho, float* &delta, unsigned int* &nearestHigher, int* &clusterIndex, uint8_t* &isSeed, bool use_accelerator, int size) {
  if (use_accelerator) {
    #if !defined(USE_ALPAKA) 
      // Allocate CUDA-managed memory
      CHECK_CUDA_ERROR(cudaMallocManaged(&rho, size * sizeof(float)));
      CHECK_CUDA_ERROR(cudaMallocManaged(&delta, size * sizeof(float)));
      CHECK_CUDA_ERROR(cudaMallocManaged(&nearestHigher, size * sizeof(int)));
      CHECK_CUDA_ERROR(cudaMallocManaged(&clusterIndex, size * sizeof(int)));
      CHECK_CUDA_ERROR(cudaMallocManaged(&isSeed, size * sizeof(uint8_t)));

    #endif
  }
  else{
    rho = new float[size];
    delta = new float[size];
    nearestHigher = new unsigned int[size];
    clusterIndex = new int[size];
    isSeed = new uint8_t[size];
  }
}

void freeInputData(float* &x, float* &y, int* &layer, float* &weight, bool use_accelerator) {
  if (use_accelerator) {
    #if !defined(USE_ALPAKA) 
      CHECK_CUDA_ERROR(cudaFree(x));
      CHECK_CUDA_ERROR(cudaFree(y));
      CHECK_CUDA_ERROR(cudaFree(layer));
      CHECK_CUDA_ERROR(cudaFree(weight));
    #endif
  }
  else{
    delete[] x;
    delete[] y;
    delete[] layer;
    delete[] weight;
  }
}

void freeOutputData(float* &rho, float* &delta, unsigned int* &nearestHigher, int* &clusterIndex, uint8_t* &isSeed, bool use_accelerator) {
  if (use_accelerator) {
    #if !defined(USE_ALPAKA) 
      CHECK_CUDA_ERROR(cudaFree(rho));
      CHECK_CUDA_ERROR(cudaFree(delta));
      CHECK_CUDA_ERROR(cudaFree(nearestHigher));
      CHECK_CUDA_ERROR(cudaFree(clusterIndex));
      CHECK_CUDA_ERROR(cudaFree(isSeed));
    #endif
  }
  else{
    delete[] rho;
    delete[] delta;
    delete[] nearestHigher;
    delete[] clusterIndex;
    delete[] isSeed;
  }
}

std::string create_outputfileName(const std::string &inputFileName,
                                  const float dc, const float rhoc,
                                  const float outlierDeltaFactor) {
  //  C++20
  //  auto suffix = std::format("_{:.2f}_{:.2f}_{:.2f}.csv", dc, rhoc,
  //  outlierDeltaFactor);
  char suffix[100];
  snprintf(suffix, 100, "_dc_%.2f_rho_%.2f_outl_%.2f.csv", dc, rhoc,
           outlierDeltaFactor);

  std::string tmpFileName;
  std::regex regexp("input");
  std::regex_replace(back_inserter(tmpFileName), inputFileName.begin(),
                     inputFileName.end(), regexp, "output");

  std::string outputFileName;
  std::regex regexp2(".csv");
  std::regex_replace(back_inserter(outputFileName), tmpFileName.begin(),
                     tmpFileName.end(), regexp2, suffix);

  return outputFileName;
}



void mainRun(const std::string &inputFileName,
             const std::string &outputFileName, const float dc,
             const float rhoc, const float outlierDeltaFactor,
             const bool use_accelerator, const int repeats,
             const bool verbose, char* argv[]) {

    CHECK_CUDA_ERROR(cudaFree(nullptr));   

    //////////////////////////////
    // read toy data from csv file
    //////////////////////////////

    std::cout << "Start to load input points" << std::endl;

    std::vector<std::pair<std::string, float>> timings;

    // Allocate memory
    unsigned int capacity = 1000000;
    int size;

    int gpuId;
    cudaGetDevice(&gpuId);

    float* x = nullptr;
    float* y = nullptr;
    int* layer = nullptr;
    float* weight = nullptr;

    float* rho = nullptr;
    float* delta = nullptr;
    unsigned int* nearestHigher = nullptr;
    int* clusterIndex = nullptr;
    uint8_t* isSeed = nullptr;

    std::cout << "Finished loading input points" << std::endl;
    // Vector to perform some bread and butter analysis on the timing
    vector<float> vals;
    vector<float> vals2;

    auto begin = std::chrono::high_resolution_clock::now();

    allocateInputData(x, y, layer, weight, capacity, use_accelerator);

    auto end = std::chrono::high_resolution_clock::now();

    float time_allocate_input = std::chrono::duration<float>(end - begin).count();
    
    timings.emplace_back("allocateInputData", time_allocate_input * 1000);

    begin = std::chrono::high_resolution_clock::now();

    readDataFromFile(inputFileName, x, y, layer, weight, capacity, size);

    end = std::chrono::high_resolution_clock::now();

    float time_read = std::chrono::duration<float>(end - begin).count();

    timings.emplace_back("readDataFromFile", time_read * 1000);

    begin = std::chrono::high_resolution_clock::now();
      
    allocateOutputData(rho, delta, nearestHigher, clusterIndex, isSeed, use_accelerator, size);

    end = std::chrono::high_resolution_clock::now();

    float time_allocate_output = std::chrono::duration<float>(end - begin).count();
    
    timings.emplace_back("allocateOutputData", time_allocate_output * 1000);

    //////////////////////////////
    // run CLUE algorithm
    //////////////////////////////
    std::cout << "Start to run CLUE algorithm" << std::endl;
    if (use_accelerator) {
  #if !defined(USE_ALPAKA)
      std::cout << "Native CUDA Backend selected" << std::endl;
      CLUEAlgoGPU<TilesConstants, NLAYERS> clueAlgo(dc, rhoc, outlierDeltaFactor, verbose, size, x, y, layer, weight, 
        rho, delta, nearestHigher, clusterIndex, isSeed);
      vals.clear();
      vals2.clear();
      for (unsigned r = 0; r < repeats; r++) {
        clueAlgo.setInputPoints(size, x, y, layer, weight);
        clueAlgo.setOutputPoints(size, rho, delta, nearestHigher, clusterIndex, isSeed);
        // measure excution time of makeClusters
        clueAlgo.Sync();
        auto start = std::chrono::high_resolution_clock::now();
        clueAlgo.copy_todevice();
        clueAlgo.makeClusters();
        clueAlgo.copy_tohost();
        auto finish = std::chrono::high_resolution_clock::now();
        clueAlgo.Sync();
        auto finish2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> submit = finish - start;
        std::chrono::duration<float> execute = finish2 - start;
        std::cout << "Iteration " << r;
        std::cout << " | Submission time: " << submit.count() * 1000 << " ms\n";
        std::cout << " | Execution time: " << execute.count() * 1000 << " ms\n";
        // Skip first event
        if (r != 0 or repeats == 1) {
          vals.push_back(submit.count() * 1000);
          vals2.push_back(execute.count() * 1000);
        }
      }

      printTimingReport(vals, repeats, timings, "SUMMARY WorkDivByPoints submission times:");
      printTimingReport(vals2, repeats, timings, "SUMMARY WorkDivByPoints execution times:");

      begin = std::chrono::high_resolution_clock::now();

      // output result to outputFileName. -1 means all points.
      clueAlgo.verboseResults(outputFileName, -1);

      end = std::chrono::high_resolution_clock::now();

      float time_write = std::chrono::duration<float>(end - begin).count();

      timings.emplace_back("writeDataToFile", time_write * 1000);

      begin = std::chrono::high_resolution_clock::now();
      
      freeInputData(x, y, layer, weight, use_accelerator);
      
      end = std::chrono::high_resolution_clock::now();

      float time_free_input = std::chrono::duration<float>(end - begin).count();

      timings.emplace_back("freeInputData", time_free_input * 1000);

      begin = std::chrono::high_resolution_clock::now();
      
      freeOutputData(rho, delta, nearestHigher, clusterIndex, isSeed, use_accelerator);
      
      end = std::chrono::high_resolution_clock::now();

      float time_free_output = std::chrono::duration<float>(end - begin).count();

      timings.emplace_back("freeOutputData", time_free_output * 1000);    

  #endif
    } else {

      std::cout << "Native CPU(serial) Backend selected" << std::endl;
      CLUEAlgo<TilesConstants, NLAYERS> clueAlgo(dc, rhoc, outlierDeltaFactor,
                                                verbose);

      vals.clear();
      for (int r = 0; r < repeats; r++) { 
        clueAlgo.setInputPoints(size, x, y, layer, weight);
        clueAlgo.setOutputPoints(size, rho, delta, nearestHigher, clusterIndex, isSeed);
        // measure excution time of makeClusters
        auto start = std::chrono::high_resolution_clock::now();
        clueAlgo.makeClusters();
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count() * 1000 << " ms\n";
        // Skip first event
        if (r != 0 or repeats == 1) {
          vals.push_back(elapsed.count() * 1000);
        }
      }

      printTimingReport(vals, repeats, timings, "SUMMARY Native CPU:");
      // output result to outputFileName. -1 means all points.
      if (verbose)
        clueAlgo.verboseResults(outputFileName, -1);
    }

    std::string run_number = argv[13];
    std::string filename = "Results/results_unified" + run_number + ".csv";

    std::ofstream results(filename);
    if (!results.is_open()) {
      std::cerr << "Failed to open file.\n";
      return;
    }

    results << "Operation,Time\n";
    for (const auto& entry : timings) {
        results << entry.first << "," << entry.second << "\n";
    }

    results.close();

    std::cout << "Finished running CLUE algorithm" << std::endl;
  //}
}  // end of testRun()

int main(int argc, char *argv[]) {
  //////////////////////////////
  // MARK -- set algorithm parameters
  //////////////////////////////

  extern char *optarg;

  bool use_accelerator = false;
  bool verbose = false;
  float dc = 20.f, rhoc = 80.f, outlierDeltaFactor = 2.f;
  int repeats = 10;
  int TBBNumberOfThread = 1;
  int opt;
  std::string inputFileName;

  while ((opt = getopt(argc, argv, "i:d:r:o:e:t:uv")) != -1) {
    switch (opt) {
    case 'i': /* input filename */
      inputFileName = string(optarg);
      break;
    case 'd': /* delta_c */
      dc = stof(string(optarg));
      break;
    case 'r': /* critical density */
      rhoc = stof(string(optarg));
      break;
    case 'o': /* outlier factor */
      outlierDeltaFactor = stof(string(optarg));
      break;
    case 'e': /* number of repeated session(s) a the selected input file */
      repeats = stoi(string(optarg));
      break;
    case 't': /* number of TBB threads */
      TBBNumberOfThread = stoi(string(optarg));
      std::cout << "Using " << TBBNumberOfThread;
      std::cout << " TBB Threads" << std::endl;
      break;
    case 'u': /* Use accelerator */
      use_accelerator = true;
      break;
    case 'v': /* Verbose output */
      verbose = true;
      break;
    default:
      std::cout << "bin/main -i [fileName] -d [dc] -r [rhoc] -o "
                   "[outlierDeltaFactor] -e [repeats] -t "
                   "[NumTBBThreads] -u -v"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  if (verbose) {
    std::cout << "Setting up " << TBBNumberOfThread << " TBB Threads"
              << std::endl;
  }
  tbb::global_control init(tbb::global_control::max_allowed_parallelism, TBBNumberOfThread);
#endif

  //////////////////////////////
  // MARK -- set input and output files
  //////////////////////////////
  std::cout << "Input file: " << inputFileName << std::endl;

  std::string outputFileName =
      create_outputfileName(inputFileName, dc, rhoc, outlierDeltaFactor);
  std::cout << "Output file: " << outputFileName << std::endl;

  //////////////////////////////
  // MARK -- test run
  //////////////////////////////
  mainRun(inputFileName, outputFileName, dc, rhoc, outlierDeltaFactor,
          use_accelerator, repeats, verbose, argv);

  return 0;
}
