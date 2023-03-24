#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <regex>
#include "CLUEAlgo.h"
#ifndef USE_CUPLA
#include "CLUEAlgoGPU.h"
#else
#include "CLUEAlgoCupla.h"
#ifdef FOR_TBB
#include "tbb/task_scheduler_init.h"
#endif
#endif

using namespace std;

std::string create_outputfileName(const std::string inputFileName,
    const float dc,
    const float rhoc, const float outlierDeltaFactor){
  //  C++20
  //  auto suffix = std::format("_{:.2f}_{:.2f}_{:.2f}.csv", dc, rhoc, outlierDeltaFactor);
  char suffix[100];
  snprintf(suffix, 100, "_dc_%.2f_rho_%.2f_outl_%.2f.csv", dc, rhoc, outlierDeltaFactor);

  std::string tmpFileName;
  std::regex regexp("input");
  std::regex_replace(back_inserter(tmpFileName),
                     inputFileName.begin(), inputFileName.end(), regexp, "output");

  std::string outputFileName;
  std::regex regexp2(".csv");
  std::regex_replace(back_inserter(outputFileName),
                     tmpFileName.begin(), tmpFileName.end(), regexp2, suffix);

  return outputFileName;
}

void mainRun(const std::string & inputFileName,
    const std::string & outputFileName,
    const float dc, const float rhoc, const float outlierDeltaFactor,
    const bool use_accelerator, const int repeats, const bool verbose) {

  //////////////////////////////
  // read toy data from csv file
  //////////////////////////////
  std::cout << "Start to load input points" << std::endl;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;

  // make dummy layers
  for (int l=0; l<NLAYERS; l++){
    // open csv file
    std::ifstream iFile(inputFileName);
    std::string value = "";
    // Iterate through each line and split the content using delimeter
    while (getline(iFile, value, ',')) {
      x.push_back(std::stof(value)) ;
      getline(iFile, value, ','); y.push_back(std::stof(value));
      getline(iFile, value, ','); layer.push_back(std::stoi(value) + l);
      getline(iFile, value); weight.push_back(std::stof(value));
    }
    iFile.close();
  }
  std::cout << "Finished loading input points" << std::endl;

  //////////////////////////////
  // run CLUE algorithm
  //////////////////////////////
  std::cout << "Start to run CLUE algorithm" << std::endl;
  if (use_accelerator) {
#ifndef USE_CUPLA
    std::cout << "Using CLUEAlgoGPU: " << std::endl;
    CLUEAlgoGPU clueAlgo(dc, rhoc, outlierDeltaFactor,
                         verbose);
    for (unsigned r = 0; r<repeats; r++){
      clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]);
      // measure excution time of makeClusters
      auto start = std::chrono::high_resolution_clock::now();
      clueAlgo.makeClusters();
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      std::cout << "Iteration " << r;
      std::cout << " | Elapsed time: " << elapsed.count()*1000 << " ms\n";
    }

    // output result to outputFileName. -1 means all points.
    clueAlgo.verboseResults(outputFileName, -1);

#else
    std::cout << "Using CLUEAlgoCupla: " << std::endl;
    CLUEAlgoCupla<cupla::Acc> clueAlgo(dc, rhoc, outlierDeltaFactor,
                                       verbose);
  for (int r = 0; r<repeats; r++){
    clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]);
    // measure excution time of makeClusters
    auto start = std::chrono::high_resolution_clock::now();
    clueAlgo.makeClusters();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() *1000 << " ms\n";
  }

  // output result to outputFileName. -1 means all points.
  if(verbose)
    clueAlgo.verboseResults(outputFileName, -1);
#endif


  } else {
    std::cout << "Using CLUEAlgo: " << std::endl;
    CLUEAlgo clueAlgo(dc, rhoc, outlierDeltaFactor, verbose);
    for (int r = 0; r<repeats; r++){
      clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]);
      // measure excution time of makeClusters
      auto start = std::chrono::high_resolution_clock::now();
      clueAlgo.makeClusters();
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      std::cout << "Elapsed time: " << elapsed.count() *1000 << " ms\n";
    }

    // output result to outputFileName. -1 means all points.
    if(verbose)
      clueAlgo.verboseResults(outputFileName, -1);
  }

  std::cout << "Finished running CLUE algorithm" << std::endl;
} // end of testRun()



int main(int argc, char *argv[]) {

  //////////////////////////////
  // MARK -- set algorithm parameters
  //////////////////////////////

  extern char *optarg;

  bool use_accelerator=false;
  bool verbose=false;
  float dc=20.f, rhoc=80.f, outlierDeltaFactor=2.f;
  int totalNumberOfEvent = 10;
  int TBBNumberOfThread = 1;
  int opt;
  std::string inputFileName;

  while((opt = getopt(argc, argv, "i:d:r:o:e:t:uv")) != -1 ) {
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
      case 'e': /* number of events */
        totalNumberOfEvent = stoi(string(optarg));
        break;
      case 't': /* number of TBB threads */
        TBBNumberOfThread = stoi(string(optarg));
        std::cout << "Using " << TBBNumberOfThread;
        std::cout << " TBB Threads" << std::endl;
        break;
      case 'u': /* Use accelerator */
        use_accelerator = true;
        break;
      case 'v': /* Use accelerator */
        verbose = true;
        break;
      default:
        std::cout << "bin/main -i [fileName] -d [dc] -r [rhoc] -o [outlierDeltaFactor] -e [totalNumberOfEvent] -t [NumTBBThreads] -u -v" << std::endl;
        exit(EXIT_FAILURE);
    }
  }


#ifdef FOR_TBB
  if (verbose) {
    std::cout << "Setting up " << TBBNumberOfThread << " TBB Threads" << std::endl;
  }
  tbb::task_scheduler_init init(TBBNumberOfThread);
#endif

  //////////////////////////////
  // MARK -- set input and output files
  //////////////////////////////
  std::cout << "Input file: " << inputFileName << std::endl;


  std::string outputFileName = create_outputfileName(inputFileName, dc, rhoc, outlierDeltaFactor);
  std::cout << "Output file: " << outputFileName << std::endl;


  //////////////////////////////
  // MARK -- test run
  //////////////////////////////
  mainRun(inputFileName, outputFileName,
          dc, rhoc, outlierDeltaFactor,
          use_accelerator, totalNumberOfEvent, verbose);

  return 0;
}
