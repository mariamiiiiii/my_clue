#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "CLUEAlgo.h"
#ifndef USE_CUPLA
#include "CLUEAlgoGPU.h"
#else
#include "CLUEAlgoCupla.h"
#ifdef FOR_TBB
#include "tbb/task_scheduler_init.h"
#endif
#endif

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string create_outputfileName(std::string inputFileName, float dc,
                                 float rhoc, float outlierDeltaFactor,
                                 bool useParallel){
  std::string underscore = "_", suffix = "";
  suffix.append(underscore);
  suffix.append(to_string_with_precision(dc,2));
  suffix.append(underscore);
  suffix.append(to_string_with_precision(rhoc,2));
  suffix.append(underscore);
  suffix.append(to_string_with_precision(outlierDeltaFactor,2));
  suffix.append(".csv");

  std::string outputFileName = inputFileName;
  replace(outputFileName, "input", "output");
  replace(outputFileName, ".csv", suffix);
  return outputFileName;
}



//std::string string("hello $name");
//replace(string, "$name", "Somename");



void mainRun( std::string inputFileName, std::string outputFileName,
              float dc, float rhoc, float outlierDeltaFactor,
              bool useGPU, int repeats, bool verbose  ) {

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
  if (useGPU) {
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
  float dc=20.f, rhoc=80.f, outlierDeltaFactor=2.f;
  bool useGPU=false;
  int totalNumberOfEvent = 10;
  bool verbose=false;

  int TBBNumberOfThread = 1;

  std::string inputFileName = argv[1];
  if (argc == 8 || argc == 9) {
    dc = std::stof(argv[2]);
    rhoc = std::stof(argv[3]);
    outlierDeltaFactor = std::stof(argv[4]);
    useGPU = (std::stoi(argv[5])==1)? true:false;
    totalNumberOfEvent = std::stoi(argv[6]);
    verbose = (std::stoi(argv[7])==1)? true:false;
    if (argc == 9) {
      TBBNumberOfThread = std::stoi(argv[8]);
      if (verbose) {
        std::cout << "Using " << TBBNumberOfThread;
	std::cout << " TBB Threads" << std::endl;
      }
    }
  } else {
    std::cout << "bin/main [fileName] [dc] [rhoc] [outlierDeltaFactor] [useGPU] [totalNumberOfEvent] [verbose] [NumTBBThreads]" << std::endl;
    return 1;
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


  std::string outputFileName = create_outputfileName(inputFileName, dc, rhoc, outlierDeltaFactor, useGPU);
  std::cout << "Output file: " << outputFileName << std::endl;


  //////////////////////////////
  // MARK -- test run
  //////////////////////////////
  mainRun(inputFileName, outputFileName,
          dc, rhoc, outlierDeltaFactor, 
          useGPU, totalNumberOfEvent, verbose);

  return 0;
}
