#ifndef Points_h
#define Points_h

struct Points {
  const float* p_x;
  const float* p_y;
  const int* p_layer;
  const float* p_weight;
  const float* p_sigmaNoise;

  float* p_rho;
  float* p_delta;
  unsigned int* p_nearestHigher;
  int* p_clusterIndex;
  std::vector<std::vector<int>> followers;
  uint8_t* p_isSeed;

  int n;

  void clear() {

    p_x = nullptr;
    p_y = nullptr;
    p_layer = nullptr;
    p_weight = nullptr;

    p_rho = nullptr;
    p_delta = nullptr;
    p_nearestHigher = nullptr;
    p_clusterIndex = nullptr;
    followers.clear();
    p_isSeed = nullptr;

    n = 0;
  }
};
#endif
