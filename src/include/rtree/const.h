#ifndef _CONST_H_
#define _CONST_H_

#include <vector>

#define INF_N -std::numeric_limits<float>::max()
#define INF_P std::numeric_limits<float>::max()

typedef float Coord;
typedef std::vector<float> Coord_V;
typedef std::vector<Coord_V> Coord_VV;

const float EPS = 1e-6;

#endif /*_CONST_H_*/
