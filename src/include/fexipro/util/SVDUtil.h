#ifndef SVDUTIL_H
#define SVDUTIL_H

#include "Base.h"
#include <armadillo>
//#include <mlpack/core.hpp>

using namespace arma;

int SVD(const fmat &P_t, const int m, const int n, Matrix &u, Matrix &v, const float SIGMA){

	fmat U_t;
	fvec s;
	fmat V;

	// see: http://arma.sourceforge.net/docs.html#svd_econ
//	svd_econ(U_t, s, V, P_t, "both", "std");
	svd_econ(U_t, s, V, P_t);

	U_t = U_t.t();

	float *uData = new float[m * m];
	float *vData = new float[m * n];

	for (int rowIndex = 0; rowIndex < m; rowIndex++) {
		for (int colIndex = 0; colIndex < m; colIndex++) {
			uData[rowIndex * m + colIndex] = s[rowIndex] * U_t(rowIndex, colIndex);
		}

	}

	vector<float> sum(m);
	sum[0] = s[0];
	for (int colIndex = 1; colIndex < m; colIndex++) {
		sum[colIndex] = sum[colIndex - 1] + s[colIndex];
	}

	int checkDim = 0;
	for (int colIndex = 0; colIndex < m; colIndex++) {
		if(sum[colIndex] / sum[m - 1] >= SIGMA) {
			checkDim = colIndex;
			break;
		}
	}

	for(int rowIndex = 0; rowIndex < n; rowIndex++){
		for (int colIndex = 0; colIndex < m; colIndex++) {
			vData[rowIndex * m + colIndex] = V(rowIndex, colIndex);
		}
	}

	u.init(uData, m, m);
	v.init(vData, n, m);

	return checkDim;
}

int SVD(const fmat &P_t, const int m, const int n, Matrix &u, Matrix &v, vector<float> &sigmaValues, const float SIGMA){

	fmat U_t;
	fvec s;
	fmat V;

	// see: http://arma.sourceforge.net/docs.html#svd_econ
//	svd_econ(U_t, s, V, P_t, "both", "std");
	svd_econ(U_t, s, V, P_t);

	U_t = U_t.t();

	float *uData = new float[m * m];
	float *vData = new float[m * n];

	for (int rowIndex = 0; rowIndex < m; rowIndex++) {
		for (int colIndex = 0; colIndex < m; colIndex++) {
			uData[rowIndex * m + colIndex] = s[rowIndex] * U_t(rowIndex, colIndex);
		}

	}

	sigmaValues.resize(m);
	vector<float> sum(m);
	sigmaValues[0] = s[0] / s[m-1];
	sum[0] = s[0];
	for (int colIndex = 1; colIndex < m; colIndex++) {
		sigmaValues[colIndex] = s[colIndex] / s[m-1];
		sum[colIndex] = sum[colIndex - 1] + s[colIndex];
	}

	int checkDim = 0;
	for (int colIndex = 0; colIndex < m; colIndex++) {
		if(sum[colIndex] / sum[m - 1] >= SIGMA) {
			checkDim = colIndex;
			break;
		}
	}

	for(int rowIndex = 0; rowIndex < n; rowIndex++){
		for (int colIndex = 0; colIndex < m; colIndex++) {
			vData[rowIndex * m + colIndex] = V(rowIndex, colIndex);
		}
	}

	u.init(uData, m, m);
	v.init(vData, n, m);

	return checkDim;
}

void SVD(const fmat &P_t, const int m, const int n, Matrix &u, Matrix &v, float *vSubNorms) {

	fmat U_t;
	fvec s;
	fmat V;

	// see: http://arma.sourceforge.net/docs.html#svd_econ
//	svd_econ(U_t, s, V, P_t, "both", "std");
	svd_econ(U_t, s, V, P_t);

	U_t = U_t.t();

	u.init(m, m);
	v.init(n, m);

	for (int rowIndex = 0; rowIndex < m; rowIndex++) {
		float *uPtr = u.getRowPtr(rowIndex);
		for (int colIndex = 0; colIndex < m; colIndex++) {
			uPtr[colIndex] = s[rowIndex] * U_t(rowIndex, colIndex);
		}
	}

	for(int rowIndex = 0; rowIndex < n; rowIndex++){
		float *subVNorm = &vSubNorms[rowIndex * m];
		float norm = 0;
		float *vPtr = v.getRowPtr(rowIndex);
		for (int colIndex = m-1; colIndex >= 0; colIndex--) {
			vPtr[colIndex] = V(rowIndex, colIndex);
			norm += V(rowIndex, colIndex) * V(rowIndex, colIndex);
			subVNorm[colIndex] = sqrt(norm);
		}
	}

}
#endif //SVDUTIL_H
