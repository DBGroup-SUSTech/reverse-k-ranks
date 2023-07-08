#ifndef BASICBALLTREENODE_H
#define BASICBALLTREENODE_H

#include "../util/Base.h"
#include "../util/Conf.h"

class BasicBallTreeNode {
protected:
	float *center;
	vector<int> pointIDs;
	vector<const float*> pointPtrs;
	float constrain;
	bool isLeaf;
	int dimension;
public:

	inline BasicBallTreeNode(){}

	inline BasicBallTreeNode(vector<int> &pointIDs, vector<const float*> &pointPtrs, int dimension) {

		this->pointIDs = pointIDs;
		this->pointPtrs = pointPtrs;
		this->isLeaf = false;
		this->constrain = 0;
		this->dimension = dimension;
		this->center = new float[dimension];

		for (int dimIndex = 0; dimIndex < dimension; dimIndex++) {
			center[dimIndex] = 0;
		}

		for (int i = 0; i < pointPtrs.size(); i++) {
			for (int dimIndex = 0; dimIndex < dimension; dimIndex++) {
				center[dimIndex] += pointPtrs[i][dimIndex];
			}
		}

		for (int dimIndex = 0; dimIndex < dimension; dimIndex++) {
			center[dimIndex] /= pointPtrs.size();
		}

	}

	inline ~BasicBallTreeNode() {
		if (this->center)
			delete[]center;
	}

	virtual void splitNode(vector<int> &leftPointIDs, vector<const float *> &leftPointPtrs,
	vector<int> &rightPointIDs, vector<const float *> &rightPointPtrs) = 0;

	inline virtual BasicBallTreeNode *getLeftNode() = 0;

	inline virtual BasicBallTreeNode *getRightNode() = 0;

	inline int getSize() const {
		return this->pointIDs.size();
	}

	inline int getID(const int index) const {
		return pointIDs[index];
	}

	inline const float *getPtr(const int index) const {
		return pointPtrs[index];
	}

	inline float *getMean() const {
		return center;
	}

	inline void setLeafFlag() {
		this->isLeaf = true;
	}

	inline bool isLeafNode() const {
		return isLeaf;
	}

	inline vector<const float *> &getPointPtrs() {
		return this->pointPtrs;
	}

	inline float getConstrain() const {
		return this->constrain;
	}
};
#endif //BASICBALLTREENODE_H