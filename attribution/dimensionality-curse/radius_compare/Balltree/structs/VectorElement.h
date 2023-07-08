#ifndef VECTORELEMENT_H
#define VECTORELEMENT_H

class VectorElement {
public:
	int id;
	float data;

	inline VectorElement() {
		id = -1;
		data = 0;
	}

	inline VectorElement(int _id, float _data) {
		id = _id;
		data = _data;
	}

	inline bool operator==(const VectorElement & other) const {
		if (this == &other)
			return true;
		return data == other.data && id == other.id;
	};

	inline bool operator!=(const VectorElement & other) const {
		if (this == &other)
			return false;
		return data != other.data || id != other.id;
	};

	inline bool operator<(const VectorElement & other) const {
		return data < other.data;
	}

	inline bool operator<=(const VectorElement & other) const {
		return data <= other.data;
	}

	inline bool operator>(const VectorElement & other) const {
		return data > other.data;
	}

	inline bool operator>=(const VectorElement & other) const {
		return data >= other.data;
	}

};

#endif //VECTORELEMENT_H
