#pragma once

#include <utility>
#include <limits>
#include <vector>

class Point;

class Rectangle {

    typedef std::pair<float, float> interval;

    size_t vec_dim_;
    std::vector<interval> bounds;

public:

    Rectangle(const size_t &vec_dim) {
        vec_dim_ = vec_dim;
        bounds.resize(vec_dim_);
    };

    typedef interval *iterator;
    typedef const interval *const_iterator;

    iterator begin();

    iterator end();

    const_iterator begin() const;

    const_iterator end() const;

    Rectangle &operator=(const Rectangle &rect);

    interval &operator[](size_t idx);

    interval operator[](size_t idx) const;

    float get_area();

    float get_margin();

    void reset();

    void adjust(const Rectangle &rect);

    float get_overlap(const Rectangle &rect);

    float MINDIST(const Point &pt);

    float MINMAXDIST(const Point &pt);

};

class Point {

    size_t vec_dim_;
    std::vector<float> coords;

public:

    Point(const size_t &vec_dim) {
        vec_dim_ = vec_dim;
        coords.resize(vec_dim);
    }

    typedef float *iterator;
    typedef const float *const_iterator;

    iterator begin();

    iterator end();

    const_iterator begin() const;

    const_iterator end() const;

    Point &operator=(const Point &rect);

    float &operator[](size_t idx);

    float operator[](size_t idx) const;

    Rectangle get_rect() const;

    std::vector<float> get_vector();
};

// Rectangle Implementation

typename Rectangle::iterator Rectangle::begin() {
    return bounds.data();
}

typename Rectangle::iterator Rectangle::end() {
    return bounds.data() + vec_dim_;
}

typename Rectangle::const_iterator Rectangle::begin() const {
    return bounds.data();
}

typename Rectangle::const_iterator Rectangle::end() const {
    return bounds.data() + vec_dim_;
}

Rectangle &Rectangle::operator=(const Rectangle &rect) {
    this->vec_dim_ = rect.vec_dim_;
    std::copy(rect.begin(), rect.end(), begin());
    return *this;
}

typename Rectangle::interval &Rectangle::operator[](size_t idx) {
    return bounds[idx];
}

typename Rectangle::interval Rectangle::operator[](size_t idx) const {
    return bounds[idx];
}

float Rectangle::get_area() {
    float area = 1;
    for (size_t i = 0; i < vec_dim_; ++i) {
        if ((*this)[i].second != (*this)[i].first)
            area *= ((*this)[i].second - (*this)[i].first);
    }
    return area == 1 ? 0 : area;
}

float Rectangle::get_margin() {
    float margin = 0;
    for (size_t i = 0; i < vec_dim_; ++i) {
        margin += ((*this)[i].second - (*this)[i].first);
    }
    margin *= (1 << (vec_dim_ - 1));
    return margin;
}

void Rectangle::reset() {
    for (size_t i = 0; i < vec_dim_; ++i) {
        (*this)[i].first = std::numeric_limits<float>::max();
        (*this)[i].second = std::numeric_limits<float>::min();
    }
}

void Rectangle::adjust(const Rectangle &rect) {
    for (size_t i = 0; i < vec_dim_; ++i) {
        (*this)[i].first = std::min((*this)[i].first, rect[i].first);
        (*this)[i].second = std::max((*this)[i].second, rect[i].second);
    }
}

float Rectangle::get_overlap(const Rectangle &rect) {
    float area = 1;
    for (size_t i = 0; i < vec_dim_; ++i) {
        float left = std::max((*this)[i].first, rect[i].first);
        float right = std::min((*this)[i].second, rect[i].second);
        if (right != left)
            area *= std::max(float(0), right - left);
    }
    return area == 1 ? 0 : area;
}

float Rectangle::MINDIST(const Point &pt) {
    float ans = 0, dist;
    for (size_t i = 0; i < vec_dim_; ++i) {
        if (pt[i] < bounds[i].first) {
            dist = pt[i] - bounds[i].first;
        } else if (pt[i] > bounds[i].second) {
            dist = pt[i] - bounds[i].second;
        } else {
            dist = 0;
        }
        ans += dist * dist;
    }
    return ans;
}

float Rectangle::MINMAXDIST(const Point &pt) {
    float S = 0, dist;
    for (size_t i = 0; i < vec_dim_; ++i) {
        if (pt[i] >= (bounds[i].first + bounds[i].second) / 2) {
            dist = pt[i] - bounds[i].first;
        } else {
            dist = pt[i] - bounds[i].second;
        }
        S += dist * dist;
    }
    float ans = std::numeric_limits<float>::max(), cur;
    for (size_t i = 0; i < vec_dim_; ++i) {
        if (pt[i] >= (bounds[i].first + bounds[i].second) / 2) {
            dist = pt[i] - bounds[i].first;
        } else {
            dist = bounds[i].second - pt[i];
        }
        cur = S - (dist * dist);
        if (pt[i] <= (bounds[i].first + bounds[i].second) / 2) {
            dist = pt[i] - bounds[i].first;
        } else {
            dist = bounds[i].second - pt[i];
        }
        cur += dist * dist;
        ans = std::min(ans, cur);
    }
    return ans;
}

// Point Implementation

typename Point::iterator Point::begin() {
    return coords.data();
}

typename Point::iterator Point::end() {
    return coords.data() + vec_dim_;
}

typename Point::const_iterator Point::begin() const {
    return coords.data();
}

typename Point::const_iterator Point::end() const {
    return coords.data() + vec_dim_;
}

Point &Point::operator=(const Point &p) {
    std::copy(p.begin(), p.end(), begin());
    return *this;
}

float &Point::operator[](size_t idx) {
    return coords[idx];
}

float Point::operator[](size_t idx) const {
    return coords[idx];
}

Rectangle Point::get_rect() const {
    Rectangle r(vec_dim_);
    for (size_t i = 0; i < vec_dim_; ++i) {
        r[i].first = r[i].second = coords[i];
    }
    return r;
}

std::vector<float> Point::get_vector() {
    std::vector<float> vect(vec_dim_);
    std::copy(begin(), end(), vect.begin());
    return vect;
}
