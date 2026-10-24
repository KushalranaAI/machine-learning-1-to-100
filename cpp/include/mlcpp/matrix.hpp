// A tiny dense matrix (row-major) - just enough linear algebra for the algorithms in this folder.
// Readability over speed: every operation is a plain loop you can follow.
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlcpp {

using Vector = std::vector<double>;

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0) : rows_(rows), cols_(cols), data_(rows * cols, value) {}
    Matrix(std::initializer_list<std::initializer_list<double>> init) {
        rows_ = init.size();
        cols_ = rows_ ? init.begin()->size() : 0;
        for (const auto& row : init) {
            if (row.size() != cols_) throw std::invalid_argument("ragged initializer");
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    double operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    Vector row(std::size_t i) const { return Vector(data_.begin() + i * cols_, data_.begin() + (i + 1) * cols_); }
    Vector col(std::size_t j) const {
        Vector c(rows_);
        for (std::size_t i = 0; i < rows_; ++i) c[i] = (*this)(i, j);
        return c;
    }

    Matrix transpose() const {
        Matrix t(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) t(j, i) = (*this)(i, j);
        return t;
    }

    Matrix operator*(const Matrix& b) const {  // (n×k)(k×m) - i-k-j loop order is cache friendly
        if (cols_ != b.rows_) throw std::invalid_argument("shape mismatch in matrix product");
        Matrix out(rows_, b.cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t k = 0; k < cols_; ++k) {
                const double a = (*this)(i, k);
                for (std::size_t j = 0; j < b.cols_; ++j) out(i, j) += a * b(k, j);
            }
        return out;
    }

    Vector operator*(const Vector& v) const {
        if (cols_ != v.size()) throw std::invalid_argument("shape mismatch in matrix-vector product");
        Vector out(rows_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) out[i] += (*this)(i, j) * v[j];
        return out;
    }

private:
    std::size_t rows_ = 0, cols_ = 0;
    std::vector<double> data_;
};

inline double dot(const Vector& a, const Vector& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double squared_distance(const Vector& a, const Vector& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

// Solve A x = b for symmetric positive-definite A with a Cholesky factorisation A = L Lᵀ.
// Used for the normal equation (XᵀX) w = Xᵀy.
inline Vector cholesky_solve(const Matrix& A, const Vector& b) {
    const std::size_t n = A.rows();
    Matrix L(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = A(i, j);
            for (std::size_t k = 0; k < j; ++k) s -= L(i, k) * L(j, k);
            if (i == j) {
                if (s <= 0) throw std::runtime_error("matrix is not positive definite (collinear features?)");
                L(i, i) = std::sqrt(s);
            } else {
                L(i, j) = s / L(j, j);
            }
        }
    }
    Vector y(n), x(n);
    for (std::size_t i = 0; i < n; ++i) {  // forward substitution: L y = b
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) s -= L(i, k) * y[k];
        y[i] = s / L(i, i);
    }
    for (std::size_t i = n; i-- > 0;) {  // back substitution: Lᵀ x = y
        double s = y[i];
        for (std::size_t k = i + 1; k < n; ++k) s -= L(k, i) * x[k];
        x[i] = s / L(i, i);
    }
    return x;
}

// Minimal CSV reader: numeric columns become the matrix; a non-numeric last column becomes string labels.
struct CsvData {
    Matrix X;
    std::vector<std::string> labels;
    std::vector<std::string> header;
};

inline CsvData read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    CsvData d;
    std::string line;
    std::getline(in, line);
    std::stringstream hs(line);
    for (std::string cell; std::getline(hs, cell, ',');) d.header.push_back(cell);
    std::vector<std::vector<double>> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<std::string> cells;
        for (std::string cell; std::getline(ss, cell, ',');) cells.push_back(cell);
        std::vector<double> nums;
        for (std::size_t j = 0; j + 1 < cells.size(); ++j) nums.push_back(std::stod(cells[j]));
        try {
            nums.push_back(std::stod(cells.back()));
        } catch (const std::exception&) {
            d.labels.push_back(cells.back());
        }
        rows.push_back(nums);
    }
    d.X = Matrix(rows.size(), rows.empty() ? 0 : rows[0].size());
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < rows[i].size(); ++j) d.X(i, j) = rows[i][j];
    return d;
}

}  // namespace mlcpp
