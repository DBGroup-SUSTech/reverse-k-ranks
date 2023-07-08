//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_K_RANKS_BASEIPBOUND_HPP
#define REVERSE_K_RANKS_BASEIPBOUND_HPP
namespace ReverseMIPS {
    class BaseIPBound {
    public:
        virtual void Preprocess(VectorMatrix &user, VectorMatrix &data_item) = 0;

        virtual void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) = 0;

        virtual double
        IPUpperBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) = 0;

        virtual double
        IPLowerBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) = 0;

        virtual std::pair<double, double>
        IPBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) = 0;

        virtual void
        IPBound(const double *user_vecs, const int &userID, const std::vector<int> &item_cand_l,
                const VectorMatrix &item, std::pair<double, double> *IPbound_l) = 0;

        virtual ~BaseIPBound() = default;

    };
}

#endif //REVERSE_K_RANKS_BASEIPBOUND_HPP
