//
// Created by BianZheng on 2022/5/26.
//

#ifndef REVERSE_K_RANKS_CAUSERITEMPQ_HPP
#define REVERSE_K_RANKS_CAUSERITEMPQ_HPP

#include <cfloat>

#include "alg/Cluster/KMeansParallel.hpp"
#include "BaseIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SpaceEuclidean.hpp"

namespace ReverseMIPS {

    class CAUserItemPQ : public BaseIPBound {
        int n_user_, n_data_item_, vec_dim_, n_codeword_, n_codebook_;
        int normal_dim_, last_dim_, appr_IP_offset_;
        std::unique_ptr<double[]> item_norm_l_;//n_data_item, store the norm of approximate item
        std::unique_ptr<double[]> user_norm_l_;//n_user, store the norm of approximate user

        std::unique_ptr<double[]> item_codebook_;//n_codeword * vec_dim, it is concat by diff codeword
        std::unique_ptr<double[]> user_codebook_;//n_codeword * vec_dim, same as item_codebook_

        std::unique_ptr<unsigned int[]> item_codeword_;//n_data_item * n_codebook, store the code of item
        std::unique_ptr<unsigned int[]> user_codeword_;//n_user * n_codebook, store the code of user

        std::unique_ptr<double[]> item_error_l_;//n_data_item, store the euclidean distance of approximate
        std::unique_ptr<double[]> user_error_l_;//n_user, same as item_error_l_

        std::unique_ptr<double[]> appr_IP_l_;//n_codebook * n_codeword * n_codeword, store the approximate IP


    public:

        inline CAUserItemPQ() {
            this->n_user_ = -1;
            this->n_data_item_ = -1;
            this->vec_dim_ = -1;
            this->n_codeword_ = -1;
            this->n_codebook_ = -1;

            this->normal_dim_ = -1;
            this->last_dim_ = -1;

            this->appr_IP_offset_ = -1;
        }

        inline CAUserItemPQ(const int &n_user, const int &n_data_item, const int &vec_dim,
                            const int &n_codebook, const int &n_codeword) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->n_codebook_ = n_codebook;
            this->n_codeword_ = n_codeword;
            this->appr_IP_offset_ = n_codeword * n_codeword;

            this->item_norm_l_ = std::make_unique<double[]>(n_data_item);
            this->user_norm_l_ = std::make_unique<double[]>(n_user);

            this->item_codebook_ = std::make_unique<double[]>(n_codeword * vec_dim);
            this->user_codebook_ = std::make_unique<double[]>(n_codeword * vec_dim);

            this->item_codeword_ = std::make_unique<unsigned int[]>(n_data_item * n_codebook);
            this->user_codeword_ = std::make_unique<unsigned int[]>(n_user * n_codebook);

            this->item_error_l_ = std::make_unique<double[]>(n_data_item);
            this->user_error_l_ = std::make_unique<double[]>(n_user);

            this->appr_IP_l_ = std::make_unique<double[]>(n_codebook * n_codeword * n_codeword);

            if (n_codebook_ <= 1) {
                spdlog::error("n_codebook is too small, program exit");
                exit(-1);
            }
            if (n_codebook_ > vec_dim_) {
                spdlog::error("n_codebook is too large, program exit");
                exit(-1);
            }

            //determine which dimension should use
            this->normal_dim_ = vec_dim_ / n_codebook_;
            this->last_dim_ = normal_dim_ + vec_dim_ % n_codebook_;

            spdlog::info("n_codebook {}, n_codeword {}, appr_IP_offset_ {}, normal_dim {}, last_dim {}",
                         n_codebook_, n_codeword_, appr_IP_offset_, normal_dim_, last_dim_);
        }

        void BuildCode(const VectorMatrix &vm,
                       double *vm_codebook, unsigned int *vm_codeword, double *vecs_error_l, double *vecs_norm_l) {
            const int n_vector = vm.n_vector_;

            for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                vecs_error_l[vecsID] = 0;
                vecs_norm_l[vecsID] = 0;
            }

            for (int cbookID = 0; cbookID < n_codebook_; cbookID++) {
                const int part_dim = cbookID == n_codebook_ - 1 ? last_dim_ : normal_dim_;
                const int dim_offset = normal_dim_ * cbookID;
                assert(dim_offset + part_dim <= vec_dim_);

                std::unique_ptr<double[]> vm_raw_data = std::make_unique<double[]>(n_vector * part_dim);
                for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                    memcpy(vm_raw_data.get() + vecsID * part_dim, vm.getVector(vecsID, dim_offset),
                           part_dim * sizeof(double));
                }
                VectorMatrix part_vm;
                part_vm.init(vm_raw_data, n_vector, part_dim);

                std::tuple<std::vector<std::vector<double>>, std::vector<uint32_t>> cluster_data =
                        KMeans::ClusterData(part_vm, n_codeword_);
                std::vector<uint32_t> label_l = std::get<1>(cluster_data);
                std::vector<std::vector<double>> centroid_l = std::get<0>(cluster_data);
                assert(label_l.size() == n_vector);
                assert(centroid_l.size() == n_codeword_);

                //assign the codebook
                for (int cwordID = 0; cwordID < n_codeword_; cwordID++) {
                    memcpy(vm_codebook + cwordID * vec_dim_ + dim_offset, centroid_l[cwordID].data(),
                           part_dim * sizeof(double));
                }

                //assign the codeword
                for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                    const uint32_t min_cwordID = label_l[vecsID];
                    vm_codeword[vecsID * n_codebook_ + cbookID] = min_cwordID;

                    const double *centroid = centroid_l[min_cwordID].data();
                    assert(centroid_l[min_cwordID].size() == part_dim);
                    const double min_dist = EuclideanDistanceSquare(centroid, vm.getVector(vecsID, dim_offset),
                                                                    part_dim);
                    vecs_error_l[vecsID] += min_dist;

                    assert(centroid_l[min_cwordID].size() == part_dim);
                    const double part_norm = InnerProduct(centroid, centroid, part_dim);
                    vecs_norm_l[vecsID] += part_norm;
                }

            }
            for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                vecs_error_l[vecsID] = std::sqrt(vecs_error_l[vecsID]);
                vecs_norm_l[vecsID] = std::sqrt(vecs_norm_l[vecsID]);
            }

        }

        void Preprocess(VectorMatrix &user, VectorMatrix &data_item) {
            //perform kmeans on each dimension, and assign the codebook and codeword

            BuildCode(user,
                      user_codebook_.get(), user_codeword_.get(), user_error_l_.get(), user_norm_l_.get());
            BuildCode(data_item,
                      item_codebook_.get(), item_codeword_.get(), item_error_l_.get(), item_norm_l_.get());

            for (int cbookID = 0; cbookID < n_codebook_; cbookID++) {
                const int part_dim = cbookID == n_codebook_ - 1 ? last_dim_ : normal_dim_;
                const int dim_offset = normal_dim_ * cbookID;
                assert(dim_offset <= vec_dim_);

                for (int cuserID = 0; cuserID < n_codeword_; cuserID++) {
                    for (int citemID = 0; citemID < n_codeword_; citemID++) {
                        const double *cuser_ptr = user_codebook_.get() + cuserID * vec_dim_ + dim_offset;
                        const double *citem_ptr = item_codebook_.get() + citemID * vec_dim_ + dim_offset;
                        const double IP = InnerProduct(citem_ptr, cuser_ptr, part_dim);
                        appr_IP_l_[cbookID * appr_IP_offset_ + cuserID * n_codeword_ + citemID] = IP;
                    }
                }
            }

        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) {
            memcpy(query_write_vecs, query_vecs, vec_dim * sizeof(double));
        }

        double IPUpperBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            double appr_IP = 0;
            for (int cbookID = 0; cbookID < n_codebook_; cbookID++) {
                unsigned int citemID = item_codeword_[itemID * n_codebook_ + cbookID];
                unsigned int cuserID = user_codeword_[userID * n_codebook_ + cbookID];
                assert(0 <= cuserID && cuserID < n_codeword_);
                assert(0 <= citemID && citemID < n_codeword_);
                appr_IP += appr_IP_l_[cbookID * appr_IP_offset_ + cuserID * n_codeword_ + citemID];
            }
            double error_times = item_error_l_[itemID] * user_error_l_[userID];
            double norm_error =
                    user_error_l_[userID] * item_norm_l_[itemID] + user_norm_l_[userID] * item_error_l_[itemID];

            double IP_ub = appr_IP + error_times + norm_error;

            return IP_ub;
        }

        double IPLowerBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            double appr_IP = 0;
            for (int cbookID = 0; cbookID < n_codebook_; cbookID++) {
                unsigned int citemID = item_codeword_[itemID * n_codebook_ + cbookID];
                unsigned int cuserID = user_codeword_[userID * n_codebook_ + cbookID];
                assert(0 <= cuserID && cuserID < n_codeword_);
                assert(0 <= citemID && citemID < n_codeword_);
                appr_IP += appr_IP_l_[cbookID * appr_IP_offset_ + cuserID * n_codeword_ + citemID];
            }
            double error_times = item_error_l_[itemID] * user_error_l_[userID];
            double norm_error =
                    user_error_l_[userID] * item_norm_l_[itemID] + user_norm_l_[userID] * item_error_l_[itemID];

            double IP_lb = appr_IP - error_times - norm_error;

            return IP_lb;
        }

        std::pair<double, double>
        IPBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            double appr_IP = 0;
            for (int cbookID = 0; cbookID < n_codebook_; cbookID++) {
                unsigned int citemID = item_codeword_[itemID * n_codebook_ + cbookID];
                unsigned int cuserID = user_codeword_[userID * n_codebook_ + cbookID];
                assert(0 <= cuserID && cuserID < n_codeword_);
                assert(0 <= citemID && citemID < n_codeword_);
                appr_IP += appr_IP_l_[cbookID * appr_IP_offset_ + cuserID * n_codeword_ + citemID];
            }
            double error_times = item_error_l_[itemID] * user_error_l_[userID];
            double norm_error =
                    user_error_l_[userID] * item_norm_l_[itemID] + user_norm_l_[userID] * item_error_l_[itemID];

            double IP_lb = appr_IP - error_times - norm_error;
            double IP_ub = appr_IP + error_times + norm_error;

            return std::make_pair(IP_lb, IP_ub);
        }

        void
        IPBound(const double *user_vecs, const int &userID,
                const std::vector<int> &item_cand_l,
                const VectorMatrix &item,
                std::pair<double, double> *IPbound_l) override {
            for (const int &itemID: item_cand_l) {
                const double *item_vecs = item.getVector(itemID);
                IPbound_l[itemID] = IPBound(user_vecs, userID, item_vecs, itemID);
            }
        }

    };
}
#endif //REVERSE_K_RANKS_CAUSERITEMPQ_HPP
