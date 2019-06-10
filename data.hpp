/* Random data generator and reference result on CPU */

#pragma once
#include <numa.h>
#include <vector>
#include <numeric>
#include <utility>
#include <iterator>
#include <algorithm>
#include <boost/range/irange.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

typedef double d_type; 
typedef unsigned int i_type;

using namespace boost;

/*!brief Data structure to store values of a single feature across all examples */
struct feature {
    /*!brief feature data */
    std::vector<d_type> f_data;
    /*!brief example index of each feature */
    std::vector<i_type> f_index;
    /*!brief permuation for sorted feature values */
    std::vector<i_type> f_inv_perm;
    /*!brief ID of this feature */
    size_t feature_id;
    /*!brief number of values in this feature */
    size_t feature_size;
    feature (size_t feature_size = 0, size_t feature_id = 0) : f_data(feature_size), f_index(feature_size), feature_id(feature_id), feature_size(feature_size) {
    }
    inline void add(i_type example_idx, d_type feature_val) {
        feature_size++;
        // if feature_size is power of 2, reserve
        if (feature_size == 1) {
            f_data.reserve(64);
            f_index.reserve(64);
        }
        else if (feature_size >= 64 && !(feature_size & (feature_size - 1))) {
            f_data.reserve(feature_size * 2);
            f_index.reserve(feature_size * 2);
        }
        // push back avoids unnecessary initialization, as it calls the copy constructor
        f_data.push_back(feature_val);
        f_index.push_back(example_idx);
    }

    void sort_feature() {
        std::vector<i_type> perm;
        perm.reserve(f_index.size());
        /* find the permutation */
        boost::push_back(perm, boost::irange(0, int(f_index.size())));
        std::sort(perm.begin(), perm.end(), feature_comp(f_data));
        /* apply permutation to f_data and f_index */
        std::vector<d_type> d_tmp = f_data;
        for (size_t i = 0; i < f_data.size(); ++i) {
            f_data[i] = d_tmp[perm[i]];
        }
        std::vector<i_type> i_tmp = f_index;
        for (size_t i = 0; i < f_index.size(); ++i) {
            f_index[i] = i_tmp[perm[i]];
        }
        /* build the inverse of the permutation */
        f_inv_perm.resize(f_index.size());
        for (size_t i = 0; i < f_index.size(); ++i) {
            f_inv_perm[perm[i]] = i;
        }
    }
    
    /*!brief Given an example ID, find if it has the feature, and feature value */
    bool locate_example(i_type example_id, d_type& val) {
        // binary search to find this example_id
        i_type mid, high, low;
        size_t size = f_index.size();
        if (unlikely(!size)) {
            return false;
        }
        high = size;
        low = 0;
        mid = (high + low) / 2;
        while (low != mid) {
            i_type index = f_index[f_inv_perm[mid]];
            bool cmp_gt = (index > example_id);
            bool cmp_eq = (index == example_id);
            if (cmp_eq) {
                val = f_data[f_inv_perm[mid]];
                return true;
            }
            if (cmp_gt)
                high = mid;
            else
                low = mid;
            mid = (high + low) / 2;
        }
        val = f_data[f_inv_perm[mid]];
        return example_id == f_index[f_inv_perm[mid]];
    }

    /*!brief an index/feature pair supporting swap operation */
    struct f_pair {
        i_type* p_index;
        d_type* p_data;
        f_pair (i_type* p_index, d_type* p_data) : p_index(p_index), p_data(p_data) {}
        f_pair& operator=( const f_pair& rhs ) {
            printf("assigning from %d to %d\n", *(rhs.p_index), *p_index);
            *p_index = *(rhs.p_index);
            *p_data = *(rhs.p_data);
            return *this;
        }
        friend void swap(f_pair a, f_pair b) {
            printf("o%d->%d->", *(a.p_index), *(b.p_index));
            i_type i;
            d_type d;
            i = *(a.p_index);
            d = *(a.p_data);
            *(a.p_index) = *(b.p_index);
            *(a.p_data) = *(b.p_data);
            *(b.p_index) = i;
            *(b.p_data) = d;
            printf("%d\n", *(a.p_index));
        }
        inline friend bool operator< (const f_pair& lhs, const f_pair& rhs){ printf("comp %d (%f) - %d (%f)\n", *(lhs.p_index), *(lhs.p_data), *(rhs.p_index), *(rhs.p_data));return *(lhs.p_data) < *(rhs.p_data); }
        inline friend bool operator> (const f_pair& lhs, const f_pair& rhs){ return rhs < lhs; }
        inline friend bool operator<=(const f_pair& lhs, const f_pair& rhs){ return !(lhs > rhs); }
        inline friend bool operator>=(const f_pair& lhs, const f_pair& rhs){ return !(lhs < rhs); }
    };

    /*!brief an iterator for easily accessing feature data */
    class iterator : public std::iterator<std::random_access_iterator_tag, f_pair>
    {
        typedef std::iterator<std::random_access_iterator_tag, f_pair>::difference_type diffference_type;
        // Lifecycle:
        public:
            iterator() {}
            iterator(std::vector<i_type>::iterator index_iter, std::vector<d_type>::iterator feat_iter) : index_iter(index_iter), feat_iter(feat_iter) {}
            iterator(const iterator &rhs) : index_iter(rhs.index_iter), feat_iter(rhs.feat_iter) {}

        // Operators : misc
        public:
            inline iterator& operator+=(const int& rhs) {index_iter += rhs; feat_iter += rhs; return *this;}
            inline iterator& operator-=(const int& rhs) {index_iter -= rhs; feat_iter -= rhs; return *this;}
            inline f_pair operator*() {return f_pair(&*index_iter, &*feat_iter);}
            // inline f_pair operator->() {return f_pair(*index_iter, *feat_iter);}
            inline f_pair operator[](const difference_type& rhs) {return f_pair(&index_iter[rhs], &feat_iter[rhs]);}

        // Operators : arithmetic
        public:
            inline iterator& operator++() {++index_iter; ++feat_iter; return *this;}
            inline iterator& operator--() {--index_iter; --feat_iter; return *this;}
            inline iterator operator++(int) {iterator tmp(*this); ++index_iter; ++feat_iter; return tmp;}
            inline iterator operator--(int) {iterator tmp(*this); --index_iter; --feat_iter; return tmp;}
            inline difference_type operator-(const iterator& rhs) {return index_iter - rhs.index_iter;}
            inline iterator operator+(difference_type rhs) const {return iterator(index_iter + rhs, feat_iter + rhs);}
            inline iterator operator-(difference_type rhs) const {return iterator(index_iter - rhs, feat_iter - rhs);}
            // inline iterator operator+(const int& rhs) {return iterator(index_iter + rhs, feat_iter + rhs);}
            // inline iterator operator-(const int& rhs) {return iterator(index_iter - rhs, feat_iter - rhs);}
            // friend inline iterator operator+(const int& lhs, const iterator& rhs) {return iterator(lhs + rhs.index_iter, lhs + rhs.feat_iter);}
            // friend inline iterator operator-(const int& lhs, const iterator& rhs) {return iterator(lhs - rhs.index_iter, lhs - rhs.feat_iter);}

        // Operators : comparison
        public:
            inline bool operator==(const iterator& rhs) {return index_iter == rhs.index_iter;}
            inline bool operator!=(const iterator& rhs) {return index_iter != rhs.index_iter;}
            inline bool operator>(const iterator& rhs) {return index_iter > rhs.index_iter;}
            inline bool operator<(const iterator& rhs) {return index_iter < rhs.index_iter;}
            inline bool operator>=(const iterator& rhs) {return index_iter >= rhs.index_iter;}
            inline bool operator<=(const iterator& rhs) {return index_iter <= rhs.index_iter;}

        // Data members
        protected:
            std::vector<i_type>::iterator index_iter;
            std::vector<d_type>::iterator feat_iter;
    };

    iterator begin() {return iterator(f_index.begin(), f_data.begin());}
    iterator end() {return iterator(f_index.end(), f_data.end());}

    private:
    /*!brief comparison function for sort values of each feature */
        struct feature_comp {
            const std::vector<d_type>& vals;
            feature_comp (const std::vector<d_type>& vals) : vals(vals) {
            }
            bool operator()(i_type i1, i_type i2) {
                return vals[i1] < vals[i2];
            }
        };
};

/*!brief synthetic feature (generate a random feature) */
struct synth_feature : public feature {
    synth_feature (size_t feature_size, size_t feature_id) : feature(feature_size, feature_id) {
        // generate features with random numbers
        std::generate(f_data.begin(), f_data.end(), std::rand);
        // f_index is a list of 0,1,2,3,...
        // std::iota(f_index.begin(), f_index.end(), 0);
    }

};

/*!brief Data structure holding the entire training data */
struct example_data {
    /*!brief All feature data */
    std::vector<feature> features_data;
    /*!brief All feature data transpose */
    std::vector<feature> data_mat;
    /*!brief number of features*/
    size_t n_feat;
    /*!brief Label for each example*/
    std::vector<d_type> y;
    /*!brief Node ID for each example */
    std::vector<int> nid;
    /*!brief Gradient statistics */
    std::vector<d_type> grad;
    /*!brief summation of all gradients */
    d_type total_grad;
    /*!brief Hessian statistics */
    std::vector<d_type> hess;
    /*!brief summation of all hessians */
    d_type total_hess;
    example_data() : features_data(0), n_feat(0), y(0), grad(0), total_grad(0.0), hess(0), total_hess(0.0) {
        y.reserve(1024);
        features_data.reserve(1024);
    }
    void add_y(d_type y_val) {
        // if (fabs(y_val) < 1e-10)
        //     y_val = -1.0;
        y.push_back(y_val);
    }
    // void finalize(loss_func& loss) {
    void finalize(void) {
        /* resize to the real size */
        features_data.resize(n_feat);
        /* assigned feature ID to each feature */
        for (size_t i = 0; i < features_data.size(); ++i) {
           features_data[i].feature_id = i; 
        }
        /* all examples assigned to node 1 (root node) */
        nid.resize(y.size());
        std::fill(nid.begin(), nid.end(), 1);
        /* initialize gradient and hessian statistics */
        /*
        grad.resize(y.size());
        hess.resize(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            grad[i] = loss.grad(y[i], 0.0);
            total_grad += grad[i];
            hess[i] = loss.hess(y[i], 0.0);
            total_hess += hess[i];
            #if DEBUG > 1
            std::cout << "index:" << i << ",label:" << y[i] << ",grad:" << grad[i] << ",hess:" << hess[i] << std::endl;
            #endif
        }
        */
    }
    inline void add_feature_val(i_type example_index, i_type feature_index, d_type feature_value) {
        /* if this feature is seen first time, add it to the map */
        if (n_feat <= feature_index) {
            n_feat = feature_index + 1; // assumes feature_index starts from 0
        }
        if (features_data.size() <= feature_index) {
            // features_data.resize(feature_index + 1);
            // use exponential growth
            features_data.resize(feature_index * 2 + 1);
        }
        features_data[feature_index].add(example_index, feature_value);
        if (data_mat.size() <= example_index) {
            // features_data.resize(feature_index + 1);
            // use exponential growth
            data_mat.resize(example_index+1);
        }
        data_mat[example_index].add(feature_index, feature_value);
    }
};


