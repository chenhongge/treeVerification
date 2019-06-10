#include <iostream>
#include "json/single_include/nlohmann/json.hpp"
#include <fstream>
#include <iomanip>
#include <string> 
#include <limits>
#include <algorithm>
#include <vector>
#include <math.h>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

template<typename Tkey, typename Tval>
using interval_map = std::unordered_map<Tkey, Tval>;


struct Interval{
  double lower;
  double upper;
};


class Leaf{
  public:
  interval_map<int,Interval> box;
  int nodeid;
  int treeid;
  double value;
  int class_label;
  // class_label<0 means this is a binary model
  Leaf(interval_map<int,Interval> b, int tid, int nid, double val, int class_l){
    box = b;
    nodeid = nid;
    treeid = tid;
    value = val;
    class_label = class_l;
  }
  string represent() const {
    stringstream ss;
    ss << "node id: " << nodeid << ", tree id: " << treeid << ", label: " << class_label << ", value: " << value << std::endl << "{";
    for (auto it = box.begin(); it != box.end(); ++it) {
      ss << it->first << ": [" << it->second.lower << ", " << it->second.upper <<"], ";
    }
    ss << "}";
    return ss.str();
  }
};

void print_trees(const vector<vector<Leaf>>& trees) {
  int counter = 0;
  for (const auto& i : trees) {
    cout << "tree " << counter++ << std::endl;
    for (const auto& j : i) {
      cout << j.represent() << std::endl;
    }
  }
}

void print_concrete(const vector<Leaf>& tree, const vector<double>& x, int feature_start, double factor) {
  int best_idx = -1;
  int idx = 0;
  double best_score = numeric_limits<double>::min();
  for (const auto& i : tree) {
    if (i.value * factor > best_score) {
      best_score = i.value * factor;
      best_idx = idx;
    }
    ++idx;
  }
  cout << "Best score is " << best_score / factor << endl;
  auto box = tree[best_idx].box;
  double eps = 0.0;
  for(int j = 0; j < x.size(); ++j) {
    // feature in box starts with feature_start rather than 0
    auto i = box.find(j + feature_start);
    if (i != box.end()) {
      // if this feature exist for this leaf
      double val;
      if (x[j] > i->second.lower && x[j] < i->second.upper) {
        // x is within the box, use original value
        val = x[j];
      }
      else if (x[j] > i->second.upper) {
        val = i->second.upper - 1e-5;
      }
      else if (x[j] < i->second.lower) {
        val = i->second.lower + 1e-5;
      }
      cout << j + feature_start << ":" << setprecision(16) << val << " ";
      // cout << "j = " << j << ", " << x[j] << ", " << i->second.lower << ", " << i->second.upper << endl;
      // double check the epsilon
      eps = max(abs(val - x[j]), eps);
      // cout << abs(val - x[j]) << endl;
    }
    else {
      cout << j + feature_start << ":" << setprecision(16) << x[j] << " ";
    }
  }
  cout << endl;
  cout << "eps = " << eps << endl;
}

void print_box (const interval_map<int, Interval>& x){
  cout << '{';
  for(interval_map<int, Interval>::const_iterator it = x.begin(); it != x.end(); ++it)
    cout << it->first << ": [" << it->second.lower << ", " << it->second.upper <<"], ";
  cout << "}";
}


interval_map<int,Interval> build_1D_box (int attribute, double lower_bound, double upper_bound){
  // build a 1D box by constaining a single feature
  interval_map<int, Interval>  box;
  Interval interval = {lower_bound, upper_bound};
  box[attribute] = interval;
  return box;
}


bool box_intersec (interval_map<int,Interval>& box1, const interval_map<int,Interval>& box2){
  //intersect two boxes to box1
  //has key -100 means empty box, feature number should not be less than 0
  //note that empty interval_map means no constraint, an infinitely large box
  //return a bool: if the result is not empty
  if (box1.find(-100) != box1.end())
    return false; 
  if (box2.find(-100) != box2.end()){
    box1.clear();
    Interval empty_interval = {0,0};
    box1[-100] = empty_interval;
    return false;
  } 
  for (interval_map<int,Interval>::const_iterator it = box2.cbegin(); it != box2.cend(); ++it) {
      int key = it->first; 
      auto b1 = box1.find(key);
      if (b1 == box1.end()){
        box1[key] = it->second;
      }
      else{
        
        double l1 = b1->second.lower;
        double l2 = it->second.lower; // box2[key].lower; 
        double u1 = b1->second.upper;
        double u2 = it->second.upper; // box2[key].upper;
        double l = max(l1, l2);
        double u = min(u1, u2);
        Interval interval = {l,u};
        if (l >= u) {
          box1.clear();
          Interval empty_interval = {0,0};
          box1[-100] = empty_interval;
          return false;
        }
        b1->second = interval;
      }
  }
  return true;
}


double point_interval_dist(double att_val, double l, double u, double order){
  double dist = 0;
  if (att_val > u){
    if (order>0){
      dist = pow(att_val - u, order); 
    }
    else{
      if (order==0){
        dist = 1;
      }
      else{
        dist = att_val  - u;
      }
    }
  }
  if (att_val < l){
    if (order>0){
      dist = pow(l - att_val, order); 
    }
    else {
      if (order==0){ 
        dist = 1;
      }
      else{
        dist = l - att_val;
      }
    }
  }
  return dist;
}


double point_box_dist(const vector<double>& p, const interval_map<int,Interval>& b, double order, int feature_start, bool one_attr, int only_attr){
  // order < 0 means linf norm
  double res = 0;
  double dist = 0;
  for (interval_map<int, Interval>::const_iterator it = b.cbegin(); it != b.cend(); ++it) {
    int attr = it->first; 
    attr = attr - feature_start;
    if (attr >= p.size()){
      char buffer [100];
      sprintf (buffer, "point dimension is %d box has attribute  %d", int(p.size()), attr);
      cout << "point dimension is " << p.size() << ", box has attribute " << attr << '\n';
      throw invalid_argument(buffer);
      //throw invalid_argument("box and point are not compatible");
    }
    double l = it->second.lower;
    double u = it->second.upper; 
    dist = point_interval_dist(p[attr], l, u, order);
    if (one_attr && only_attr != attr && dist > 0){
      return std::numeric_limits<double>::max(); 
    }
    if (order>=0){
      res = res + dist;
    }
    else {
      res = max(res, dist);
    }
  } 
  if (order>0){
    return pow(res, 1/order);
  }
  else{
    if (order==0){
      return double(int(res));
    }
    else{
      return res;
    }
  }
}


void read_libsvm(string data_file, vector<vector<double>>&X, vector<int>&y, bool binary){
  X.clear();
  y.clear();
  example_data test_data;
  svm_reader reader(data_file, test_data);
  reader.load();
  cout << "\ndata shape:" << test_data.data_mat.size() << " * " << test_data.data_mat[0].f_data.size() <<'\n';
  for (int i=0; i< test_data.data_mat.size(); i++){
    y.push_back(test_data.y[i]);
    X.push_back(test_data.data_mat[i].f_data); 
  }
}


void print_slice(vector<vector<double>> X, vector<int> y, int start_idx, int end_idx){
  for (int i=start_idx; i< end_idx; i++){
    cout << '\n' << y[i] << ":\t";
    for (int j=0; j < X[i].size(); j++){
      cout << X[i][j]  <<'\t';
    }
  }
  cout << '\n';
}


struct compare_length {
  //compare the length of two vectors
  bool operator()(const vector<Leaf>& first, const vector<Leaf>& second) {
    return first.size() > second.size();
  }
};
