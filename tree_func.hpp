#include <iostream>
//#include "json/single_include/nlohmann/json.hpp"
#include <fstream>
#include <iomanip>
#include <string> 
#include <tuple>
#include <limits>
#include <algorithm>
#include <vector>
#include <math.h>
#include <chrono>
#include <random>
#include "box.hpp"

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;



void dfs (json tree, int treeid, interval_map<int, Interval> p_box, vector<Leaf>& Leaf_vec, int class_label){
  // class_label<0 means this is a binary model
  // p_box is the parent node's bounding box 
  if (tree.find("leaf") != tree.end()){
    double leaf_val_num = tree["leaf"];
    Leaf_vec.push_back(Leaf(p_box, treeid, int(tree["nodeid"]), double(leaf_val_num), class_label));
  }
  else {
      int attr = tree["split"]; 
      double threshold = double(tree["split_condition"]); 
      int nodeid = int(tree["nodeid"]);
      json left_subtree;
      json right_subtree;
      if (int(tree["children"][0]["nodeid"] == int(tree["yes"])) && int(tree["children"][1]["nodeid"] == int(tree["no"]))) {
        left_subtree = tree["children"][0];
        right_subtree = tree["children"][1];
      }
      else if (int(tree["children"][1]["nodeid"] == int(tree["yes"])) && int(tree["children"][0]["nodeid"] == int(tree["no"]))){
        left_subtree = tree["children"][1];
        right_subtree = tree["children"][0];
      }
      else{
        throw invalid_argument( "node id not match!" );
      }

      interval_map<int, Interval> left_box;
      interval_map<int, Interval> right_box;

      if (p_box.find(-100) != p_box.end()){ 
        left_box = p_box;
        right_box = p_box;
      }
      else { 
        left_box = p_box;
        right_box = p_box;
        box_intersec(left_box, build_1D_box(attr, -numeric_limits<float>::max(), threshold));
        box_intersec(right_box, build_1D_box(attr, threshold, numeric_limits<float>::max())); 
        
      }
  dfs(left_subtree, treeid, left_box, Leaf_vec, class_label);
  dfs(right_subtree, treeid, right_box, Leaf_vec, class_label);
  }
}



tuple<vector<vector<Leaf>>, double> find_k_partite_clique(vector<vector<Leaf>>all_tree_reachable_leaves, int max_clique, double eps, int label, int neg_label, int num_classes, bool dp){
  // label is the point's true label
  // if a leaf's label is neg_label, we minus instead of add
  // neg_label is valid only if it's >=0
  // if dp is true, use dynamic programmnig to compute the final sum. otherwise, simple sum up the max.
  vector<vector<Leaf>> new_nodes_array;
  //compare_length c;
  //sort(all_tree_reachable_leaves.begin(), all_tree_reachable_leaves.end(), c);
  ////cout << "number of reachable leaves on each tree after sort:" << '\n';
  //cout << "number of reachable leaves on each tree:" << '\n';
  //for (int i=0; i< all_tree_reachable_leaves.size(); i++){
  //  cout << all_tree_reachable_leaves[i].size() << '\t';
  //}
  //cout << '\n';
  if (dp) {
    cout << "\n[using DP]\n";
  }
  vector<tuple<interval_map<int, Interval>, double>>* DP_best_old;
  vector<tuple<interval_map<int, Interval>, double>>* DP_best_new; 
  vector<tuple<interval_map<int, Interval>, double>> DP_buffer[2];
  int dp_buf_idx = 0;
  DP_best_old = &DP_buffer[0];
  DP_best_new = &DP_buffer[1];
  vector<double> best_scores;

  for (int start_tree=0; start_tree < all_tree_reachable_leaves.size(); start_tree = start_tree+max_clique){ 
    // finding cliques 
    //cout << "\n---------------------------------- " << start_tree << " to " << min(int(all_tree_reachable_leaves.size()), start_tree+max_clique)-1  << " clique finding loop starts----------------------------------------"<<'\n'; 
    vector<tuple<interval_map<int, Interval>, double>>* LL_old;
    vector<tuple<interval_map<int, Interval>, double>>* LL_new; 
    vector<tuple<interval_map<int, Interval>, double>> buffer[2];

    LL_old = &buffer[0];
    LL_new = &buffer[1];
    int buf_idx = 0;

    //each element of LL_old/LL_new is a tuple of the intersection box of the clique and sum value 
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for (int m=0; m < all_tree_reachable_leaves[start_tree].size(); m++){ 
      double new_leaf_value;
      if (num_classes>2 && neg_label>=0 && all_tree_reachable_leaves[start_tree][m].class_label == neg_label){
        new_leaf_value = - all_tree_reachable_leaves[start_tree][m].value;
      }
      else{
        new_leaf_value = all_tree_reachable_leaves[start_tree][m].value;
      }
      LL_old->emplace_back(make_tuple(all_tree_reachable_leaves[start_tree][m].box, new_leaf_value));
    }
    
    ////cout << "LL_old size: " << LL_old->size() <<  '\n';
    for (int k=start_tree+1; k < min(int(all_tree_reachable_leaves.size()), start_tree+max_clique); k++){//loop all trees
      //cout << "\n\n\n" << k << "th tree starts:" << '\n';
      LL_new->clear();
      for (int j=0; j < LL_old->size(); j++){//loop all previous cliques
        ////cout << "\n\n" << j << "th clique starts:" << '\n'; 
        for (int m=0; m < all_tree_reachable_leaves[k].size(); m++){//loop nodes in new trees
          ////cout  << '\n' << m << "th node starts:" << '\t';
          interval_map<int, Interval> intersection = all_tree_reachable_leaves[k][m].box;
          if (box_intersec(intersection, get<0>((*LL_old)[j]))){
            double new_leaf_value;
            if (num_classes>2 && neg_label>=0 && all_tree_reachable_leaves[k][m].class_label == neg_label){
              new_leaf_value = - all_tree_reachable_leaves[k][m].value; 
            } 
            else{
              new_leaf_value = all_tree_reachable_leaves[k][m].value;
            }
            LL_new->emplace_back(make_tuple(std::move(intersection), new_leaf_value + get<1>((*LL_old)[j])));
            // if (LL_new->size()>=10000 && LL_new->size()%10000==0) {
            //   cout << "LL_new size=" << LL_new->size() <<'\n';
            // }
          }
          
        }
      }
      
      // LL_old = LL_new;
      // swap two buffers, avoids copy
      LL_old = &buffer[(++buf_idx) & 1];
      LL_new = &buffer[(buf_idx+1) & 1];
      //cout << "number of cliques from  "<< start_tree << "th tree to "<< k << "th trees: " << LL_old->size() << '\n';
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //cout << "----------------------------------clique finding loop ends, final number of cliques: " << int(LL_old->size()) << " ----------------------------------------"<<'\n';
    if (dp){
      if(start_tree==0){
        DP_best_old->clear();
        for (int i=0; i<LL_old->size(); i++){
          DP_best_old->emplace_back((*LL_old)[i]);
        }
      }
      else{
        DP_best_new->clear();
        for (int i=0; i<LL_old->size(); i++){
          double node_best;
          if (label<0.5 && num_classes<=2){
            node_best = - std::numeric_limits<float>::max();
          }
          else{
            node_best = std::numeric_limits<float>::max();
          }
          for (int j=0; j<DP_best_old->size(); j++){
            interval_map<int, Interval> tmp_box = get<0>((*LL_old)[i]);
            if (!box_intersec(tmp_box, get<0>((*DP_best_old)[j]))){
              continue;
            }
            if (label<0.5 && num_classes<=2){
              node_best = max(node_best, get<1>((*LL_old)[i])+get<1>((*DP_best_old)[j]));
            }
            else{
              node_best = min(node_best, get<1>((*LL_old)[i])+get<1>((*DP_best_old)[j]));
            }
          }
          DP_best_new->emplace_back(make_tuple(get<0>((*LL_old)[i]), node_best));
        }
        DP_best_old = &DP_buffer[(++dp_buf_idx) & 1];
        DP_best_new = &DP_buffer[(dp_buf_idx+1) & 1];
      }
    } 
    double best_score;
    new_nodes_array.push_back(vector<Leaf>());
    vector<Leaf>& new_nodes = new_nodes_array.back(); 
    for (int i=0; i<LL_old->size(); i++){ 
      double score_sum = get<1>((*LL_old)[i]);
      new_nodes.emplace_back(Leaf(get<0>((*LL_old)[i]), -1, -1, score_sum, -1));//set a fake node with no treeid and no nodeid
      
      if (i==0){
        best_score = score_sum;
      }
      else{
        if (label<0.5 && num_classes<=2){
          best_score = max(best_score, score_sum);
        }
        else{
          best_score = min(best_score, score_sum);
        }
      }
      //cout << "\t score sum: " << score_sum << "\t best score: " << best_score <<'\n';
    }
    best_scores.push_back(best_score);
    //high_resolution_clock::time_point t3 = high_resolution_clock::now();
    //cout << "time of the for loops and after for loops:" << duration_cast<microseconds>( t2 - t1 ).count() << ", " << duration_cast<microseconds>( t3 - t2 ).count() << "\n\n\n";
    //cout << '\n'; 
  }

  //cout << "\noriginal label: " << label  <<'\n';
  double sum_best;
  if (dp){ 
    if (label<0.5 && num_classes<=2){
      sum_best = - std::numeric_limits<float>::max();
    }
    else{
      sum_best = std::numeric_limits<float>::max();
    }
    for (int j=0; j<DP_best_old->size(); j++){
      if (label<0.5 && num_classes<=2){
        sum_best = max(sum_best, get<1>((*DP_best_old)[j]));
      }
      else{
        sum_best = min(sum_best, get<1>((*DP_best_old)[j]));
      }
    }
  }
  else{
    sum_best = 0;
    //cout <<"best scores:\t";
    for (int i=0; i<best_scores.size(); i++){
      //cout << best_scores[i] <<'\t';
      sum_best = sum_best + best_scores[i];
    }
  }
  return make_tuple(new_nodes_array, sum_best);
}



vector<vector<Leaf>> find_reachable_leaves (const vector<double>& x, vector<vector<Leaf>> all_tree_leaves, double eps, int label, int neg_label, int num_classes, int feature_start, bool one_attr, int only_attr){

  // if neg_label < 0 assume binary model, all trees are used
  cout << "all tree leaves size: " << all_tree_leaves.size() << std::endl;
  if (one_attr){
    cout << "only attribute " << only_attr << " is used!" << std::endl;
  }
  vector<vector<Leaf>> all_tree_reachable_leaves;
  vector<Leaf> one_tree_reachable_leaves;
  if (num_classes > 2 && label == neg_label && neg_label>=0)
      throw invalid_argument("multi-class model's target label and original label cannot be the same!");
  for (int i=0; i< all_tree_leaves.size(); i++){
    if (num_classes <= 2 || neg_label < 0 || ((i % num_classes) == label) || ((i % num_classes) == neg_label)){
      one_tree_reachable_leaves.clear();
      for (int j=0; j<all_tree_leaves[i].size(); j++){
        /*cout << "\n!!!!" << point_box_dist(x, all_tree_leaves[i][j].box, -1, feature_start, one_attr, only_attr)<<'\n';
        cout <<"nodeid: " << all_tree_leaves[i][j].nodeid << '\t';
        print_box(all_tree_leaves[i][j].box);
        cout << '\t';
        for(interval_map<int, Interval>::const_iterator it = all_tree_leaves[i][j].box.begin(); it != all_tree_leaves[i][j].box.end(); ++it){
          cout << it->first << ": "<< x[it->first-feature_start] << '\t';
        }
        */
        if ((all_tree_leaves[i][j].box.find(-100) == all_tree_leaves[i][j].box.end()) && point_box_dist(x, all_tree_leaves[i][j].box, -1, feature_start, one_attr, only_attr)<=eps) {
          one_tree_reachable_leaves.push_back(all_tree_leaves[i][j]);
        }
      }
      if (one_tree_reachable_leaves.size() < 1)
        throw invalid_argument("number of reachable leaves less than 1, error!");
      all_tree_reachable_leaves.push_back(one_tree_reachable_leaves);
    }
  }
  //cout << "\nnumber of trees used:  " << all_tree_reachable_leaves.size() << '\n';
  //cout << "All reacheable leaves:" << std::endl;
  //print_trees(all_tree_reachable_leaves);
  return all_tree_reachable_leaves;

} 



vector<double> find_multi_level_best_score (const vector<double>& x, int label, int neg_label, vector<vector<Leaf>> all_tree_leaves, int num_classes, int max_level, double eps, int max_clique, int feature_start, bool one_attr, int only_attr, bool must_use_dp){
  //pick the reachable leaves on each tree
  vector<vector<Leaf>> all_tree_reachable_leaves = find_reachable_leaves(x, all_tree_leaves, eps, label, neg_label, num_classes, feature_start, one_attr, only_attr);  
  //shuffle trees
  //auto rng = std::default_random_engine {};
  //std::shuffle(std::begin(all_tree_reachable_leaves), std::end(all_tree_reachable_leaves), rng);
  
  //print number of reachable leaves on each tree
  cout << "number of reachable leaves on each tree:" << '\n';
  for (int i=0; i< all_tree_reachable_leaves.size(); i++){
    cout << all_tree_reachable_leaves[i].size() << '\n';
    for (int j=0; j<all_tree_reachable_leaves[i].size();j++){
      cout<<", "<<all_tree_reachable_leaves[i][j].treeid<<","<<all_tree_reachable_leaves[i][j].nodeid;
    }
    cout<<'\n';
  }
  cout << '\n'; 
  
  vector<double> sum_best;
  vector<vector<Leaf>> new_nodes_array = all_tree_reachable_leaves;
  
  for (int l=0; l<max_level; l++){
    cout << "\n\n[level " << l << " starts]\n\n";
    bool use_dp = (l==max_level-1) && must_use_dp;
    tuple<vector<vector<Leaf>>, double> res;
    if (num_classes > 2 && l == 0)
      res = find_k_partite_clique(new_nodes_array, max_clique, eps, label, neg_label, num_classes, use_dp);
    else
      res = find_k_partite_clique(new_nodes_array, max_clique, eps, label, -1, num_classes, use_dp);
    sum_best.push_back(get<1>(res)); 
    new_nodes_array = get<0>(res); 
    if (new_nodes_array.size() <=1 ){
      //cout << "\nonly one partite left, break level "<< l <<'\n';
      cout << "reached root, print the best example found:" << std::endl;
      if (label<0.5 && num_classes<=2) {
        // print the max score leaf
        // print_concrete(new_nodes_array[0], x, feature_start, +1);
      }
      else {
        // print the min score leaf
        // print_concrete(new_nodes_array[0], x, feature_start, -1);
      }
      break;
    }
  }

 return sum_best; 
}











