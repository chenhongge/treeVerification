#include <iostream>
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
#include "svmreader.hpp"
#include "tree_func.hpp"

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;


int main(int argc, char** argv){

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  string config_file = string(argv[1]);
  ifstream config(config_file);
  json param;
  config >> param;
  
  string ori_file;
  string tree_file;
  int start_idx;
  int num_attack;
  double eps_init;
  int max_clique;
  int max_search;
  int max_level;
  int num_classes;
  bool dp;
  bool one_attr;
  int only_attr;
  int feature_start;

  if (param.find("inputs") != param.end()){
    ori_file = param["inputs"];
  }
  else {
    throw invalid_argument("inputs datapoints in LIBSVM format is missing");
  }

  if (param.find("model") != param.end()){
    tree_file = param["model"];
  }
  else {
    throw invalid_argument("model is missing in config file");
  }

  if (param.find("start_idx") != param.end()){
    start_idx = int(param["start_idx"]);
  }
  else {
    throw invalid_argument("start_idx is missing in config file");
  }

  if (param.find("num_attack") != param.end()){
    num_attack = int(param["num_attack"]);
  }
  else {
    throw invalid_argument("num_attack is missing in config file");
  }

  if (param.find("eps_init") != param.end()){
    eps_init = double(param["eps_init"]);
  }
  else {
    throw invalid_argument("eps_init is missing in config file");
  }

  if (param.find("max_clique") != param.end()){
    max_clique = int(param["max_clique"]);
  }
  else {
    throw invalid_argument("max_clique is missing in config file");
  }

  if (param.find("max_search") != param.end()){
    max_search = int(param["max_search"]);
  }
  else {
    throw invalid_argument("max_search is missing in config file");
  }

  if (param.find("max_level") != param.end()){
    max_level = int(param["max_level"]);
  }
  else {
    throw invalid_argument("max_level is missing in config file");
  }

  if (param.find("num_classes") != param.end()){
    num_classes = int(param["num_classes"]);
  }
  else {
    throw invalid_argument("num_classes is missing in config file");
  }

  if (param.find("dp") != param.end()){
    dp = bool(int(param["num_classes"]));
  }
  else {
    dp = false;
  }
  
  if (param.find("one_attr") != param.end()){
    one_attr = true;
    only_attr = int(param["one_attr"]);
  }
  else {
    one_attr = false;
    only_attr = -100;
  }
 
  if (param.find("feature_start") != param.end()){
    feature_start = int(param["feature_start"]);
  }
  else {
    feature_start = 1;
  }

  if (num_classes < 2) { num_classes = 2; }
  cout << "inputs: " << ori_file << "\nmodel: "<< tree_file  << "\nstart_idx: " << start_idx << "\nnum_attack: " << num_attack << "\neps_init: " << eps_init << "\nmax_clique: " << max_clique << "\nmax_search: " << max_search << "\nmax_level: " << max_level << "\nnum_classes: " << num_classes <<"\ndp: " << dp << "\none_attr: "<< one_attr << "\nonly_attr: "<< only_attr <<'\n';
  
  
  cout << "\nfeature starts at "<< feature_start << "\n";
  

  // read data inputs 
  vector<vector<double>> ori_X;
  vector<int> ori_y;  
  read_libsvm(ori_file, ori_X, ori_y, num_classes<=2);
  
  ifstream tree_data(tree_file);
  json model;
  tree_data >> model;  
   
  vector<vector<Leaf>> all_tree_leaves;
  vector<Leaf> one_tree_leaves; 
  
  //calculate and print leave bounds
  for (int i=0; i<model.size(); i++){
    interval_map<int,Interval> no_constr;
    no_constr.clear();
    one_tree_leaves.clear();
    int class_label;
    if (num_classes==2)
      class_label = -1;
    else
      class_label = i % num_classes;
    dfs(model[i], i, no_constr, one_tree_leaves, class_label);
    all_tree_leaves.push_back(one_tree_leaves);
    cout <<"\n\n" << i <<"th tree\n";
    
  } 

  high_resolution_clock::time_point t5 = high_resolution_clock::now(); 
  double avg_bound = 0;
  num_attack = min(int(ori_X.size())-start_idx, num_attack);
  cout << "number of points: "<< num_attack  << '\n';
  int n_initial_success = 0;
  for (int n=start_idx; n<num_attack+start_idx; n++){ //loop all points
    cout << "\n\n\n\n=================start index:" << start_idx << ", num of points:" << num_attack << ", current index:" << n << ", current label: "<< ori_y[n]  <<" =================\n";
    double eps = eps_init;
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    vector<bool> rob_log;
    vector<double> eps_log;
    int last_rob = -1;
    int last_unrob = -1;
    for (int search_step=0; search_step<max_search; search_step++){
      cout << "\n\n************** eps=" << eps << " starts ******************\n";
      
      bool robust = true;
      if (num_classes <= 2){ 
        cout << "\n^^^^^^^^^^^^^^^^ binary model  ^^^^^^^^^^^^^^^\n";
        vector<double> sum_best = find_multi_level_best_score(ori_X[n], ori_y[n], -1, all_tree_leaves, num_classes, max_level, eps, max_clique, feature_start, one_attr, only_attr, dp); 
        
        robust = (ori_y[n]<0.5&&sum_best.back()<0)||(ori_y[n]>0.5&&sum_best.back()>0);
      }
      else{
        cout << "\n^^^^^^^^^^^^^^^^ " << num_classes  << "  classes model  ^^^^^^^^^^^^^^^\n";
        for (int neg_label=0; neg_label<num_classes; neg_label++){
          if (neg_label != ori_y[n]){
            cout << "\n^^^^^^^^^^^^^^^^ original class: " << ori_y[n]  << " target class: " << neg_label << " starts ^^^^^^^^^^^^^^^\n";
            vector<double> sum_best = find_multi_level_best_score(ori_X[n], ori_y[n], neg_label, all_tree_leaves, num_classes, max_level, eps, max_clique, feature_start, one_attr, only_attr, dp);
            cout << "\n best score for each level:\t";
            for (int i=0;i<sum_best.size(); i++){
              cout << sum_best[i] <<'\t'; 
            }
            
            robust = robust && (sum_best.back()>0);
            if (!robust){
              break;
            }
          }
        }
      
      }
      // at the first search, evaluate the verified error 
      if (search_step == 0 && robust) {
        n_initial_success += 1;
      }
      cout << "Can model be guaranteed robust within eps " << eps << "? (0 for no, 1 for yes): " << robust  <<'\n';
      rob_log.push_back(robust);
      eps_log.push_back(eps);
      if (robust) {
        last_rob = rob_log.size() - 1;
      }
      else {
        last_unrob = rob_log.size() - 1;
      }

      if (last_rob<0) {
        eps = eps * 0.5;
      }
      else {
        if (last_unrob<0){ 
          if (eps >= 1){
            cout << "\n eps >=1, break binary search!\n";
            break;
          }
          eps = min(eps * 2.0, 1.0);
        }
        else {
          eps = 0.5 * (eps_log[last_rob] + eps_log[last_unrob]);
        }
      }

      cout << "\n**************** this eps ends, next eps:" << eps  <<" *********************\n";
    }
    
    double clique_bound = 0;
    if (last_rob>=0){
      clique_bound = eps_log[last_rob];
      avg_bound = avg_bound + clique_bound;
    }
    else{
      cout<< "\npoint "<< n << ": WARNING! no robust eps found, verification bound is set as 0 !!!!!!!!\n";
    }
    high_resolution_clock::time_point t4 = high_resolution_clock::now();
    auto point_duration = duration_cast<microseconds>( t4 - t3).count();
    cout << "=============================== end of point "<< n  <<", running time: " << point_duration  <<" microseconds, clique res: " << clique_bound << " ====================================" <<'\n';
  }
  double verified_err = 1.0 - n_initial_success / (double)num_attack;
  avg_bound = avg_bound / num_attack; 
  cout << "\nclique method average bound:" << avg_bound << endl;
  cout << "verified error at epsilon " << eps_init << " = " << verified_err << endl;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto total_duration = duration_cast<microseconds>( t2 - t1 ).count();
  cout << " total running time: " << double(total_duration)/1000000.0 << " seconds\n";
  cout << " per point running time: " << double(total_duration)/1000000.0/num_attack << " seconds\n";
  return 0;
}










