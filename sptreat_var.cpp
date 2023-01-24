//#include <RcppArmadillo.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;
//using namespace Rcpp;


bool P_conflict (Rcpp::NumericMatrix m1, Rcpp::NumericMatrix m2, int idx_m1, int idx_m2, Rcpp::IntegerVector S1_in_S2) {
  int nconflict = S1_in_S2.length();
  // iterate over overlap
  for (int idx_s1=0; idx_s1<nconflict; idx_s1++) {
    if (!Rcpp::IntegerVector::is_na(S1_in_S2(idx_s1))) {
      int state_s1 = m1(idx_m1,idx_s1);
      int state_s2 = m2(idx_m2,S1_in_S2(idx_s1)-1);  // Rcpp match index starts at 1 (R-style), so need to subtract 1
      if (state_s1 != state_s2) {
        return true;  // we have a conflict!
      }
    }
  }
  return false;  // no conflict found
}

// [[Rcpp::export]]
double calc_pmsa (Rcpp::List Mi, int m, int s, int a) {
  double pmsa = 0.0;
  if (a==1) {
    Rcpp::NumericMatrix pmat = Mi["pmsa1"];
    pmsa = pmat(m,s);
  } else {
    Rcpp::NumericMatrix pmat = Mi["pmsa0"];
    pmsa = pmat(m,s);
  }
  return(pmsa);
}

double calc_v (Rcpp::List M, int i, int s, int m, int a, double w, Rcpp::NumericVector p, Rcpp::NumericVector nbar) {
  Rcpp::List Mi = M[i];
  Rcpp::NumericMatrix m1 = Mi["m"];
  Rcpp::NumericVector S1 = Mi["S"];
  
  double pmsa = calc_pmsa(Mi,m,s,a);
  double visma = (1 - pmsa) * pow(p[s]/(1-p[s]),1-a) * w / nbar(a);
  Rcpp::NumericVector Iconflict = Mi["Iconflict"];
  int nconflict = Iconflict.length();
  for (int idx=0; idx<nconflict; idx++) {
    int i2 = Iconflict(idx);
    Rcpp::List Mi2 = M[i2];
    Rcpp::NumericVector S2 = Mi2["S"];
    Rcpp::IntegerVector S1_in_S2 = Rcpp::match(S1,S2);
    Rcpp::NumericMatrix m2 = Mi2["m"];
    
    Rcpp::DataFrame dat_w = Mi2["w"];
    int nw2 = dat_w.nrow();
    Rcpp::NumericVector w2vec = dat_w["w"];
    Rcpp::NumericVector s2vec = dat_w["s"];
    int nm2 = m2.nrow();
    for (int idx_m2=0; idx_m2<nm2; idx_m2++) {
      if (P_conflict(m1,m2,m,idx_m2,S1_in_S2)) {
        for (int idx_s2=0; idx_s2<nw2; idx_s2++) {
          double w2 = w2vec(idx_s2);
          int s2 = s2vec(idx_s2);
          visma = visma + calc_pmsa(Mi2,idx_m2,s2,0) * (p[s2]/(1-p[s2])) * w2 / nbar(0);
          visma = visma + calc_pmsa(Mi2,idx_m2,s2,1) * w2 / nbar(1);
        }
      }
    }
  }
  visma = pow(p[s]/(1-p[s]),1-a) * visma;
  return(visma);
}



// [[Rcpp::export]]
Rcpp::NumericVector calc_Va (Rcpp::NumericMatrix dt_out, Rcpp::NumericVector p, Rcpp::List M,
                int S, Rcpp::NumericVector muhat, Rcpp::NumericVector nbar) {
  double Vt = 0;
  double Vc = 0;
  int rows = dt_out.nrow();
  for (int idx=0; idx<rows; idx++) {
    int i = dt_out(idx,0);
    int s = dt_out(idx,1);
    //double d = dt_out(idx,2);
    double w = dt_out(idx,3);
    //double p = dt_out(idx,4);
    int A = dt_out(idx,5);
    double Y = dt_out(idx,6);
    Rcpp::List Mi = M[i];
    int m = Mi["m_realized"];  // which exposure is realized
    double visma = calc_v(M,i,s,m,A,w,p,nbar);
    if (A==1) {
      Vt = Vt + (w/nbar(1))*visma*pow(Y-muhat(1),2); //*pisma/pisma  //1(observed)/pisma
    } else {
      Vc = Vc + (w/nbar(0))*visma*pow(Y-muhat(0),2); //*pisma/pisma  //1(observed)/pisma
    }
  }
  
  Rcpp::NumericVector V (2);
  V(0) = Vc/S;
  V(1) = Vt/S;
  
  return(V);
}

