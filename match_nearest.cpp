#include <Rcpp.h>
#include <set>

using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector match_v(NumericMatrix T, NumericMatrix C, int num_left, int num_right) {

  std::set<int> set_matches;


  for (int i = 0; i < T.nrow(); i++) {
    double vt = T(i,1);
    double d = std::numeric_limits<double>::infinity();
    int nearest = 0;
    for (int j = nearest; j < C.nrow(); j++) {
      double vc = C(j,1);
      if ((abs(vt - vc) < d) || (d == 0.0)) {
        d = abs(vt - vc);
        nearest = j;
        if (d == 0.0) {
          set_matches.insert((int) (C(j,0) + 0.5));
        }
      } else if (abs(vt-vc) > d) {
        break;
      }
    }
    set_matches.insert((int) (C(nearest,0)+0.5));
    if (num_left > 0) {
      for (int k = 1; k <= num_left; k++) {
        if (nearest - k >= 0) {
          set_matches.insert((int) (C(nearest-k,0)+0.5));
        } else {
          break;
        }
      }
    }
    if (num_right > 0) {
      for (int k = 1; k <= num_right; k++) {
        if (nearest + k < C.nrow()) {
          set_matches.insert((int) (C(nearest+k,0)+0.5));
        } else {
          break;
        }
      }
    }
  }

  int len = set_matches.size();

  IntegerVector vec_matches(len);

  int k = 0;
  for (auto elem : set_matches) {
    vec_matches(k) = elem;
    k++;
  }

  return(vec_matches);

  /*
    set_matches = set();
    for i in range(len(T)):
        vt = T[i,1]
        d = 1
        nearest = 0
        for j in range(nearest, len(C)):
            vc = C[j,1]
            if abs(vt - vc) < d:
                d = abs(vt - vc)
                nearest = j
                if d == 0:
                    set_matches.add(C[j,0])
            elif abs(vt - vc) > d:
                break
                # pass
        set_matches.add(C[nearest,0])
        if nearest > 0:
            set_matches.add(C[nearest-1,0])
        if nearest > 1:
            set_matches.add(C[nearest-2,0])
        if nearest < len(C) - 1:
            set_matches.add(C[nearest+1,0])
        if nearest < len(C) - 2:
            set_matches.add(C[nearest+2,0])
    return [int(e) for e in set_matches]
  */

}
