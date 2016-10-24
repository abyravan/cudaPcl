/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace jsc {

template<uint32_t D>
double vMF<D>::log2SinhOverZ(double tau) {
  if (tau > 10.) {
    return - log(tau) + tau;
  } else {
    return log(2.+tau*tau/3.+tau*tau*tau*tau/50.
        + tau*tau*tau*tau*tau*tau/2520.);
  }
}

template <uint32_t D>
vMF<D>::vMF(const Eigen::Matrix<double, D, 1>& mu, double tau, double
    pi) : mu_(mu), tau_(tau), pi_(pi)
{}

template <uint32_t D>
double vMF<D>::logPdf(const Eigen::Matrix<double, D, 1>& x) const {
  return GetLogZ() + tau_*mu_.dot(x);
}

template <uint32_t D>
double vMF<D>::GetLogZ() const {
  return -log2SinhOverZ(tau_) - log(2.*M_PI);
}

template <uint32_t D>
double vMF<D>::MLEstimateTau(const Eigen::Vector3d& xSum, const
    Eigen::Vector3d& mu, double count) {
  double tau = 1.0;
  double prevTau = 0.;
  double eps = 1e-8;
  double R = xSum.norm()/count;
  while (fabs(tau - prevTau) > eps) {
//    std::cout << "tau " << tau << " R " << R << std::endl;
    double inv_tanh_tau = 1./tanh(tau);
    double inv_tau = 1./tau;
    double f = -inv_tau + inv_tanh_tau - R;
    double df = inv_tau*inv_tau - inv_tanh_tau*inv_tanh_tau + 1.;
    prevTau = tau;
    tau -= f/df;
  }
  return tau;
};

}
