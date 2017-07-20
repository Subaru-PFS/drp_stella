#include <Eigen/Dense>
#include <unsupported/Eigen/LevenbergMarquardt>

using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{

    struct gaussian_functor : Eigen::DenseFunctor<float> {
	Eigen::VectorXf m_x0;
	Eigen::VectorXf m_y0;
 
	gaussian_functor(const Eigen::MatrixX2f & f0) : 
		Eigen::DenseFunctor<float>(3,f0.rows()), 
		m_x0(f0.col(0)),
		m_y0(f0.col(1))
					 {}
 
	int operator()(const InputType &x, ValueType &fvec)  {
	  auto num = -(m_x0 - ValueType::Constant(values(),x[1])).array().square();
		auto den = 2 * x[2] * x[2];
          
		fvec = x[0] * (num / den).exp() - m_y0.array();
		return 0;
	}
 
	int df(const InputType &x, JacobianType &fjac)  {
		auto bv = ValueType::Constant(values(), x[1]);
		auto tmp = (m_x0 - bv).array();
		auto num = -tmp.square();
		auto c2 = x[2] * x[2];
		auto den = 2 * c2;
 
		auto j0 = (num / den).exp();
        	auto j1 = x[0] * tmp * j0 / c2;
 
		fjac.col(0) = j0;
		fjac.col(1) = j1;
		fjac.col(2) = tmp * j1 / x[2];
 
		return 0;
	}
    };
}}}}
