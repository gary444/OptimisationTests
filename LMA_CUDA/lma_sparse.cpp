#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <unsupported/Eigen/LevenbergMarquardt>

// #include <unsupported/Eigen/NumericalDiff>
// #include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace Eigen;

#define ADD_OUTLIER 0
#define ROBUSTIFY   1

#define NUMERICAL_GRADIENT_CALCULATION 1

#define HUBER_LOSS 1

// from SSBA repo=====================================================
double sqr(double const x) { return x*x; }
double psi(double const tau2, double const r2)
{
  double const r4 = r2*r2, tau4 = tau2*tau2;
  return (r2 < tau2) ? r2*(3.0 - 3*r2/tau2 + r4/tau4)/6.0f : tau2/6.0;
}
double psi_weight(double const tau2, double const r2)
{
  return sqr(std::max(0.0, 1.0 - r2/tau2));
}

// double sqr(double const x) { return x*x; }
// double psi(double const tau2, double const r2)
// {
//   double const r4 = r2*r2, tau4 = tau2*tau2;
//   return (r2 < tau2) ? r2*(3.0 - 3*r2/tau2 + r4/tau4)/6.0f : tau2/6.0;
// }
// double psi_weight(double const tau2, double const r2)
// {
//   return sqr(std::max(0.0, 1.0 - r2/tau2));
// }
// =====================================================
    
double const huber_delta = 0.1;

double huber_loss(double const c, double const r, bool& outlier){ // robustness constant, residual
    if (abs(r) < c){ //inlier
        outlier = false;
        return 0.5 * r * r;
    } else { //outlier
        outlier = true;
        return c*abs(r) - 0.5*c*c;
    }
}

// derivative from sepwww.stanford.edu/public/docs/sep92/jon2/paper_html/node2.html
double huber_first_derivative(double const c, double const r){ // robustness constant, residual
    if (abs(r) < c){ //inlier
        return r;
    } else { //outlier
        return r > 0 ? c : (r < 0 ? -c : 0); // c*sgn(r). sgn() is the sign function
    }
}


// =====================================================


template <typename _Scalar, typename _Index>
struct CustomSparseFunctor : public SparseFunctor<_Scalar, _Index>
{
	typedef _Scalar Scalar;
	typedef _Index Index;
	typedef Matrix<Scalar,Dynamic,1> InputType;
	typedef Matrix<Scalar,Dynamic,1> ValueType;
	typedef SparseMatrix<Scalar, ColMajor, Index> JacobianType;

	typedef SparseQR<JacobianType, COLAMDOrdering<int> > QRSolver;
	// enum {
	// 	InputsAtCompileTime = Dynamic,
	// 	ValuesAtCompileTime = Dynamic
	// };

	// CustomSparseFunctor(int inputs, int values) : n(inputs), m(values) {}
	CustomSparseFunctor(int inputs, int values) : SparseFunctor<_Scalar, _Index>(inputs, values) {}

	// int inputs() const { return m; }
	// int values() const { return n; }

	// const int m, n;
	// to be defined in the functor

	   // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
	// int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
	int operator()(const InputType &x, ValueType& fvec)
	{
	    // find the residuals / evaluate each term / calculate squared difference at each data point
	    // with given parameters in x
	    // store in fvec

	    // function is ax^2 + bx + c
        double a = x[0];
        double b = x[1];
        double c = x[2];
        double d = x[3];

        uint32_t outliers = 0;

        for (Index i = 0; i < 100; ++i)
        {
            fvec[i] = i;
        }


	    for (Index m_i = 0; m_i < Index(this->values()); ++m_i) {


	        double x_val = x_data[m_i];
	        double y_val = y_data[m_i];
	        
	        if (m_i < Index(50)){

                // std::cout << "a" << a << std::endl;
                // std::cout << "b" << b << std::endl;
                // std::cout << "x_val" << x_val << std::endl;
                // std::cout << "y_val" << y_val << std::endl;

	        	fvec[m_i] = y_val - exp(a*x_val*x_val+b*x_val);

                // std::cout << "result" << fvec[m_i] << std::endl;


	        } else {

                // std::cout << "c" << c << std::endl;
                // std::cout << "d" << d << std::endl;
                // std::cout << "x_val" << x_val << std::endl;
                // std::cout << "y_val" << y_val << std::endl;

	        	fvec[m_i] = y_val - exp(c*x_val*x_val+d*x_val);

                // std::cout << "result" << fvec[m_i] << std::endl;

	        }

            if (ROBUSTIFY) {
                // multiply residual by sqaure root of robustified kernel divided by length of residual
                // double const sqrt_psi = sqrt(psi(m_sqrInlierThreshold, sqrNorm_L2(r)));
                // double const rnorm_r = 1.0 / std::max(eps_psi_residual, norm_L2(r));
                // e[0] *= sqrt_psi * rnorm_r;
                // e[1] *= sqrt_psi * rnorm_r;

                // residuals are 1 dimensional in this case
                double res = fvec[m_i];
#if HUBER_LOSS
                bool is_outlier;
                double const sqrt_psi = sqrt(huber_loss(huber_delta, res, is_outlier ));
                if (is_outlier) ++outliers;
#else
                double const sqrt_psi = sqrt(psi(m_sqrInlierThreshold, res*res ));
#endif           
                double const rnorm_r  = 1.0 / std::max(eps_psi_residual, abs(res) ) ;


                // std::cout << "--- " << std::endl;
                // std::cout << "in: " << res << std::endl;
                // std::cout << "sqrt_psi: " << sqrt_psi << std::endl;
                // std::cout << "rnorm_r: " << rnorm_r << std::endl;


                fvec[m_i] = double(res * sqrt_psi * rnorm_r);

                // std::cout << "out: " << fvec[m_i] << std::endl;
                // std::cout << "--- " << std::endl;

            }

	    }

        std::cout << "Fvec\n" << fvec << std::endl;

#if HUBER_LOSS
        std::cout << "Outliers (huber) : " << outliers << std::endl;
#endif



	    return 0;

	}


    // Compute the jacobian of the errors
    // int df(const Eigen::VectorXd &x, Eigen::MatrixXf &fjac) const
	int df(const InputType &x, JacobianType& fjac)
    {
        // find partial derivatives of each term / data point WRT each parameter in x
        double a = x[0];
        double b = x[1];
        double c = x[2];
        double d = x[3];


        const float epsilon = 1e-5f;


        if (NUMERICAL_GRADIENT_CALCULATION){



            for (Index m_i = 0; m_i < Index(this->values()); ++m_i)
            {
                double x_val = x_data[m_i];
    	        double y_val = y_data[m_i];

                double pds[2];
                double r;
    	       
                if (m_i < 50) {
    	            pds[0] = -x_val*x_val*exp(a*x_val*x_val+b*x_val);
    	            pds[1] = -x_val*exp(a*x_val*x_val+b*x_val);

                    r = y_val - exp(a*x_val*x_val+b*x_val);

                } else {

    	            pds[0] = -x_val*x_val*exp(c*x_val*x_val+d*x_val);
    	            pds[1] = -x_val*exp(c*x_val*x_val+d*x_val);

                    r = y_val - exp(c*x_val*x_val+d*x_val);
    	        }

                if (ROBUSTIFY){

                    // Vector2d const q = this->projectPoint(_Xs[point], view);
                    // Vector2d const r = q - _measurements[k];

                    // double const r2 = sqrNorm_L2(r);
                    // double const W = psi_weight(m_sqrInlierThreshold, r2);
                    // double const sqrt_psi = sqrt(psi(m_sqrInlierThreshold, r2));
                    // double const rsqrt_psi = 1.0 / std::max(eps_psi_residual, sqrt_psi);

                    // Matrix2x2d outer_deriv, r_rt, rI;
                    // double const rcp_r2 = 1.0 / std::max(eps_psi_residual, r2);
                    // double const rnorm_r = 1.0 / std::max(eps_psi_residual, sqrt(r2));
                    // makeOuterProductMatrix(r, r_rt); scaleMatrixIP(rnorm_r, r_rt);
                    // makeIdentityMatrix(rI); scaleMatrixIP(sqrt(r2), rI);
                    // outer_deriv = W/2.0*rsqrt_psi * r_rt + sqrt_psi * rcp_r2 * (rI - r_rt);

                    // Matrix<double> J(Jdst.num_rows(), Jdst.num_cols());
                    // copyMatrix(Jdst, J);
                    // multiply_A_B(outer_deriv, J, Jdst);

                    // first attempt
                    // double const r2 = sqr(r);
                    // double const W = psi_weight(m_sqrInlierThreshold, r2);
                    // double const sqrt_psi = sqrt(psi(m_sqrInlierThreshold, r2));
                    // double const rsqrt_psi = 1.0 / std::max(eps_psi_residual, sqrt_psi);
                    // double const rcp_r2 = 1.0 / std::max(eps_psi_residual, r2);
                    // double const rnorm_r = 1.0 / std::max(eps_psi_residual, double(r));
                    // double const r_rt = r2;
                    // double const rI = r;
                    // double const deriv = W/2.0*rsqrt_psi * r_rt + sqrt_psi * rcp_r2 * (rI - r_rt);

                    double const r_abs = abs(r);
                    double const rcp_r_abs = 1.0 / std::max(eps_psi_residual, r_abs);
                    double const r2 = sqr(r);
                    double const r3 = r_abs*r_abs*r_abs;
                    double const rcp_r3 = 1.0 / std::max(eps_psi_residual, r3);
#if HUBER_LOSS
                    double const W = huber_first_derivative(huber_delta, r);
                    bool is_outlier_dummy;
                    double const sqrt_psi = sqrt(huber_loss(huber_delta, r, is_outlier_dummy));
#else
                    double const W = psi_weight(m_sqrInlierThreshold, r);
                    double const sqrt_psi = sqrt(psi(m_sqrInlierThreshold, r));
#endif            
                    double const rsqrt_psi = 1.0 / std::max(eps_psi_residual, sqrt_psi);
                    double const rcp_r2 = 1.0 / std::max(eps_psi_residual, r2);
                    double const rnorm_r = 1.0 / std::max(eps_psi_residual, double(r));
                    double const r_rt = r2;
                    double const rI = r;

                    // 2nd term seems to cancel out if there is only one component of the residual
                    // double const deriv = 0.5*rcp_r_abs*rsqrt_psi * r * W + sqrt_psi * rcp_r3 * (r2 - r_rt);
                    double const deriv = 0.5 * rcp_r_abs * rsqrt_psi * r * W;

                    pds[0] *= deriv;
                    pds[1] *= deriv;

                }





                if (m_i < 50) {

                    fjac.coeffRef(m_i, 0) = pds[0];
                    fjac.coeffRef(m_i, 1) = pds[1];
                                    
                } else {

                    fjac.coeffRef(m_i, 2) = pds[0];
                    fjac.coeffRef(m_i, 3) = pds[1];
                }


            }
        } else {
            // direct evaluation of derivatives   

                std::cout << "Direct evaluation" << std::endl;

                // for each parameter, evaluate fVec with adjusted parameter values

                for (int i = 0; i < x.size(); i++) {

                    Eigen::VectorXd xPlus(x);
                    xPlus(i) += epsilon;
                    Eigen::VectorXd xMinus(x);
                    xMinus(i) -= epsilon;

                    Eigen::VectorXd fvecPlus(this->values());
                    operator()(xPlus, fvecPlus);

                    Eigen::VectorXd fvecMinus(this->values());
                    operator()(xMinus, fvecMinus);

                    Eigen::VectorXd fvecDiff(this->values());
                    fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

                    // fjac.block(0, i, this->values(), 1) = fvecDiff;

                    for (int row = 0; row < this->values(); ++row)
                    {
                        if (fvecDiff(row) != 0.0) {
                            fjac.coeffRef(row, i) = fvecDiff(row);
                        }
                    }
                }

        }

        // std::cout << "fjac\n" << fjac << std::endl;
        fjac.makeCompressed();
        return 0;


    }

    vector<double> x_data;
    vector<double> y_data;

    const double m_sqrInlierThreshold = 1.f;
    // const double m_sqrInlierThreshold = 10.f;
    const double eps_psi_residual = 1e-20;

};


int main(int argc, char **argv) {


    double ar = 1.5, br = 2.0, cr = 1.6, dr = 2.1;         // real parameter value
    double ae = 1.5, be = 2.0, ce = 1.5, de = 2.0;        // Estimated parameter value
    int N = 100;                                 // data point
    double w_sigma = 0.2;                        // Noise Sigma value
    cv::RNG rng;                                 // OpenCV random number generator


    std::cout << "Actual paramaters: " << std::endl;
    std::cout << "A: " << ar << std::endl;
    std::cout << "b: " << br << std::endl;
    std::cout << "c: " << cr << std::endl;
    std::cout << "d: " << dr << std::endl;

    // for LMA
    int n = 4;//num parameters
    int m = N;//num constraints (terms)
    Eigen::VectorXd x_vec (4); // parameter values (initial)    

    x_vec(0) = ae;
    x_vec(1) = be;
    x_vec(2) = ce;
    x_vec(3) = de;

    vector<double> x_data, y_data;      // data
    for (int i = 0; i < N; i++) {
        // double x = i / double(N) * 0.5;

        double x = rng.uniform((double)0, (double)0.5);

        // double x = i / double(N);
        x_data.push_back(x);

        if (i < 50){
        	y_data.push_back(exp(ar * x * x + br * x) + rng.gaussian(w_sigma));
        } else {
        	y_data.push_back(exp(cr * x * x + dr * x) + rng.gaussian(w_sigma));
        }

    }
    if (ADD_OUTLIER){
        double x = x_data[10];
        y_data[10] *= 3.0;
    }
    

    // for (int i = 0; i < x_data.size(); ++i)
    // {
    //     std::cout << x_data[i] << " : ";
    //     std::cout << y_data[i] << std::endl;
    // }
    // return 0;

    CustomSparseFunctor<double, int> lmfunctor (n, m);
    lmfunctor.x_data = x_data;
    lmfunctor.y_data = y_data;


    Eigen::LevenbergMarquardt<CustomSparseFunctor<double, int>> lm (lmfunctor);
    lm.minimize(x_vec);




    cout << "estimated abcd = " << x_vec << endl;


    // std::cout << "X:" << std::endl;
    // for (auto x : x_data) std::cout << x << std::endl;

    // std::cout << "Y:" << std::endl;
    // for (auto y : y_data) std::cout << y << std::endl;



    return 0;
}