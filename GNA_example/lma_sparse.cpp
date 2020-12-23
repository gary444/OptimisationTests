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
	// int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
	int operator()(const InputType &x, ValueType& fvec)
	{
	    // find the residuals / evaluate each term / calculate squared difference at each data point
	    // with given parameters in x
	    // store in fvec

	    // function is ax^2 + bx + c

	    for (int m_i = 0; m_i < this->values(); ++m_i) {

	        float a = x[0];
	        float b = x[1];
	        float c = x[2];
	        float x_val = x_data[m_i];
	        fvec[m_i] = y_data[m_i] - exp(a*x_val*x_val+b*x_val+c);

	    }

	    return 0;

	}


    // Compute the jacobian of the errors
    // int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const
	int df(const InputType &x, JacobianType& fjac)
    {
        // find partial derivatives of each term / data point WRT each parameter in x
        for (int m_i = 0; m_i < this->values(); ++m_i)
        {

            float a = x[0];
            float b = x[1];
            float c = x[2];
            float x_val = x_data[m_i];

            // analytical version
            fjac.coeffRef(m_i, 0) = -x_val*x_val*exp(a*x_val*x_val+b*x_val+c);
            fjac.coeffRef(m_i, 1) = -x_val*exp(a*x_val*x_val+b*x_val+c);
            fjac.coeffRef(m_i, 2) = -exp(a*x_val*x_val+b*x_val+c);

        }
    	fjac.makeCompressed();
        return 0;


    }

    vector<double> x_data;
    vector<double> y_data;


};


int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;         // real parameter value
    double ae = 2.0, be = -1.0, ce = 5.0;        // Estimated parameter value
    int N = 100;                                 // data point
    double w_sigma = 1.0;                        // Noise Sigma value
    cv::RNG rng;                                 // OpenCV random number generator


    // for LMA
    int n = 3;//num parameters
    int m = N;//num constraints (terms)
    Eigen::VectorXf x (3); // parameter values (initial)    
    x(0) = ae;
    x(1) = be;
    x(2) = ce; // parameter values (initial)    


    vector<double> x_data, y_data;      // data
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma));
    }



    CustomSparseFunctor<float, int> lmfunctor (n, m);
    lmfunctor.x_data = x_data;
    lmfunctor.y_data = y_data;


    Eigen::LevenbergMarquardt<CustomSparseFunctor<float, int>> lm (lmfunctor);
    lm.minimize(x);


    cout << "estimated abc = " << x << endl;


    return 0;
}