#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>


#include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace Eigen;


struct LMFunctor
{   
    // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
    int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
    {
        // find the residuals / evaluate each term / calculate squared difference at each data point
        // with given parameters in x
        // store in fvec

        // function is ax^2 + bx + c

        for (int m_i = 0; m_i < values(); ++m_i) {

            float a = x[0];
            float b = x[1];
            float c = x[2];
            float x_val = x_data[m_i];
            fvec[m_i] = y_data[m_i] - exp(a*x_val*x_val+b*x_val+c);

        }

        return 0;

    }

    // Compute the jacobian of the errors
    int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const
    {
        // find partial derivatives of each term / data point WRT each parameter in x
        for (int m_i = 0; m_i < values(); ++m_i)
        {

            float a = x[0];
            float b = x[1];
            float c = x[2];
            float x_val = x_data[m_i];

            // analytical version
            fjac(m_i, 0) = -x_val*x_val*exp(a*x_val*x_val+b*x_val+c);
            fjac(m_i, 1) = -x_val*exp(a*x_val*x_val+b*x_val+c);
            fjac(m_i, 2) = -exp(a*x_val*x_val+b*x_val+c);

        }
        return 0;
    }

    // Number of data points, i.e. values.
    int m;

    // Returns 'm', the number of values.
    int values() const { return m; }

    // The number of parameters, i.e. inputs.
    int n;

    // Returns 'n', the number of inputs.
    int inputs() const { return n; }

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
    // Eigen::VectorXf fVec (m); // term evaluations / error for each data point
    // Eigen::MatrixXf fJac (m, n); // term evaluations / error for each data point


    vector<double> x_data, y_data;      // data
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma));
    }

    LMFunctor lmfunctor;
    lmfunctor.m = m;
    lmfunctor.n = n;
    lmfunctor.x_data = x_data;
    lmfunctor.y_data = y_data;

    Eigen::LevenbergMarquardt<LMFunctor, float> lm (lmfunctor);

    lm.minimize(x);

    cout << "estimated abc = " << x << endl;


    return 0;
}