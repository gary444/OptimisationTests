#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;         // real parameter value
    double ae = 2.0, be = -1.0, ce = 5.0;        // Estimated parameter value
    int N = 100;                                 // data point
    double w_sigma = 1.0;                        // Noise Sigma value
    cv::RNG rng;                                 // OpenCV random number generator

    vector<double> x_data, y_data;      // data
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma));
    }

    // Start Gauss-Newton iteration
    int iterations = 100;    // Number of iterations
    double cost = 0, lastCost = 0;  // The cost of this iteration and the cost of the previous iteration

    for (int iter = 0; iter < iterations; iter++) {
        Matrix3d H = Matrix3d::Zero();             // Hessian = J^T J in Gauss-Newton
        Vector3d b = Vector3d::Zero();             // bias
        cost = 0;
        for (int i = 0; i < N; i++) {
            double xi = x_data[i], yi = y_data[i];  // i-th data point
            double error = 0;   // The calculation error of the i-th data point
            error = yi-exp(ae*xi*xi+be*xi+ce);
            Vector3d J;

            // see explanation of differentiating exponential functions with chain rule:
            //https://www.khanacademy.org/math/in-in-grade-12-ncert/xd340c21e718214c5:continuity-differentiability/xd340c21e718214c5:exponential-functions-differentiation/a/differentiating-exponential-functions-review
            // practice set 2 explanation
            // differentiation of exp(u(a)) is exp(u(a))*(du/da)
            J[0] = -xi*xi*exp(ae*xi*xi+be*xi+ce);
            J[1] = -xi*exp(ae*xi*xi+be*xi+ce);
            J[2] = -exp(ae*xi*xi+be*xi+ce);


            // this shows that JTJ matrix can be calculated as the sum of JTJ matrices formed from individual terms
            // JTJ probably still to large to calculate for every term in ED graph case, so a more localised approach is needed. 
            //How does this work when multiple blocks of parameters affect a term
            H += J * J.transpose(); // GN approximate H

            // this is JTf(x)
            // note the minus sign
            b += -error * J;

            // total cost f(x)^T.f(x)
            cost += error * error;
        }
        // Solve the linear equation Hx=b, it is recommended to use ldlt
        Vector3d dx;
        //dx=H.inverse()*b; //Direct inverse method to solve the increment
        dx = H.ldlt().solve(b); // ldlt method
        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost > lastCost) {
            // The error has increased, indicating that the approximation is not good enough
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update abc estimate
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        cout << "total cost: " << cost << endl;
    }

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}