
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>


#include <vector>


#include <cusparse_cholesky_solver.h>


#define USE_CUDA_SOLVER 1


using Scalar = double;
// using SparseMatrixCSC = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::ColMajor>;
using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::RowMajor>;
// using Triplet = Eigen::Triplet<Scalar>;
using VectorR = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;


// checks diagonal elements of a matrix and fins the lowest
double get_max_diagonal_coeff(const Eigen::SparseMatrix<double>& sp_mat) 
{
	double max_coeff = DBL_MIN;
	assert (sp_mat.rows() == sp_mat.cols());
	for (int k=0; k<sp_mat.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(sp_mat, k); it; ++it) {
			if (it.row() == it.col() && max_coeff < it.value())
				max_coeff = it.value();
		}
	}
	return max_coeff;
}




typedef int Index;


class Deformer
{
public:
	Deformer(const uint32_t _inputs_n, const uint32_t _values_m, 
		const std::vector<double>& _x_data,
		const std::vector<double>& _y_data)
	 : inputs_n(_inputs_n), 
	   values_m(_values_m),
	   x_data(_x_data), 
	   y_data(_y_data)
	   {}
	// ~Deformer() {}
	
	uint32_t values(){
		return values_m;
	}
	uint32_t inputs(){
		return inputs_n;
	}


	   // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd& fvec)
	{
	    // find the residuals / evaluate each term / calculate squared difference at each data point
	    // with given parameters in x
	    // store in fvec

        double a = x[0];
        double b = x[1];
        double c = x[2];
        double d = x[3];

	    for (Index m_i = 0; m_i < Index(this->values()); ++m_i) {

	        double x_val = x_data[m_i];
	        double y_val = y_data[m_i];
	        
	        if (m_i < Index(50)){

	        	fvec[m_i] = y_val - exp(a*x_val*x_val+b*x_val);

	        } else {
	        	fvec[m_i] = y_val - exp(c*x_val*x_val+d*x_val);
	        }

	    }

        // std::cout << "Fvec\n" << fvec << std::endl;

	    return 0;
	}


    // Compute the jacobian of the errors
	int df(const Eigen::VectorXd &x, Eigen::SparseMatrix<double>& fjac)
    {
        // find partial derivatives of each term / data point WRT each parameter in x
        double a = x[0];
        double b = x[1];
        double c = x[2];
        double d = x[3];

        const float epsilon = 1e-5f;

        for (Index m_i = 0; m_i < Index(this->values()); ++m_i)
        {
            double x_val = x_data[m_i];
	        double y_val = y_data[m_i];

            double pds[2];
	       
            if (m_i < 50) {
	            pds[0] = -x_val*x_val*exp(a*x_val*x_val+b*x_val);
	            pds[1] = -x_val*exp(a*x_val*x_val+b*x_val);


                fjac.coeffRef(m_i, 0) = pds[0];
                fjac.coeffRef(m_i, 1) = pds[1];

            } else {

	            pds[0] = -x_val*x_val*exp(c*x_val*x_val+d*x_val);
	            pds[1] = -x_val*exp(c*x_val*x_val+d*x_val);


                fjac.coeffRef(m_i, 2) = pds[0];
                fjac.coeffRef(m_i, 3) = pds[1];
	        }
        }

        fjac.makeCompressed();
        return 0;
    }

	const uint32_t values_m;
	const uint32_t inputs_n;
	const std::vector<double> x_data;
	const std::vector<double> y_data;

	const uint32_t m_max_entries_per_col = 40;
};


double minimize_levenberg_marquardt(Deformer& deformer, 
		Eigen::VectorXd& X
		) {


    // energy_minimised = false;

	const double eps[4] = {1e-15, 1e-4, 1e-11, 0};
    const double lambda_init = 1e-2;    //if init guess is a good approximation, lambda_init should be small, else lambda_init = 10^-3 or even 1
    const int ITER_MAX = 7;

    const int DoF = deformer.values() - deformer.inputs() + 1; // statistical degrees of freedom

    //create f vector
	Eigen::VectorXd fVec(deformer.values()); 
	fVec.setZero();
	
	// create jacobian
    std::cout << "fJac size: " << deformer.values() << " x " << deformer.inputs() << std::endl; 
	Eigen::SparseMatrix<double> fJac(deformer.values(), deformer.inputs());
	fJac.reserve(Eigen::VectorXi::Constant(deformer.inputs(),  deformer.m_max_entries_per_col));
	
	// create identity matrix
	Eigen::SparseMatrix<double> _Identity(deformer.inputs(), deformer.inputs());
	_Identity.reserve(Eigen::VectorXi::Constant(deformer.inputs(), 1));
	for (int k=0; k<deformer.inputs(); ++k)  _Identity.coeffRef(k, k) = 1.0;

	// fill fVec and jacobian
    deformer(X, fVec);
    deformer.df(X, fJac);

    // get JTJ and JTf
	Eigen::SparseMatrix<double> A = fJac.transpose();
	Eigen::VectorXd             g = A*fVec;
	A = A*fJac;

	// calculate initial energy
    double best_energy = sqrt(fVec.dot(fVec)); 
	double cur_energy = best_energy;

    std::cout << "LM: initial energy = " << cur_energy << std::endl;

	// this is method 3 in Gavin's tutorial
	// damping parameter is normalised according to Marquardt's normalisation theorem
    double mu = lambda_init * get_max_diagonal_coeff(A);

    double nu = 2.0; // factor for scaling up mu if step not accepted

    int iter = 0; 
    int best_iter = -1;
    const double max_coeff = g.maxCoeff();
    bool found = max_coeff < eps[0]; // check for gradient convergence before we start
	if (found) std::cout << "Converged: gradient convergence ( max coeff: " << max_coeff << ")" << std::endl;


	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;

	// main optimisation loop
    while (!found && iter++ < ITER_MAX) {

		//TODO: after uncomment this line, the min energy is bigger than comment condition, however, this condition can speedup
		//each factorization time.

		A = A + _Identity*mu;

        std::cout << "start factorisation..." << std::endl; 

		Eigen::VectorXd h ( deformer.inputs() );

        if (USE_CUDA_SOLVER) {

        	// estimate x with cuSparse
			SparseMatrixCSR Acsr = A; // solver supports CSR format
        	const int nnz = Acsr.nonZeros();
			auto solver = CuSparseCholeskySolver<Scalar>::create( deformer.inputs() ); // matrix must be nxn for cholesky solvers
			solver->analyze(nnz, Acsr.outerIndexPtr(), Acsr.innerIndexPtr());

			// VectorR xhatGPU( deformer.inputs() );
			solver->factorize(Acsr.valuePtr());

	        std::cout << "solving..." << std::endl; 
			Eigen::VectorXd b = -g;
			solver->solve(b.data(), h.data());

        } else {

	        solver.compute(A);
	        
	        // pre computation of pattern did not seem to work in this case (but did with GNA, ???)
	        // if (iter == 0) solver.analyzePattern(A);
	        // solver.factorize(A);

			if (solver.info() != Eigen::Success)
			{
				printf("!!!Cholesky factorization on Jac failed. solver.info() = %d\n", solver.info());
				break;
			}

	        std::cout << "solving..." << std::endl; 

			// calculate parameter update
			h = solver.solve(-g);

        }






		// check for convergence in parameters
		// this differs slightly from Gavin's
		if (sqrt(h.dot(h)) <= eps[1]* (sqrt(X.dot(X))+eps[1])) {
			found = true;
			std::cout << "Converged: parameter convergence" << std::endl;
			break;
		}

        std::cout << "Update objective function" << std::endl;

		// calculate updated X and evaluate objective function
		const Eigen::VectorXd _vX = X + h;
        deformer(_vX, fVec);
		cur_energy = sqrt(fVec.dot(fVec)); 

		std::cout << "best_energy = " << best_energy << "(iter " << best_iter << ")" << "\tcur_energy : " << cur_energy << std::endl;


		// step evaluation: calculate rho
        const double rho = (best_energy - cur_energy) / (h.dot(mu*h-g)); // JTf, here expressed as g, is negated when compared with Gavins tutorial

        if (rho > 0) {   //accept step
            best_energy = cur_energy;
			X = _vX;
            deformer.df(X, fJac);
			A = (fJac).transpose();
            g = A*fVec;
            A = A*fJac;

		    const double max_coeff = g.maxCoeff();
		    found = max_coeff < eps[0]; // check for gradient convergence before we start
			if (found) std::cout << "Converged: gradient convergence ( max coeff: " << max_coeff << ")" << std::endl;

            // update damping parameter
            mu = mu*std::max(1.0/3, 1-(2*rho-1)*(2*rho-1)*(2*rho-1));
            nu = 2.0;

    		// energy_minimised = true;
    		best_iter = iter;

    		// check for convergence in objective function
    		// reduced energy = energy / (m−n+ 1)
    		const double red_energy = best_energy / DoF;
    		if (red_energy < eps[2]){
    			found = true;
    			std::cout << "Converged: objective function (" << red_energy << ")" << std::endl;
    		} 



        } else {
            mu = mu*nu; 
            nu = 2*nu;
        }
    }	
	
	std::cout << "iter = " << iter << "\tbest_energy = " << best_energy << "(iter " << best_iter << ")" << std::endl;
	return best_energy;
}

double func1 (const double a, const double b, const double x){
	return exp(a * x * x + b * x);
}

int main(int argc, char **argv) {

	double ar = 1.2, br = 2.2, cr = 2.3, dr = 1.1;         // real parameter value
    double ae = 1.5, be = 2.0, ce = 2.0, de = 2.0;        // Estimated parameter value
    int m = 100;                                         // data points
    double w_sigma = 0.0001;                             // Noise Sigma value
    cv::RNG rng;                                         // OpenCV random number generator

    std::cout << "Actual parameters: " << std::endl;
    std::cout << "A: " << ar << std::endl;
    std::cout << "b: " << br << std::endl;
    std::cout << "c: " << cr << std::endl;
    std::cout << "d: " << dr << std::endl;

    // for LMA
    int n = 4;//num parameters
    Eigen::VectorXd x_vec (4); // parameter values (initial)    

    x_vec(0) = ae;
    x_vec(1) = be;
    x_vec(2) = ce;
    x_vec(3) = de;


    std::vector<double> x_data, y_data;      // data
    for (int i = 0; i < m; i++) {

        double x = rng.uniform((double)0, (double)0.5);

        x_data.push_back(x);

        if (i < (m/2) ){
        	y_data.push_back(func1(ar, br, x) + rng.gaussian(w_sigma));
        } else {
        	y_data.push_back(func1(cr, dr, x) + rng.gaussian(w_sigma));
        }
    }

	Deformer deformer(n, m, x_data, y_data);



	double result = minimize_levenberg_marquardt(deformer, x_vec);


    std::cout << "estimated abcd = " << x_vec << std::endl;

	return 0;
}