/*  Computes the regression coefficients for ridge logistic regression.
 *  Algorithm by Genkin Et Al, 2007.
 *
 * function beta=clg_rr(X,y,lambda)
 *
 * Parameters:
 *   X       = regressor matrix [n x d]
 *   y       = targets, y_i = {-1, 1} [n x 1]
 *   lambda  = ridge penalty  (lambda * sum(x.^2) [L x 1]
 *
 * Returns:
 *   beta   = shrunken regression parameters [d x 1]
 *   loglik = log-likelihood for beta, [L x 1] 
 *
 * References:
 *
 * Large-Scale Bayesian Logistic Regression for Text Categorization
 * Alexander Genkin, David D Lewis, David Madigan. 
 * Technometrics, 49(3): 291-304, 2007.
 *
 * (c) Copyright Enes Makalic, 2010
 */
#include "math.h"
#include "mex.h"


// prototype
double min(const double &a, const double &b);
double max(const double &a, const double &b);
void rr(double* X, double* y, double *beta, double *lambda, mwSize n, mwSize d, mwSize L, double *r, double *loglik, double maxiter);
double update(double *X, double *y, double beta, double lambda, double *r, double *delta, int j, mwSize n);
bool have_converged(double *, double*, const double, mwSize);


// Call with: [b,loglik]=clg_lasso(X,y,lambda);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check that the function is called properly
    if (nrhs != 4) {
        mexErrMsgTxt("Four input arguments required.");
    } 
    if (nlhs > 3) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    // Check matrix dimensions; X~2D matrix and Y~1D matrix
    int numdims_X = mxGetNumberOfDimensions(prhs[0]);
    int numdims_Y = mxGetNumberOfDimensions(prhs[1]);
    int numdims_L = mxGetNumberOfDimensions(prhs[2]);
    int numdims_M = mxGetNumberOfDimensions(prhs[3]);
    
    const mwSize *dims_X = mxGetDimensions(prhs[0]);
    const mwSize *dims_Y = mxGetDimensions(prhs[1]);
    const mwSize *dims_L = mxGetDimensions(prhs[2]);
    const mwSize *dims_M = mxGetDimensions(prhs[3]);
        
    if(numdims_X != 2) {
        mexErrMsgTxt("First argument must be a 2D matrix.");
    }
    if(numdims_Y != 2) {
        mexErrMsgTxt("Second argument must be a column vector.");
    }
    if(numdims_L != 2) {
        mexErrMsgTxt("Third argument must be a column vector.");
    }    
    if(dims_Y[1] != 1){
        mexErrMsgTxt("Second argument must be a column vector.");
    }
    if(dims_L[1] != 1){
        mexErrMsgTxt("Third argument must be a column vector.");
    }
    if(dims_X[0] != dims_Y[0]) {
        mexErrMsgTxt("Data have incompatible dimensions.");
    }   

    // get the data sizes
    mwSize n = dims_X[0];   // number of samples
    mwSize d = dims_X[1];   // number of regressors
    mwSize L = dims_L[0];   // number of lambdas
    double maxiter = *mxGetPr(prhs[3]);
    
    // create output vector; initialized to beta(j)=9, j=1,...,d
    plhs[0] = mxCreateDoubleMatrix(d, L, mxREAL);
    double *beta=mxGetPr(plhs[0]);
    
    // compute log-likelihood if requested
    double *r=NULL;
    double *loglik=NULL;
    if(nlhs >= 2) {
        plhs[1] = mxCreateDoubleMatrix(L,1,mxREAL);
        loglik=mxGetPr(plhs[1]);

        plhs[2] = mxCreateDoubleMatrix(n,1,mxREAL);
        r=mxGetPr(plhs[2]);
    }    

    // call LASSO
    double *X=mxGetPr(prhs[0]);
    double *y=mxGetPr(prhs[1]);
    double *lambda=mxGetPr(prhs[2]);

    rr(X,y,beta,lambda,n,d,L,r,loglik,maxiter);
    
    return;
}

void rr(double* X, double* y, double *beta, double *lambda, mwSize n, mwSize d, mwSize L, double *r, double *loglik, double maxiter)
{
    const double epsilon=1e-5;      // stopping condition

    // allocate memmory
    double *delta=new double[d];    // change in beta[j]
    double *delta_r=new double[n]; 
    double *sum_delta_r=new double[n];
    
    // initialise arrays
    for(int i=0; i<n; i++)
    {
        r[i]=0.0;                   // r=zeroes(n,1);
        delta_r[i]=0.0;             // delta_r=zeroes(n,1);
    }
    for(int j=0; j<d; j++)
    {
        delta[j]=1.0;               // delta=ones(d,1);
    }
    
    // MAIN ALGORITHM
    for(int k=0; k<L; k++)
    {
        // Find solution for a single lambda[k]
        unsigned int iter=0;
        bool done = false;
        
        while(!done)
        {
            iter++; // next iteration
            for(int i=0; i<n; i++)
            {
                sum_delta_r[i]=0.0; // sum_delta_r=zeroes(n,1);
            }            
            
            // loop over all predictors
            for(int j=0; j<d; j++)
            {
                double delta_beta_j=0.0; // change to beta[j]
                double delta_v_j=0.0;     

                if(j==0) {  // hack for constant reg. where we dont want shrinkage
                    delta_v_j=update(X,y,beta[d*k+j],0,r,delta,j,n);
                } else {
                    // compute update step
                    delta_v_j=update(X,y,beta[d*k+j],lambda[k],r,delta,j,n);
                }

                // trust region size
                delta_beta_j = min(max(delta_v_j, -delta[j]), delta[j]);

                // update predictor inner product            
                // delta_r = delta_beta_j * X(:,j).* y; r = r + delta_r;
                for(int i=0; i<n; i++)                
                {
                    delta_r[i] = delta_beta_j * X[j*n+i] * y[i];                
                    r[i] += delta_r[i];
                    sum_delta_r[i] += delta_r[i];
                }

                // update regressor coeff
                beta[d*k+j] += delta_beta_j; 

                // update size of trust region
                delta[j] = max(2*abs(delta_beta_j), delta[j]/2);   
            }

            done = have_converged(r,sum_delta_r,epsilon,n);
            if(iter >= maxiter) {
//                mexPrintf("Maximum number of iterations reached.\n");
                done=true;
            }            
        }
        
        // compute loglikelihood if requested
        // loglik(k) = -sum(log(1.0 + exp(-r)));
        if(loglik != NULL) {
            for(int i=0; i<n; i++)
            {
                loglik[k] -= log(1.0 + exp(-r[i]));
            }
        }        
        
        // if not last lambda, initialize next beta's to previous estimates
        if(k != (L-1))
        {
            for(int j=0; j<d; j++)
                beta[d*(k+1)+j] = beta[d*k+j];
        }
    }
    
    // clear memmory
    delete[] delta;
    delete[] delta_r;
    delete[] sum_delta_r;
}

double update(double *X, double *y, double beta, double lambda, double *r, double *delta, int j, mwSize n)
{
    double delta_v_j=0.0;
    
    // delta_v_j = sum(Xy(:,j) ./ (1 + exp(r))) - lambda*s;
    for(int i=0; i<n; i++)
    {
        delta_v_j -=  (X[j*n+i] * y[i]) / (1.0 + exp(r[i]));
    }
    delta_v_j += 2*lambda*beta;
    
    // compute F
    double F=0.0;
    double denum=0.0;
    for(int i=0; i<n; i++)
    {
        double a=delta[j]*abs(X[j*n+i]);     // a=delta(j)*X(:,j); 
        
        if(abs(r[i]) <= a)
        {
            F=0.25;
        }
        else
        {
            F=1.0 / (2.0 + exp(abs(r[i]) - a) + exp(a - abs(r[i])));
        }
        denum += X[j*n+i]*X[j*n+i]*F;
    }
    
    delta_v_j = -(delta_v_j / (denum + 2.0*lambda));
    
    return delta_v_j;
}

bool have_converged(double *r, double *delta_r, const double epsilon, int n)
{
    double num=0.0, denum=0.0;
    for(int i=0; i<n; i++)
    {
        num += abs(delta_r[i]);
        denum += abs(r[i]);
    }
    
    return (num/(1.0 + denum)) <= epsilon;
}

double min(const double &a, const double &b)
{
    return (a < b) ? a : b;
}

double max(const double &a, const double &b)
{
    return (a > b) ? a : b;
}
