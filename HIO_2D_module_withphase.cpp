#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <fftw3.h>
#include <time.h>

//---Define parameters---
#define factor1 0.9
#define factor2 0.7

using namespace std;

double **HIO_2D_withphase(double **HIOinput, int n1, int n2, int support1, int support2, int iteration, double *shift)
{   
    // The input intensity should not be fft-shifted.
    // The input matrix should be odd.

    int i=0,j=0,iter=0;
    
	// checker is to check if the intensity==0, for example, the beamstop
    // 0 stands for false; 1 stands for true
    bool **checker=(bool **)malloc(sizeof(bool*) *n1);
    for (i=0;i<n1;i++)
        checker[i]=(bool *)malloc(sizeof(bool) *n2); 
    
    double epsilon = pow(10,-10);
    for (i=0;i<n1;i++){
    	for (j=0;j<n2;j++){
        	if (HIOinput[i][j]<epsilon){
            	checker[i][j] = false;
                HIOinput[i][j] = 1;
            }
            else
            	checker[i][j] = true;
        }
    }
    
	// create FFT and iFFT plans ("out" is in real space and "in" is in k-space.)
    fftw_complex *in,*out;
    fftw_plan forward_p,inverse_p;
    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n1*n2);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n1*n2);
    forward_p = fftw_plan_dft_2d(n1, n2, out, in,FFTW_FORWARD, FFTW_MEASURE);
    inverse_p = fftw_plan_dft_2d(n1, n2, in, out,FFTW_BACKWARD, FFTW_MEASURE);
    
	// shift offers input phases
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++){
			out[j+n1*i][0] = shift[j+n1*i];
			out[j+n1*i][1] = 0;
		}
	}
	fftw_execute(forward_p);

	// Normalze "in" and assign the intensity from HIOinput to "in" and use the input phase.
    double norm=0;
    for (i=0;i<n1;i++){
    	for (j=0;j<n2;j++){
			if (checker[i][j] == true){
				norm = sqrt(pow(in[j+n1*i][0],2)+pow(in[j+n1*i][1],2));
        		in[j+n1*i][0] = HIOinput[i][j]*in[j+n1*i][0]/norm;
            	in[j+n1*i][1] = HIOinput[i][j]*in[j+n1*i][1]/norm;
			}
			else{
				in[j+n1*i][0] = in[j+n1*i][0];
                in[j+n1*i][1] = in[j+n1*i][1];
			}
        }
    }     
    // Now "in" becomes our first trial solution.
    
    // Construct a matrix in order to store previous values of "out".
    double **previous=(double **)malloc(sizeof(double*) *n1);
    for (i=0;i<n1;i++)
    	previous[i]=(double *)malloc(sizeof(double) *n2);

	int N = n1*n2;
    fftw_execute(inverse_p);
    
	for (i=0;i<n1;i++){
    	for (j=0;j<n2;j++){
        	out[j+n1*i][0] = out[j+n1*i][0]/N;
            previous [i][j] = out[j+n1*i][0];
        }
    }        
    
	// Iteration starts here.
    for (iter=0;iter<iteration/2;iter++){
    	fftw_execute(inverse_p);

        // Since fftw only gives us unnormalized solution, we have to normalize "out" ourselves.
        for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++){
            	out [j+n1*i][0] = out[j+n1*i][0]/N;
                out [j+n1*i][1] = 0;
            }
        } 
        
		// Modify the real space data. (HIO)
        for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++){         
            	// the center square
                if ( support1<=i && i<=support2 && support1<=j && j<=support2 ){       
                	if (out[j+n1*i][0]<0)        
                    	out[j+n1*i][0] = previous[i][j]-factor1*out[j+n1*i][0];
                    else
                        out[j+n1*i][0] = out[j+n1*i][0];
                }
				// the upper left triangle                     
                else if (i<j){
                	if (out[j+n1*i][0]==0)
                    	out[j+n1*i][0] = out[j+n1*i][0]; 
                    else
                        out[j+n1*i][0] = previous[i][j]-factor1*out[j+n1*i][0];
                }
                // the lower right triangle
                else{
                	if (out[j+n1*i][0]==0)
                    	out[j+n1*i][0] = out[j+n1*i][0]; 
                    else
                        out[j+n1*i][0] = previous[i][j]-factor2*out[j+n1*i][0];
                }
            }
        }
            
        for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++)
            	previous [i][j] = out[j+n1*i][0];
        } 

        // transfrom back to k-space and deal with the beamstop area.
        fftw_execute(forward_p);
            
		for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++){                 
            	norm = sqrt( pow(in[j+n1*i][0],2) + pow(in[j+n1*i][1],2) );
                if (norm<epsilon){
                	cout<<"Warning! There might be a zero-division error."<<endl;
					exit(0);
                }
                else{
                    // For the beamstop area, keep the value
                    if (checker[i][j] == false){
                    	in[j+n1*i][0] = in[j+n1*i][0];
                        in[j+n1*i][1] = in[j+n1*i][1];                  
                    }
                    // For other area, combine the intensity from HIOinput and phase from "in".
                    else{
                    	in[j+n1*i][0] = HIOinput[i][j]*(in[j+n1*i][0])/norm;
                        in[j+n1*i][1] = HIOinput[i][j]*(in[j+n1*i][1])/norm;  
                    }  
                }      
            }
        }
    }
    
    for (iter=iteration/2;iter<iteration;iter++){
    	fftw_execute(inverse_p);

        // Since fftw only gives us unnormalized solution, we have to normalize "out" ourselves.
        for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++){
            	out [j+n1*i][0] = out[j+n1*i][0]/N;
                out [j+n1*i][1] = 0;
            }
        } 
        
		// Modify the real space data. (HIO)
        for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++){         
            	// the center square
                if ( support1<=i && i<=support2 && support1<=j && j<=support2 ){       
                	if (out[j+n1*i][0]<0)        
                    	out[j+n1*i][0] = out[j+n1*i][0]*(iteration - iter)/(iteration/2);
                    else
                        out[j+n1*i][0] = out[j+n1*i][0];
                }
				// the upper left triangle                     
                else if (i<j){
                	if (out[j+n1*i][0]==0)
                    	out[j+n1*i][0] = out[j+n1*i][0]; 
                    else
                        out[j+n1*i][0] = out[j+n1*i][0]*(iteration - iter)/(iteration/2);
                }
                // the lower right triangle
                else{
                	if (out[j+n1*i][0]==0)
                    	out[j+n1*i][0] = out[j+n1*i][0]; 
                    else
                        out[j+n1*i][0] = out[j+n1*i][0]*(iteration - iter)/(iteration/2);
                }
            }
        }
            
        /*for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++)
            	previous [i][j] = out[j+n1*i][0];
        }*/ 

        // transfrom back to k-space and deal with the beamstop area.
        fftw_execute(forward_p);
            
		for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++){                 
            	norm = sqrt( pow(in[j+n1*i][0],2) + pow(in[j+n1*i][1],2) );
                if (norm<epsilon){
                	cout<<"Warning! There might be a zero-division error."<<endl;
					exit(0);
                }
                else{
                    // For the beamstop area, keep the value
                    if (checker[i][j] == false){
                    	in[j+n1*i][0] = in[j+n1*i][0];
                        in[j+n1*i][1] = in[j+n1*i][1];                  
                    }
                    // For other area, combine the intensity from HIOinput and phase from "in".
                    else{
                    	in[j+n1*i][0] = HIOinput[i][j]*(in[j+n1*i][0])/norm;
                        in[j+n1*i][1] = HIOinput[i][j]*(in[j+n1*i][1])/norm;  
                    }  
                }      
            }
        }
    }

	// Iteration ends here.
    // Create an array to store output data and release memory
    fftw_execute(inverse_p);
	    
	double **HIOoutput=(double **)malloc(sizeof(double*) *n1);
	for (i=0;i<n1;i++)
		HIOoutput[i]=(double *)malloc(sizeof(double) *n2);

	for (i=0;i<n1;i++){
    	for (j=0;j<n2;j++)
        	HIOoutput[i][j]=out[j+n1*i][0]/N;
    }

    fftw_destroy_plan(forward_p);
    fftw_destroy_plan(inverse_p);
    
	free(previous);
	free(checker);
    free(in);
    free(out);
    return HIOoutput;
}
