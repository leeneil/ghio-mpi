#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "fftw3.h"
#include <time.h>
using namespace std;

double* shift_and_invert(double* shift_model,double* shift_template,int n1,int n2,int step,int support1,int support2)
{
    // shift_model is the one to be shifted
    // shift_template is the standard to measure which shift is the best
    // basis is a medium used to shift
    int N = n1*n2;
	int i,j;
	int a,b;
    
	double **basis=(double **)malloc(sizeof(double*) *n1);
    for (i=0;i<n1;i++){
    	basis[i]=(double *)malloc(sizeof(double) *n2);
    }

	// record is used to record the error of each shift relative to shift_template
    double **record=(double **)malloc(sizeof(double*) *(2*step+1));
    for (i=0;i<2*step+1;i++)
        record[i]=(double *)malloc(sizeof(double) *(2*step+1));        
    
	// calculate the norm of shift_template inside the support 
    double template_norm = 0;
    for (i=support1;i<support2+1;i++){
    	for (j=support1;j<support2+1;j++)
    	    template_norm += pow(shift_template[j+i*n1],2);
    } 
    template_norm = sqrt(template_norm);
    
	double diff=0;
    // start to shift
    for (a=-step;a<step+1;a++) {
        for (b=-step;b<step+1;b++){
        	// diff indicate the norm of the difference between shifted_model and shift_template inside the support                               
            diff = 0; 
            for (i=support1;i<support2+1;i++){
            	for (j=support1;j<support2+1;j++)
                	basis [i+a][j+b] = shift_model[j+i*n1];
            }
            // record the error of shifted model relative to shift_template
            for (i=support1;i<support2+1;i++){
            	for (j=support1;j<support2+1;j++)
                	diff += pow(basis[i][j] - shift_template[j+i*n1],2);
            }
            record[a+step][b+step] =  sqrt(diff)/template_norm;                                   
        }
    }
    // find the value and the indices of minimum error
    
	double min1 = 1;
    int a1=0, b1=0;
    for (a=-step;a<step+1;a++){
    	for (int b=-step;b<step+1;b++){
        	if (record[a+step][b+step] < min1){
            	min1 = record[a+step][b+step];
                a1=a;
                b1=b;
            }
        }
    }
    // create FFT and iFFT plans ("out" is in real space and "in" is in k-space.)
    // This is used to invert the shift_model
    fftw_complex *in,*out;
    fftw_plan forward_p,inverse_p;
    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    forward_p = fftw_plan_dft_2d(n1, n2, out, in, FFTW_FORWARD, FFTW_MEASURE);
    inverse_p = fftw_plan_dft_2d(n1, n2, in, out, FFTW_BACKWARD, FFTW_MEASURE);
   
    for (i=0;i<n1;i++){
    	for (j=0;j<n2;j++){
        	out[j+i*n1][0] = shift_model[j+i*n1];
            out[j+i*n1][1] = 0;
        }
    }
    fftw_execute(forward_p);
    // make "in" to be its complex conjugate s.t. out can be inverted 
    for (i=0;i<n1;i++){
    	for (j=0;j<n2;j++)
        	in[j+i*n1][1] = -in[j+i*n1][1];
    }
    fftw_execute(inverse_p);

	double *out1 = (double *)malloc(sizeof(double) *N);
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++)
			out1[j+n1*i] = out[j+n1*i][0] / N;
	}
	FILE *fout;
    fout = fopen("invert.dat","wb");
    fwrite(out1,sizeof(double),n1*n2,fout);
    fclose(fout);
	free(out1);

    // normalze "out" and start to shift the inverted shift_model
    for (a=-step;a<step+1;a++){
    	for (b=-step;b<step+1;b++){
        	// diff indicate the norm of the difference between shifted_model and shift_template inside the support                               
            diff=0;
            for (i=support1;i<support2+1;i++){
            	for (j=support1;j<support2+1;j++)
                	basis [i+a][j+b] = out[j+i*n1][0]/N;
            }    
            // record the error of shifted model relative to shift_template
            for (i=support1;i<support2+1;i++){
            	for (j=support1;j<support2+1;j++)
                	diff += pow(basis[i][j]-shift_template[j+i*n1],2);
            } 
            record[a+step][b+step] =  sqrt(diff)/template_norm;                                    
        }
    }
    // find the value and the indices of minimum error
    double min2 = 1;
    int a2=0, b2=0;
    for (a=-step;a<step+1;a++){
    	for (b=-step;b<step+1;b++){
        	if (record[a+step][b+step] < min2){
            	min2 = record[a+step][b+step];
                a2=a;
                b2=b;
            }
        }
    }
    // compare the two mins  
    for (i=0;i<n1;i++){
    	for (j=0;j<n2;j++)
        	basis[i][j]=0;
    }    

    if (min1<min2){
    	for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++){
            	if (0<=i+a1 && i+a1<n1 && 0<=j+b1 && j+b1<n2)
                	basis [i+a1][j+b1] = shift_model[j+i*n1];
            }
        }
    }
    else{
    	for (i=0;i<n1;i++){
        	for (j=0;j<n2;j++){
            	if (0<=i+a2 && i+a2<n1 && 0<=j+b2 && j+b2<n2)
                	basis [i+a2][j+b2] = out[j+i*n1][0]/N;
            }
        }
    }
    // end of the shift and invert
    fftw_destroy_plan(forward_p);
    fftw_destroy_plan(inverse_p);
    free (record);

    double *outcome=(double *)malloc(sizeof(double) *N);
    for (i=0;i<n1;i++){
    	for (j=0;j<n2;j++)
        	outcome[j+i*n1] = basis[i][j];
    }
    free(basis);     
    return outcome;    
}
