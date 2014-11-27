#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <fftw3.h>
#include "HIO_2D_module.cpp"
#include "HIO_2D_module_withphase.cpp"
#include "shift_and_invert.cpp"
#include <mpi.h>  
#include <ctime> 

// #define n1 1221
// #define n2 1221
// #define support1 460
// #define support2 760
// #define iteration 2000
#define step 15              // (support2 - (upper bound of tight support)) * 2 + 1
// #define gen 10               // number of generations
// #define num_replica 48     // number of replicas should be integral times of nproc
// #define ID 6

using namespace std;
int main(int argc, char *argv[]){

int i,j;

std::cout<< "initializing... " << std::endl;

// Initialize MPI
int nproc, myid;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &nproc);
MPI_Comm_rank(MPI_COMM_WORLD, &myid);

// start to count
MPI_Barrier(MPI_COMM_WORLD);
double start_time = MPI_Wtime();


int N = 0;
int S = 0;
int iteration = 0;
int gen = 0;
int n1 = 0;
int n2 = 0;
int support1 = 0;
int support2 = 0;
int num_replica = 0;
// read in filename
char filename[128];
if (argc < 2) {
	cout << "No input file!" << endl;
	exit(0);
} else {
	sprintf(filename,"%s",argv[1]);
	if (argc < 7) {
		cout << "No input arguments specified! " << endl;
		cout << "Format: " << endl;
		cout << "ghio2d filename image_size "
			 << "support_size n_iters n_gens n_replica" << endl;
		exit(0);
	} else {
		N = atoi(argv[2]);
		S = atoi(argv[3]);
		iteration = atoi(argv[4]);
		gen = atoi(argv[5]); 
		num_replica = atoi(argv[6]);
	}
}
n1 = N;
n2 = N;
support1 = (floor(N/2.0)-1) - floor(S/2);
support2 = (floor(N/2.0)-1) + floor(S/2);

srand(time(NULL));
int ID = rand() % 100;

if (argc > 7)
	ID = atoi(argv[7]);

char dir[128];
sprintf(dir,"%s_%d",filename,ID);
mkdir(dir,0755);
char fn[128];

// read in intensity
double *intensity=(double *)malloc(sizeof(double) *n1*n2);
if (myid==0){
	FILE *fin;
	fin=fopen(filename,"rb");
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++)
			fread(&intensity[j+n1*i],1,sizeof(double),fin);
	}
	fclose(fin);
}

// broadcast intensity to other nodes
MPI_Bcast(intensity, n1*n2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

int num = num_replica / nproc;
	
double *errorF=(double *)malloc(sizeof(double) *num);
double norm = 0, tmp = 0;
double epsilon = pow(10,-10);

// create an FFTW plan
fftw_complex *in,*out;
fftw_plan forward_p,inverse_p;
in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n1*n2);
out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n1*n2);
forward_p = fftw_plan_dft_2d(n1, n2, out, in,FFTW_FORWARD, FFTW_MEASURE);
inverse_p = fftw_plan_dft_2d(n1, n2, in, out,FFTW_BACKWARD, FFTW_MEASURE);

double **image_tmp=(double **)malloc(sizeof(double*) *(n1*n2)*num);
for (i=0;i<n1*n2;i++)
	image_tmp[i]=(double *)malloc(sizeof(double) *num);

double **HIOinput=(double **)malloc(sizeof(double*) *n1*n2);
for (i=0;i<n1;i++)
	HIOinput[i]=(double *)malloc(sizeof(double) *n2);

double **M;
for (int t=0;t<num;t++){
	
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++)
			HIOinput[i][j] = intensity[j+n1*i];
	}
	
	// It should be careful that values of HIOinput would change after used by HIO_2D.
	M = HIO_2D(HIOinput, n1, n2, support1, support2, iteration, myid);
	
	// save M in image_tmp[t]
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++)
			image_tmp[j+n1*i][t] = M[i][j];
	}
	
	// calculate errorF
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++){
			out[j+n1*i][0]=M[i][j];
			out[j+n1*i][1]=0;
		}
	}
	fftw_execute(forward_p);
	norm = 0;
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++){
			if (intensity[j+n1*i]>epsilon){
				tmp = sqrt( pow(in[j+n1*i][0],2) + pow(in[j+n1*i][1],2) );
				norm += pow(intensity[j+n1*i] - tmp, 2);
			}
		}
	}
	errorF[t] = norm;
	//printf("errorF[%d] = %E, the %d-th node.\n", t, norm, myid);
}

struct {
	double value;
	int index;
} in_node, out_node;

in_node.value = errorF[0];
in_node.index = 0;
for (int t=0;t<num;t++){
	if (errorF[t]<in_node.value){
		in_node.value = errorF[t];
		in_node.index = t;
	}
}

// Now in_node.value is the min of errorF in each processor.
in_node.index = myid*num + in_node.index;

// sychronize every processor
MPI_Barrier(MPI_COMM_WORLD);

// determine the nodes with min errorF
MPI_Reduce(&in_node, &out_node, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
double minval=0;
int minrank=0, minindex=0;
if (myid==0){
	minval = out_node.value;
	minrank = out_node.index / num;
	minindex = out_node.index % num;
	printf("minval = %E, minrank = %d, minindex = %d\n", minval, minrank, minindex);
}

// Broadcast minrank and minindex to every node.
MPI_Bcast(&minrank, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&minindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);

// create shift_template and then broadcast it to every node
double *shift_template=(double *)malloc(sizeof(double) *n1*n2);
if (myid==minrank){
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++)
			shift_template[j+n1*i] = image_tmp[j+n1*i][minindex];
	}
}
MPI_Bcast(shift_template, n1*n2, MPI_DOUBLE, minrank, MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD);
if (myid==0){
	sprintf(fn, "%s/template_gen1.dat", dir);
	FILE *fout = fopen(fn, "w");
	fwrite(shift_template,sizeof(double),n1*n2,fout);
	fclose(fout);
	printf("This is the 1-th generation.\n");
}

double temp_max=0;
for (i=0;i<n1*n2;i++){
	if (temp_max < shift_template[i])
		temp_max = shift_template[i];
}

double *shift_model=(double *)malloc(sizeof(double) *n1*n2);
double *outcome;
double shift_max=0;

for (int t=0;t<num;t++){
	
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++)
			shift_model[j+n1*i] = image_tmp[j+n1*i][t];
	}
	
	for (i=0;i<n1*n2;i++){
		if (shift_max < shift_model[i])
			shift_max = shift_model[i];
	}

	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++)
			shift_model[j+n1*i] = shift_model[j+n1*i]*(temp_max/shift_max);
	}

	outcome = shift_and_invert(shift_model,shift_template,n1,n2,step,support1,support2);

	// Now image_tmp stores the images which have been shifted or inverted.
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++)
			image_tmp[j+n1*i][t] = outcome[j+n1*i]*(shift_max/temp_max);
	}
}

double *shift=(double *)malloc(sizeof(double) *n1*n2);

for (int q=1;q < gen;q++){
	
	for (int t=0;t<num;t++){

		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++)
				shift[j+n1*i] = image_tmp[j+n1*i][t];
		}

		// apply geometric average
		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++){
				if (shift[j+n1*i]>=0)
					shift[j+n1*i] = sqrt(fabs(shift[j+n1*i]*shift_template[j+n1*i]));
				else
					shift[j+n1*i] = -sqrt(fabs(shift[j+n1*i]*shift_template[j+n1*i]));
			}
		}
		
		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++)
				HIOinput[i][j] = intensity[j+n1*i];
		}

		M = HIO_2D_withphase(HIOinput, n1, n2, support1, support2, iteration, shift);

		// save M in image_tmp[t]
		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++)
				image_tmp[j+n1*i][t] = M[i][j];
		}

		// calculate errorF
		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++){
				out[j+n1*i][0]=M[i][j];
				out[j+n1*i][1]=0;
			}
		}
		fftw_execute(forward_p);
		norm = 0;
		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++){
				if (intensity[j+n1*i]>epsilon){
					tmp = sqrt( pow(in[j+n1*i][0],2) + pow(in[j+n1*i][1],2) );
					norm += pow(intensity[j+n1*i] - tmp, 2);
				}
			}
		}
		errorF[t] = norm;
	}

	in_node.value = errorF[0];
	in_node.index = 0;
	for (int t=0;t<num;t++){
		if (errorF[t]<in_node.value){
			in_node.value = errorF[t];
			in_node.index = t;
		}
	}

	// Now in_node.value is the min of errorF in each processor.
	in_node.index = myid*num + in_node.index;

	// sychronize every processor
	MPI_Barrier(MPI_COMM_WORLD);

	// determine the nodes with min errorF
	MPI_Reduce(&in_node, &out_node, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

	if (myid==0){
		minval = out_node.value;
		minrank = out_node.index / num;
		minindex = out_node.index % num;
		printf("minval = %E, minrank = %d, minindex = %d\n", minval, minrank, minindex);
	}

	// Broadcast minrank and minindex to every node.
	MPI_Bcast(&minrank, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&minindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	// create shift_template and then broadcast it to every node
	if (myid==minrank){
		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++)
				shift_template[j+n1*i] = image_tmp[j+n1*i][minindex];
		}
	}
	MPI_Bcast(shift_template, n1*n2, MPI_DOUBLE, minrank, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	if (myid==0){
		sprintf(fn, "%s/template_gen%d.dat", dir, q+1);
		FILE *fout = fopen(fn, "w");
		fwrite(shift_template,sizeof(double),n1*n2,fout);
		fclose(fout);
		printf("This is the %d-th generation.\n", q+1);
	}

	for (i=0;i<n1*n2;i++){
		if (temp_max < shift_template[i])
			temp_max = shift_template[i];
	}

	for (int t=0;t<num;t++){
	
		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++)
				shift_model[j+n1*i] = image_tmp[j+n1*i][t];
		}
	
		for (i=0;i<n1*n2;i++){
			if (shift_max < shift_model[i])
				shift_max = shift_model[i];
		}

		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++)
				shift_model[j+n1*i] = shift_model[j+n1*i]*(temp_max/shift_max);
		}

		outcome = shift_and_invert(shift_model,shift_template,n1,n2,step,support1,support2);

		// Now image_tmp stores the images which have been shifted or inverted.
		for (i=0;i<n1;i++){
			for (j=0;j<n2;j++)
				image_tmp[j+n1*i][t] = outcome[j+n1*i]*(shift_max/temp_max);
		}
	}
}

free(intensity);
free(errorF);
free(image_tmp);
free(HIOinput);
free(M);
free(shift_template);
free(shift_model);
free(outcome);
free(shift);

fftw_destroy_plan(forward_p);
fftw_destroy_plan(inverse_p);

double end_time = MPI_Wtime();
if (myid==0)
	printf("Computation time = %f (s)\n",end_time - start_time);

// end MPI
MPI_Finalize();

return 0;}
