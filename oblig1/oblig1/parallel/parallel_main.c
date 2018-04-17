#include "common_parallel.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void read_cmla(int argc, char **argv, float *kappa, int *iters,
			   char **input_jpeg_filename, char **output_jpeg_filename){
	if(argc == 5){
		*kappa = atof(argv[1]);
		*iters = atoi(argv[2]);
		*input_jpeg_filename = argv[3];
		*output_jpeg_filename = argv[4];
	}
	else{
		
		printf("Error: Please enter 4 arguments: kappa, iters, input_name and output_name\n");
	}
}

void allocate_image(image *u, int m, int n){
	u->image_data = calloc(m, sizeof(float *));
	for(int i=0; i<m ; i++){
		u->image_data[i] = calloc(n,sizeof(float));
	}
}

void deallocate_image(image *u, int m){
	for(int i = m-1; i >=0; i--){
		free(u->image_data[i]);
	}
	free(u->image_data);
}

void convert_jpeg_to_image(const unsigned char* image_chars, image *u, int n,
						   int m){
	for(int i = 0; i<m; i++){
		for(int j=0; j<n; j++){
			u->image_data[i][j] = (float) image_chars[i*n+j];
		}
	}
}
void convert_image_to_jpeg(const image *u, unsigned char* image_chars, int n,
						   int m){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			image_chars[i*n+j] = (unsigned char) u->image_data[i][j];
		}
	}
}

void copy_pix(image *u_bar, image *u, int n, int m){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			u_bar->image_data[i][j] = u->image_data[i][j];
		}
	}
}

void iso_diffusion_denoising_parallel(image *u, image *u_bar, float kappa, int iters,
									  int n, int m, int rank, int num_procs){
	for(int k=0; k<iters; k++){
		for(int i=1; i < m-1; i++){
			for(int j=1; j<n-1; j++){
				u_bar->image_data[i][j] = u->image_data[i][j] +
					kappa*(u->image_data[i-1][j] + u->image_data[i][j-1] -
						   4*u->image_data[i][j] + u->image_data[i][j+1] +
						   u->image_data[i+1][j]);
			}
		}		
		if(rank != num_procs-1){
			MPI_Send(u_bar->image_data[m-2], n, MPI_FLOAT, rank+1, k,
									  MPI_COMM_WORLD);
			
			MPI_Recv(u_bar->image_data[m-1], n, MPI_FLOAT, rank+1, k,
							 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		if(rank != 0){
			MPI_Recv(u_bar->image_data[0], n, MPI_FLOAT, rank-1, k, MPI_COMM_WORLD,
					 MPI_STATUS_IGNORE);
			
			MPI_Send(u_bar->image_data[1], n, MPI_FLOAT, rank-1, k,
					 MPI_COMM_WORLD);
		}
		copy_pix(u, u_bar, n, m);
	}
}


int main(int argc, char *argv[])
{
	int m, n, c, iters;
	int my_m, my_n, my_rank, num_procs, rest, sum;
	float kappa;
	image u, u_bar, whole_image;
	unsigned char *image_chars, *my_image_chars, *temp;
	char *input_jpeg_filename, *output_jpeg_filename;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	read_cmla(argc, argv, &kappa, &iters, &input_jpeg_filename,
			  &output_jpeg_filename); 
	
	if (my_rank==0) {
		import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
		allocate_image(&whole_image, m, n);
	}
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	/* divide the m x n pixels evenly among the MPI processes */	

	rest = m%num_procs;
	sum = 0;
	my_n = n;
	
	int *sendcounts = calloc(sizeof(int), num_procs);
	int *displs = calloc(sizeof(int), num_procs);
	int *temp_m = calloc(sizeof(int), num_procs);
   
	for(int i=0; i<num_procs; i++){
		temp_m[i] = m/num_procs;
		if(i<rest){
			temp_m[i] ++;
		}
		
		if(i==0){
			temp_m[i] ++;
		}
		
		if(i != 0 && i != num_procs-1){
			temp_m[i] += 2;
			sum -= 2*my_n;
		}
		
		if(i==num_procs-1){
			temp_m[i] ++;
			sum -= 2*my_n;
		}
		
		sendcounts[i] = temp_m[i]*my_n;
		displs[i] = sum;
		sum += sendcounts[i];
	}
	
	allocate_image(&u, temp_m[my_rank], my_n);
	allocate_image(&u_bar, temp_m[my_rank], my_n);
	
	/* each process asks process 0 for a partitioned region */
	/* of image_chars and copy the values into u */
	/*  ...  */
	my_image_chars = calloc(temp_m[my_rank]*n, sizeof(unsigned char));

	MPI_Scatterv(image_chars, sendcounts, displs, MPI_UNSIGNED_CHAR,
				 my_image_chars, temp_m[my_rank]*my_n, MPI_UNSIGNED_CHAR, 0,
				 MPI_COMM_WORLD); 

	convert_jpeg_to_image(my_image_chars, &u, my_n, temp_m[my_rank]);
	convert_jpeg_to_image(my_image_chars, &u_bar, my_n, temp_m[my_rank]);
	
	iso_diffusion_denoising_parallel(&u, &u_bar, kappa, iters, my_n,
									 temp_m[my_rank], my_rank, num_procs);
	
	/* each process sends its resulting content of u_bar to process 0 */
	/* process 0 receives from each process incoming values and */
	/* copy them into the designated region of struct whole_image */
	/*  ...  */
	convert_image_to_jpeg(&u_bar, my_image_chars, my_n, temp_m[my_rank]);

	MPI_Gatherv(my_image_chars, temp_m[my_rank]*my_n, MPI_UNSIGNED_CHAR, image_chars,
				sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	
	if (my_rank==0) {
		//convert_image_to_jpeg(&whole_image, image_chars, n, m);
		export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
		deallocate_image(&whole_image, m);
		}

	deallocate_image(&u, temp_m[my_rank]);
	deallocate_image(&u_bar, temp_m[my_rank]);
	
	MPI_Finalize();
	return 0; }
