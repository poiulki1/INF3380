#include "common_serial.h"
#include <stdio.h>
#include <stdlib.h>

void read_cmla(int argc, char **argv, float *kappa, int *iters,
			   char **input_jpeg_filename, char **output_jpeg_filename){
	if(argc == 5){
		*kappa = atof(argv[1]);
		*iters = atoi(argv[2]);
		*input_jpeg_filename = argv[3];
		*output_jpeg_filename = argv[4];
	}
	else{
		printf("Error: Please enter 4 arguments: kappa, iters, input_name and output_name");
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

void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters,
							 int n, int m){
	for(int k=0; k<iters; k++){
		for(int i=1; i < m-1; i++){
			for(int j=1; j<n-1; j++){
				u_bar->image_data[i][j] = u->image_data[i][j] +
					kappa*(u->image_data[i-1][j] + u->image_data[i][j-1] -
						   4*u->image_data[i][j] + u->image_data[i][j+1] +
						   u->image_data[i+1][j]);
			}
		}
		copy_pix(u, u_bar, n, m);
	}
}


int main(int argc, char *argv[])
{
	int m, n, c, iters;
	float kappa;
	image u, u_bar;
	unsigned char *image_chars;
	char *input_jpeg_filename, *output_jpeg_filename;
	/* read from command line: kappa, iters, input_jpeg_filename,
	 * output_jpeg_filename */
	read_cmla(argc, argv, &kappa, &iters, &input_jpeg_filename,
			  &output_jpeg_filename); 
	

	import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
	allocate_image(&u, m, n);
	allocate_image(&u_bar, m, n);
	convert_jpeg_to_image(image_chars, &u, n, m);
	iso_diffusion_denoising(&u, &u_bar, kappa, iters, n, m);
	convert_image_to_jpeg(&u_bar, image_chars, n, m);
	export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
	deallocate_image(&u, m);
	deallocate_image(&u_bar, m);
	return 0;
}
