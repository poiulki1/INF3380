#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

struct Matrix{
	char* filename;
	int num_rows, num_cols;
	double** array;
};

void distribute_matrix(double **my_a, double **whole_matrix, int m, int n, int my_m,
					   int my_n, int procs_per_dim, int mycoords[2], MPI_Comm *comm_col,
					   MPI_Comm *comm_row);
void MatrixMultiply(double **a, double **b, double **c, int m, int n, int l);
void read_matrix_bin(struct Matrix *mat);
void write_matrix_bin(struct Matrix *mat);
void alloc_matrix(struct Matrix *mat, int row_num, int col_num);

int main(int argc, char *argv[])
{
    int my_rank, num_procs;
	int temp_rows, temp_cols;
	int my_rows, my_cols;
	int sp;
	
	int dims[2], periods[2], my2drank, mycoords[2];
	MPI_Comm comm_2d, comm_col, comm_row;
	
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	
	sp = (int) sqrt((float) num_procs); //sqrt(num_procs)
	
	
//	printf("Rank: %d  sp: %d \n", my_rank, sp); //test
	struct Matrix A, B, C;
	if(my_rank == 0){
		A.filename = "../small_matrix_a.bin";
		B.filename = "../small_matrix_b.bin";
		C.filename = "../small_matrix_c_result.bin";
		read_matrix_bin(&A);
		read_matrix_bin(&B);
		alloc_matrix(&C, A.num_rows, B.num_cols);
		write_matrix_bin(&C);
		temp_rows = A.num_rows;
		temp_cols = A.num_cols;

		//MatrixMultiply(A.array, B.array, C.array, A.num_rows, B.num_cols,
		//		   B.num_rows);
	}

	MPI_Bcast(&temp_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&temp_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	dims[0] = dims[1] = sp;
    periods[0] = periods[1] = 1;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);

	MPI_Comm_rank(comm_2d, &my2drank);
    MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

    MPI_Cart_sub(comm_2d, (int[]){0, 1}, &comm_row);
    MPI_Cart_sub(comm_2d, (int[]){1, 0}, &comm_col);
	
	my_rows = temp_rows/sp + (mycoords[0] < temp_rows%sp);
	my_cols =temp_cols/sp + (mycoords[1] < temp_cols%sp);

	struct Matrix my_matrix_A, my_matrix_B;
	alloc_matrix(&my_matrix_A, my_rows, my_cols);
	alloc_matrix(&my_matrix_B, my_cols, my_rows); // B = transponert A
	printf("HAHA \n");
	distribute_matrix(my_matrix_A.array, A.array, A.num_rows, A.num_cols,
					  my_rows, my_cols, sp, mycoords,
					  &comm_col, &comm_row);
	printf("HAHAHAH\n");
	MPI_Finalize();
    return 0;
}

/* This matrix performs a serial matrix-matrix multiplication c = a * b. */
void MatrixMultiply(double **a, double **b, double **c, int m, int n, int l)
{
	int i, j, k;
	for (i=0; i < m; i++){
		for (j=0; j < n; j++){
			for (k=0; k < l; k++){
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
    return;
}

/* The meat of the program. Read through carefully. Two lines have been dummied out! */
void distribute_matrix(double **my_a, double **whole_matrix, int m, int n, int my_m, int my_n, int procs_per_dim, int mycoords[2], MPI_Comm *comm_col, MPI_Comm *comm_row)
{

    /* Buffers. */
    double *senddata_columnwise, *senddata_rowwise;

    /* Scatterv variables, step 1. */
    int *displs_y, *sendcounts_y, *everyones_m;

    /* Scatterv variables, step 2. */
    int *displs_x, *sendcounts_x, *everyones_n;
    MPI_Datatype columntype, columntype_scatter, columntype_recv, columntype_recv_scatter;

    displs_x = displs_y = sendcounts_x = sendcounts_y = everyones_m = everyones_n = NULL;
    senddata_columnwise = senddata_rowwise = NULL;

    if (mycoords[0] == 0 && mycoords[1] == 0)
    {
        senddata_columnwise = *whole_matrix;
    }
/* Step 1. Only first column participates. */
    if (mycoords[1] == 0)
    {
        if (mycoords[0] == 0)
        {
            everyones_m = (int *) calloc(procs_per_dim, sizeof(int));
            sendcounts_y = (int *)calloc(procs_per_dim, sizeof(int));
            displs_y = (int *)calloc(procs_per_dim + 1, sizeof(int));
        }

        MPI_Gather(&my_m, 1, MPI_INT, everyones_m, 1, MPI_INT, 0, *comm_col);

        if (mycoords[0] == 0)
        {
            displs_y[0] = 0;
            for (int i = 0; i < procs_per_dim; i++)
            {
                sendcounts_y[i] = n * everyones_m[i];
                displs_y[i + 1] = displs_y[i] + sendcounts_y[i];
            }
        }
        senddata_rowwise = (double *) calloc(my_m * n, sizeof(double));

        MPI_Scatterv(senddata_columnwise, sendcounts_y, displs_y, MPI_DOUBLE, senddata_rowwise, my_m * n, MPI_DOUBLE, 0, *comm_col);
    }

    /* Step 2: Send data rowwise. */
    /* First, create the column data types. */
    MPI_Type_vector(my_m, 1, my_n, MPI_DOUBLE, &columntype);    /* (count, blocklength, stride, datatype old, *new)Dummied out. */
    MPI_Type_commit(&columntype);
    MPI_Type_create_resized(columntype, 0, sizeof(double), &columntype_scatter);
    MPI_Type_commit(&columntype_scatter);

    /* Receivers need their own data type, or their data will be transposed! */
    MPI_Type_vector(my_m, 1, my_n, MPI_DOUBLE, &columntype_recv);
    MPI_Type_commit(&columntype_recv);
    MPI_Type_create_resized(columntype_recv, 0, sizeof(double), &columntype_recv_scatter);
    MPI_Type_commit(&columntype_recv_scatter);

    if (mycoords[1] == 0)
    {
		printf("6th check \n");
        everyones_n = (int *) calloc(procs_per_dim, sizeof(int));
        sendcounts_x = (int *) calloc(procs_per_dim, sizeof(int));
        displs_x = (int *) calloc(procs_per_dim + 1, sizeof(int));
    }

    MPI_Gather(&my_n, 1, MPI_INT, everyones_n, 1, MPI_INT, 0, *comm_row);

    if (mycoords[1] == 0)
    {
		printf("7th check \n");
        displs_x[0] = 0;
        for (int i = 0; i < procs_per_dim; ++i)
        {
            sendcounts_x[i] = everyones_n[i];
            displs_x[i + 1] = displs_x[i] + sendcounts_x[i];

        }
    }

    MPI_Scatterv(senddata_rowwise, sendcounts_x, displs_x, columntype_scatter, *my_a, my_n, columntype_recv_scatter, 0, *comm_row);

    /* And we have our matrices! */

    /* Finally, free everything. */
	printf("JAA!\n");
    MPI_Type_free(&columntype_recv_scatter);
    MPI_Type_free(&columntype_recv);

    if (mycoords[1] == 0)
    {
		printf("8th check \n");
        free(displs_x);
        free(sendcounts_x);
		printf("AJSAJFSKAF");
        MPI_Type_free(&columntype_scatter);
        MPI_Type_free(&columntype);
		
        if (mycoords[0] == 0)
        {
			printf("9th check \n");
            free(displs_y);
            free(sendcounts_y);
        }
		
        free(senddata_rowwise);
    }
	printf("10th check \n");
}

void alloc_matrix(struct Matrix *mat, int row_num, int col_num){
	mat->num_rows = row_num;
	mat->num_cols = col_num;
	mat->array = (double**)calloc(row_num, sizeof(double*));
	mat->array[0] = (double*)calloc(row_num*col_num, sizeof(double));
	for(int i=1; i<row_num; i++){
		mat->array[i] = mat->array[i-1] + col_num;
	}
}

void read_matrix_bin(struct Matrix *mat){
	int i;
	FILE* fp = fopen(mat->filename, "rb");
	fread(&(mat->num_rows), sizeof(int), 1, fp);
	fread(&(mat->num_cols), sizeof(int), 1, fp);

	/* storage allocation of the matrix */
	mat->array = (double**)malloc((mat->num_rows)*sizeof(double*));
	mat->array[0] = (double*)malloc((mat->num_rows)*(mat->num_cols)*sizeof(double));
	for(i = 1; i < (mat->num_rows); i++){
		mat->array[i] = mat->array[i-1] + (mat->num_cols);
	}
	/* read in the entire matrix */
	fread(mat->array[0], sizeof(double), (mat->num_rows)*(mat->num_cols), fp);
	fclose(fp);
}

void write_matrix_bin(struct Matrix *mat){
	FILE *fp = fopen(mat->filename, "wb");
	fwrite(&(mat->num_rows), sizeof(int), 1, fp);
	fwrite(&(mat->num_cols), sizeof(int), 1, fp);
	fwrite(mat->array[0], sizeof(double), (mat->num_rows)*(mat->num_cols), fp);
	fclose(fp);
}
