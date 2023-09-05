#include <stdio.h>
#include <xmmintrin.h> 		// for intrinsic functions
#include <immintrin.h>

#define dataType double

double fRand(double fMin, double fMax) {

	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void simd_mat_mul(double *A, double *B, double *C, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int k = 0; k < dim; k++) {
            __m512d constant_vector = _mm512_set1_pd(A[i * dim + k]);

            for (int j = 0; j < dim - (dim % 8); j += 8) {
                __m512d c = _mm512_loadu_pd(&C[i * dim + j]); // Load 8 elements from row i of matrix C
                __m512d b = _mm512_loadu_pd(&B[k * dim + j]); // Load 8 elements from column j of matrix B
                c = _mm512_fmadd_pd(constant_vector, b, c); // Fused multiply-add operation
                _mm512_storeu_pd(&C[i * dim + j], c); // Store the result back into c
            }

            // Handle the remaining values normally
            for (int j = dim - (dim % 8); j < dim; ++j) {
                C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
            }
        }
    }
}


void normal_mat_mul(double *A, double *B, double *C, int dim) {

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			for (int k = 0; k < dim; k++) {
				C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
			}
		}
	}
}


void initialize_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = fRand(0.0001, 1.0000); // random values between 0 and 1
		}
	}
}

void initialize_matrix2(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			// if(i==j) 
			matrix[i*cols + j]=1;
			// else matrix[i*cols+j] = 0;
		}
	}
}

void initialize_result_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = 0.0;
		}
	}
}

void print(double *C, int matrix_dim)
{
	printf("\n");
	for(int i=0; i<matrix_dim; i++)
		{
			for(int j=0; j<matrix_dim; j++)
			{
				printf("%f ",C[i*matrix_dim+j]);
			}
			printf("\n");
		}
	printf("\n");
	return;
}

void copy(double *C, double *Z, int matrix_dim)
{
	for(int i=0; i<matrix_dim; i++)
		{
			for(int j=0; j<matrix_dim; j++)
			{
				Z[i*matrix_dim+j]=C[i*matrix_dim+j];
			}
		}
	return;
}

int check(double *A, double *B, int matrix_dim)
{
	for(int i=0; i<matrix_dim; i++)
		{
			for(int j=0; j<matrix_dim; j++)
			{
				if(abs(A[i*matrix_dim+j]-B[i*matrix_dim+j])>1e-12) 
                {
                    printf("NOT MATCHED %d %d %f %f",i,j,A[i*matrix_dim+j],B[i*matrix_dim+j]);
                    return -1;
                }
			}
		}

	return 1;
}

void blocking_simd_mat_mul(double *A, double *B, double *C, int dim, int block_size) {
    for (int i = 0; i < dim; i += block_size) {
        for (int k = 0; k < dim; k += block_size) {
            for (int j = 0; j < dim; j += block_size) {

				// printf(" i: %d j: %d k: %d \n", i,j,k);
				for (int ii = i; ii < i+block_size; ++ii) {
					for (int kk = k; kk < k+block_size; ++kk) {
						__m512d constant_vector = _mm512_set1_pd(A[ii * dim + kk]);

						for (int jj = j; jj < (j+block_size) - ((j+block_size) % 8); jj += 8) {
							__m512d c = _mm512_loadu_pd(&C[ii * dim + jj]); // Load 8 elements from row i of matrix C
							__m512d b = _mm512_loadu_pd(&B[kk * dim + jj]); // Load 8 elements from column j of matrix B
							c = _mm512_fmadd_pd(constant_vector, b, c); // Fused multiply-add operation
							_mm512_storeu_pd(&C[ii * dim + jj], c); // Store the result back into c
						}

						// Handle the remaining values normally
						for (int jj = (j+block_size) - ((j+block_size) % 8); jj < j+block_size; ++jj) {
							C[ii * dim + jj] += A[ii * dim + kk] * B[kk * dim + jj];
						}
					}
				}
			}
		}
	}
}

void printRatio(double *A, double *B, int matrix_dim)
{
		printf("\n");
	for(int i=0; i<matrix_dim; i++)
		{
			for(int j=0; j<matrix_dim; j++)
			{
				double d=A[i*matrix_dim+j]/(1.0*B[i*matrix_dim+j]);
				printf("%f ",d);
			}
			printf("\n");
		}
	printf("\n");
	return;
}


#include <immintrin.h>

void blocking_simd_mat_mul2(double *A, double *B, double *C, int dim, int block_size) {
    for (int i = 0; i < dim; i += block_size) {
        for (int k = 0; k < dim; k += block_size) {
            for (int j = 0; j < dim; j += block_size) {
                for (int ii = i; ii < i + block_size; ii++) {
                    for (int kk = k; kk < k + block_size; kk++) {
                        __m512d a = _mm512_broadcastsd_pd(_mm_load_sd(&A[ii * dim + kk]));
                        for (int jj = j; jj < j + block_size; jj += 8) {
                            __m512d b = _mm512_loadu_pd(&B[kk * dim + jj]);
                            __m512d c = _mm512_loadu_pd(&C[ii * dim + jj]);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_storeu_pd(&C[ii * dim + jj], c);
                        }
                    }
                }
            }
        }
    }
}


int main()
{
    int dim;
    scanf("%d",&dim);

    dataType *A = (dataType *) malloc(dim*dim*sizeof(dataType));
    dataType *B = (dataType *) malloc(dim*dim*sizeof(dataType));
    dataType *C = (dataType *) calloc(dim*dim,sizeof(dataType));
    dataType *D = (dataType *) calloc(dim*dim,sizeof(dataType));

    initialize_result_matrix(C,dim,dim);
    initialize_result_matrix(D,dim,dim);
    initialize_matrix(A,dim,dim);
    initialize_matrix(B,dim,dim);

    normal_mat_mul(A,B,C,dim);
    blocking_simd_mat_mul2(A,B,D,dim,2);

    print(C,dim);
    print(D,dim);
    printf("\n %d \n",check(C,D,dim));
	printRatio(D,C,dim);

    return 0;
}