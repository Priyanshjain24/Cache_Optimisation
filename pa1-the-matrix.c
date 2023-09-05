
// CS 683 (Autumn 2023)
// PA 1: The Matrix

// includes
#include <stdio.h>
#include <time.h>			// for time-keeping
#include <xmmintrin.h> 		// for intrinsic functions
#include <immintrin.h>

// defines
// NOTE: you can change this value as per your requirement
#define BLOCK_SIZE 50		// size of the block
#define Prefetch_Jump 8     // No of iterations later for which prefetch is called

/**
 * @brief 		Generates random numbers between values fMin and fMax.
 * @param 		fMin 	lower range
 * @param 		fMax 	upper range
 * @return 		random floating point number
 */
double fRand(double fMin, double fMax) {

	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = fRand(0.0001, 1.0000); // random values between 0 and 1
		}
	}
}

/**
 * @brief 		Initialize result matrix of given dimension with 0.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_result_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = 0.0;
		}
	}
}

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 */
void normal_mat_mul(double *A, double *B, double *C, int dim) {

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			for (int k = 0; k < dim; k++) {
				C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
			}
		}
	}
}

/**
 * @brief 		Task 1: Performs matrix multiplication of two matrices using blocking.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the block size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void blocking_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

	for (int i=0; i<dim; i+=block_size){
		for (int j=0; j<dim; j+=block_size){
			for (int k=0; k<dim; k+=block_size){
				for (int k1=k; k1<k+block_size; k1++){
					for (int i1=i; i1<i+block_size; i1++){
						for (int j1 = j; j1<j+block_size; j1++){
							C[i1 * dim + j1] += A[i1 * dim + k1] * B[k1 * dim + j1];
						}
					}
				}
			}
		}
	}
}

/**
 * @brief 		Task 2: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/

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

/**
 * @brief 		Task 3: Performs matrix multiplication of two matrices using software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/

void prefetch_mat_mul(double *A, double *B, double *C, int dim) {

	for (int i = 0; i < dim; i++) {
		__builtin_prefetch(&A[i*dim],0,3);
		__builtin_prefetch(&B[0],0,1);
		__builtin_prefetch(&C[i*dim],1,1);

			double sum=0.0;

			for (int k = 0; k < dim; k++) {
				__builtin_prefetch(&B[(k+Prefetch_Jump)*dim],0,1);
				// if((i*dim+k+Prefetch_Jump)%Prefetch_Jump==0)__builtin_prefetch(&A[i*dim+k+Prefetch_Jump],0,3);
				__builtin_prefetch(&A[i*dim+k+Prefetch_Jump],0,3);

				sum += A[i * dim + k] * B[k * dim];
			}

			C[i*dim]=sum;

		for (int j = 1; j < dim; j++) {
			// if((i*dim+j)%Prefetch_Jump==0)__builtin_prefetch(&C[i*dim+j],1,1);
			__builtin_prefetch(&C[i*dim+j],1,1);

			sum=0.0;

			for (int k = 0; k < dim; k++) {
				__builtin_prefetch(&B[(k+Prefetch_Jump)*dim+j],0,1);
				sum += A[i * dim + k] * B[k * dim + j];
			}

			C[i*dim+j]=sum;
		}
	}

	return;
}

/**
 * @brief 		Bonus Task 1: Performs matrix multiplication of two matrices using blocking along with SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
*/

void blocking_simd_mat_mul(double *A, double *B, double *C, int dim, int block_size) {
    for (int i = 0; i < dim; i += block_size) {
        for (int j = 0; j < dim; j += block_size) {
            for (int k = 0; k < dim; k += block_size) {
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


/**
 * @brief 		Bonus Task 2: Performs matrix multiplication of two matrices using blocking along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
*/
void blocking_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

	double temp=0.0;

	for (int i=0; i<dim; i+=block_size){
		for (int j=0; j<dim; j+=block_size){
			for (int k=0; k<dim; k+=block_size){
				
				for (int k1=k; k1<k+block_size; k1++){

					__builtin_prefetch(&A[i*dim+k1],0,1);
					__builtin_prefetch(&B[k1*dim],0,3);
					__builtin_prefetch(&C[i*dim+j],1,1);


					temp=A[i*dim+k1];

					for(int j1=j; j1<j+block_size; j1++)
					{
						__builtin_prefetch(&B[k1*dim + Prefetch_Jump],0,3);
						__builtin_prefetch(&C[i*dim+j1+Prefetch_Jump],1,1);
						C[i*dim+j1] += temp * B[k1 * dim + j1];
					}

					for (int i1=i+1; i1<i+block_size; i1++){
						__builtin_prefetch(&A[i1*dim+k1+Prefetch_Jump],0,1);

						temp=A[i1*dim+k1];

						for (int j1 = j; j1<j+block_size; j1++){
							__builtin_prefetch(&C[i1*dim+j1+Prefetch_Jump],1,1);
							C[i1 * dim + j1] += temp * B[k1 * dim + j1];
						}

					}
				}
				
			}
		}
	}
}

/**
 * @brief 		Bonus Task 3: Performs matrix multiplication of two matrices using SIMD instructions along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_prefetch_mat_mul(double *A, double *B, double *C, int dim) {

    for (int i = 0; i < dim; ++i) {
        for (int k = 0; k < dim; k++) {
			__builtin_prefetch(&A[i*dim+k+Prefetch_Jump],0,1);
			__builtin_prefetch(&C[i*dim],0,3);
			__builtin_prefetch(&B[k*dim],0,3);
            __m512d constant_vector = _mm512_set1_pd(A[i * dim + k]);

            for (int j = 0; j < dim - (dim % 8); j += 8) {
				__builtin_prefetch(&C[i*dim+j+Prefetch_Jump],0,3);
				__builtin_prefetch(&B[k*dim+j+Prefetch_Jump],0,3);
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


/**
 * @brief 		Bonus Task 4: Performs matrix multiplication of two matrices using blocking along with SIMD instructions and software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void blocking_simd_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

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
				if(abs(A[i*matrix_dim+j]-B[i*matrix_dim+j])>1e-6) return -1;
			}
		}

	return 1;
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Pass the matrix dimension as argument :)\n\n");
		return 0;
	}

	else {
		int matrix_dim = atoi(argv[1]);

		// variables definition and initialization
		clock_t t_normal_mult, t_blocking_mult, t_prefetch_mult, t_simd_mult, t_blocking_simd_mult, t_blocking_prefetch_mult, t_simd_prefetch_mult, t_blocking_simd_prefetch_mult;
		double time_normal_mult, time_blocking_mult, time_prefetch_mult, time_simd_mult, time_blocking_simd_mult, time_blocking_prefetch_mult, time_simd_prefetch_mult, time_blocking_simd_prefetch_mult;

		double *A = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));
		double *B = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));
		double *C = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, matrix_dim, matrix_dim);
		initialize_matrix(B, matrix_dim, matrix_dim);

		// perform normal matrix multiplication
		t_normal_mult = clock();
		normal_mat_mul(A, B, C, matrix_dim);
		t_normal_mult = clock() - t_normal_mult;

		time_normal_mult = ((double)t_normal_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Normal matrix multiplication took %f seconds to execute \n\n", time_normal_mult);

		double *Z = (double *)malloc(matrix_dim*matrix_dim*sizeof(double));
		copy(C,Z,matrix_dim);
		print(Z,matrix_dim);


	#ifdef OPTIMIZE_BLOCKING
		// Task 1: perform blocking matrix multiplication

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_blocking_mult = clock();
		blocking_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_mult = clock() - t_blocking_mult;

		time_blocking_mult = ((double)t_blocking_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking matrix multiplication took %f seconds to execute \n", time_blocking_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_mult);
		printf("%d \n",check(C,Z,matrix_dim));
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 2: perform matrix multiplication with SIMD instructions

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_simd_mult = clock();
		simd_mat_mul(A, B, C, matrix_dim);
		t_simd_mult = clock() - t_simd_mult;

		time_simd_mult = ((double)t_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD matrix multiplication took %f seconds to execute \n", time_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_mult);
		printf("%d \n",check(C,Z,matrix_dim));
		// print(C,matrix_dim);
		
	#endif

	#ifdef OPTIMIZE_PREFETCH
		// Task 3: perform matrix multiplication with prefetching

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_prefetch_mult = clock();
		prefetch_mat_mul(A, B, C, matrix_dim);
		t_prefetch_mult = clock() - t_prefetch_mult;

		time_prefetch_mult = ((double)t_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Prefetching matrix multiplication took %f seconds to execute \n", time_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_prefetch_mult);
		printf("%d \n",check(C,Z,matrix_dim));
	#endif

	#ifdef OPTIMIZE_BLOCKING_SIMD
		// Bonus Task 1: perform matrix multiplication using blocking along with SIMD instructions

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_blocking_simd_mult = clock();
		blocking_simd_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_mult = clock() - t_blocking_simd_mult;

		time_blocking_simd_mult = ((double)t_blocking_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD matrix multiplication took %f seconds to execute \n", time_blocking_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_mult);
		print(C,matrix_dim);
		printf("%d \n",check(C,Z,matrix_dim));
		
	#endif

	#ifdef OPTIMIZE_BLOCKING_PREFETCH
		// Bonus Task 2: perform matrix multiplication using blocking along with software prefetching

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_blocking_prefetch_mult = clock();
		blocking_prefetch_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_prefetch_mult = clock() - t_blocking_prefetch_mult;

		time_blocking_prefetch_mult = ((double)t_blocking_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with prefetching matrix multiplication took %f seconds to execute \n", time_blocking_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_prefetch_mult);
		printf("%d \n",check(C,Z,matrix_dim));
	#endif

	#ifdef OPTIMIZE_SIMD_PREFETCH
		// Bonus Task 3: perform matrix multiplication using SIMD instructions along with software prefetching

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_simd_prefetch_mult = clock();
		simd_prefetch_mat_mul(A, B, C, matrix_dim);
		t_simd_prefetch_mult = clock() - t_simd_prefetch_mult;

		time_simd_prefetch_mult = ((double)t_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD with prefetching matrix multiplication took %f seconds to execute \n", time_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_prefetch_mult);
		printf("%d \n",check(C,Z,matrix_dim));
	#endif

	#ifdef OPTIMIZE_BLOCKING_SIMD_PREFETCH
		// Bonus Task 4: perform matrix multiplication using blocking, SIMD instructions and software prefetching

		// initialize result matrix to 0
		initialize_result_matrix(C, matrix_dim, matrix_dim);

		t_blocking_simd_prefetch_mult = clock();
		blocking_simd_prefetch_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_prefetch_mult = clock() - t_blocking_simd_prefetch_mult;

		time_blocking_simd_prefetch_mult = ((double)t_blocking_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD and prefetching matrix multiplication took %f seconds to execute \n", time_blocking_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_prefetch_mult);
		printf("%d \n",check(C,Z,matrix_dim));
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
