
#include "cudaLib.cuh"
#include "cpuLib.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) y[i] = scale * x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	int vectorBytes = vectorSize * sizeof(float);
	float *a, *b, *c;
	float *a_d, *b_d, *c_d;

	// Initialize vectors in CPU
	a = (float*)malloc(vectorBytes);
	b = (float*)malloc(vectorBytes);
	c = (float*)malloc(vectorBytes);
	if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}
	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	std::memcpy(c, b, vectorSize * sizeof(float));
	float scale = 2.0f;

	// Transfer from host to device
	cudaMalloc((void**) &a_d, vectorBytes);
	cudaMalloc((void**) &b_d, vectorBytes);
	cudaMalloc((void**) &c_d, vectorBytes);
	cudaMemcpy(a_d, a, vectorBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, vectorBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c, vectorBytes, cudaMemcpyHostToDevice);

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif

	// Kernel code
	saxpy_gpu<<<ceil(vectorSize/256.0), 256>>>(a_d, c_d, scale, vectorSize);

	// Transfer from device to host
	cudaMemcpy(c, c_d, vectorBytes, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";


	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
