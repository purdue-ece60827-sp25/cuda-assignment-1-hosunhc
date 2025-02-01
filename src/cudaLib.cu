
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
	auto tStart = std::chrono::high_resolution_clock::now();

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

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread 
 is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for 
 each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

/*
generateThreadCount = pSumSize
array pSums[pSumSize] = [tid_0, tid_1, tid_2, ... tid_n]
each tid = "sampleSize" samples
*/
__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// pSumSize is max number of threads
	if (threadId >= pSumSize) return;

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), threadId, 0, &rng);
	
	// Random values and count
	uint64_t hitCount = 0;
	for (uint64_t i = 0; i < sampleSize; i++) {
		// Generate "sampleSize" random values
		float x = curand_uniform(&rng); 
		float y = curand_uniform(&rng);
		if (int(x*x + y*y) == 0) hitCount++;
	}

	pSums[threadId] = hitCount;
}

/*
reduceThreadCount = pSumSize / reduceSize
array totals[reduceThreadCount] = [reduced_val_0, reduced_val_1, ...]
array pSums[pSumSize] = [tid_0, tid_1, tid_2, ... tid_n]
*/
__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	// Each thread works: data[i:i+reduceSize]
    uint64_t startIdx = threadId * reduceSize;
    uint64_t endIdx = startIdx + reduceSize;

	// Combine partial sums
	uint64_t reducePSum = 0;
	for (uint64_t i = startIdx; i < endIdx && i < pSumSize; i++) {
		reducePSum += pSums[i];
	}

	totals[threadId] = reducePSum; 
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

/*
generateThreadCount = GENERATE_BLOCKS = 1024
sampleSize = SAMPLE_SIZE = 1e6
reduceThreadCount = REDUCE_BLOCKS = GENERATE_BLOCKS / REDUCE_SIZE = 1024/32 = 32
reduceSize = REDUCE_SIZE = 32
*/
double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	uint64_t pSumSize = generateThreadCount;

	// Generate random samples: n = generateThreadCount * sampleSize
	// #(generateThreadCount) Partial Sums
	uint64_t *pSums_d;
	cudaMalloc(&pSums_d, pSumSize * sizeof(uint64_t));
	generatePoints<<<ceil(pSumSize/reduceSize),reduceSize>>>(pSums_d, pSumSize, sampleSize);
	
	#ifndef DEBUG_PRINT_DISABLE
		uint64_t *pSums_h;
		pSums_h = (uint64_t*)malloc(pSumSize * sizeof(uint64_t));
		cudaMemcpy(pSums_h, pSums_d, pSumSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		uint64_t sum = 0;
		for (int i = 0; i < pSumSize; i++) sum += pSums_h[i];
		std::cout << "pSum SUMS = " << sum << std::endl;
	#endif

	// Reduce partial sums: n = generateThreadCount / reduceSize = reduceThreadCount
	// totals num elements: #(reduceThreadCount) 
	uint64_t *totals_h, *totals_d;
	uint64_t totalsSize = reduceThreadCount * sizeof(uint64_t);
	totals_h = (uint64_t*)malloc(totalsSize);
	cudaMalloc(&totals_d, totalsSize);
	reduceCounts<<<1,reduceThreadCount>>>(pSums_d, totals_d, pSumSize, reduceSize);
	cudaMemcpy(totals_h, totals_d, totalsSize, cudaMemcpyDeviceToHost);

	// Free
	cudaFree(pSums_d); cudaFree(totals_d);

	// Accumulate elements in totals (reduced pSums)
	sum = 0;
	for (int i = 0; i < reduceThreadCount; i++) {
		sum += totals_h[i];
	}

	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "totals SUM = " <<sum << std::endl;
		std::cout << "total Samples = " << sampleSize*generateThreadCount  <<"\n";
	#endif

	// Approximate Pi
	approxPi = ((double)sum / sampleSize) / generateThreadCount;
	approxPi *= 4.0f;

	std::cout << "\n";
	return approxPi;
}
