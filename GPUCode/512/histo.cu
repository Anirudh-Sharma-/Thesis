#include <stdio.h>
#include "histo.h"

__global__ void histo_kernel(int histo_size, double reqNumThreads, int numPixPerThread, int numElemPerThread, int imgDataBufferSize, float *dev_imgDataBuffer, int nbBinsPerDim, int binSize, int *dev_histo){
    int stride = 0;
	int const part_histo_size = 12288;
	 int stride_offset = part_histo_size;
	 int histo_count = 1;

	__shared__ int temp_histo[part_histo_size];

	while(histo_count <= (((histo_size - 1)/part_histo_size)+1)){
    	temp_histo[threadIdx.x * 24] = 0;
    	temp_histo[threadIdx.x * 24 + 1] = 0;
    	temp_histo[threadIdx.x * 24 + 2] = 0;
    	temp_histo[threadIdx.x * 24 + 3] = 0;
    	temp_histo[threadIdx.x * 24 + 4] = 0;
    	temp_histo[threadIdx.x * 24 + 5] = 0;
    	temp_histo[threadIdx.x * 24 + 6] = 0;
    	temp_histo[threadIdx.x * 24 + 7] = 0;
    	temp_histo[threadIdx.x * 24 + 8] = 0;
    	temp_histo[threadIdx.x * 24 + 9] = 0;
    	temp_histo[threadIdx.x * 24 + 10] = 0;
    	temp_histo[threadIdx.x * 24 + 11] = 0;
    	temp_histo[threadIdx.x * 24 + 12] = 0;
    	temp_histo[threadIdx.x * 24 + 13] = 0;
    	temp_histo[threadIdx.x * 24 + 14] = 0;
    	temp_histo[threadIdx.x * 24 + 15] = 0;
    	temp_histo[threadIdx.x * 24 + 16] = 0;
    	temp_histo[threadIdx.x * 24 + 17] = 0;
    	temp_histo[threadIdx.x * 24 + 18] = 0;
    	temp_histo[threadIdx.x * 24 + 19] = 0;
    	temp_histo[threadIdx.x * 24 + 20] = 0;
    	temp_histo[threadIdx.x * 24 + 21] = 0;
    	temp_histo[threadIdx.x * 24 + 22] = 0;
    	temp_histo[threadIdx.x * 24 + 23] = 0;


    	__syncthreads();

    	int i = threadIdx.x + blockIdx.x * blockDim.x;
    	int offset = blockDim.x * gridDim.x;
    	while(i < reqNumThreads){
    	//	int j = 0;
    	//	while((j < numPixPerThread) && (((i * numElemPerThread)+(j * 3)) < imgDataBufferSize)){
    			//printf("Inside while\n");
    			float L = dev_imgDataBuffer[(i * numElemPerThread)];
    			float a = dev_imgDataBuffer[(i * numElemPerThread + 1)];
    			float b = dev_imgDataBuffer[(i * numElemPerThread + 2)];
    			int idx = ((((int)round(L)-0)/binSize)*nbBinsPerDim*nbBinsPerDim)+
    					(((int)round(a)+127)/binSize)*nbBinsPerDim +
    					((int)round(b)+127)/binSize;
    			/**checking if the idx lies between the current range of histogram**/
    			if(idx >= stride && idx < stride_offset){
    				int finalIdx = idx - stride;
    				atomicAdd(&temp_histo[finalIdx],1);
    			}//end of if condition
    		//	j++;
    		//}//end of while loop calculating number of n values per thread
    		i += offset;
    	}//end of image scan while condition
    	__syncthreads();

    		atomicAdd(&dev_histo[(threadIdx.x * 24) + stride], temp_histo[threadIdx.x * 24 + 0]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 1) + stride], temp_histo[threadIdx.x * 24 + 1]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 2) + stride], temp_histo[threadIdx.x * 24 + 2]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 3) + stride], temp_histo[threadIdx.x * 24 + 3]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 4) + stride], temp_histo[threadIdx.x * 24 + 4]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 5) + stride], temp_histo[threadIdx.x * 24 + 5]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 6) + stride], temp_histo[threadIdx.x * 24 + 6]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 7) + stride], temp_histo[threadIdx.x * 24 + 7]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 8) + stride], temp_histo[threadIdx.x * 24 + 8]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 9) + stride], temp_histo[threadIdx.x * 24 + 9]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 10) + stride], temp_histo[threadIdx.x * 24 + 10]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 11) + stride], temp_histo[threadIdx.x * 24 + 11]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 12) + stride], temp_histo[threadIdx.x * 24 + 12]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 13) + stride], temp_histo[threadIdx.x * 24 + 13]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 14) + stride], temp_histo[threadIdx.x * 24 + 14]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 15) + stride], temp_histo[threadIdx.x * 24 + 15]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 16) + stride], temp_histo[threadIdx.x * 24 + 16]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 17) + stride], temp_histo[threadIdx.x * 24 + 17]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 18) + stride], temp_histo[threadIdx.x * 24 + 18]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 19) + stride], temp_histo[threadIdx.x * 24 + 19]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 20) + stride], temp_histo[threadIdx.x * 24 + 20]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 21) + stride], temp_histo[threadIdx.x * 24 + 21]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 22) + stride], temp_histo[threadIdx.x * 24 + 22]);
    		atomicAdd(&dev_histo[(threadIdx.x * 24 + 23) + stride], temp_histo[threadIdx.x * 24 + 23]);

    	stride_offset += part_histo_size;
    	stride += part_histo_size;
    	histo_count++;
	}/** end of histo_count while condition **/
}/** end of kernel function **/


void createHisto(float *imgDataBuffer, int imgDataBufferSize, int nbBinsPerDim, int imgSize, int binSize, int *count){
cudaEvent_t hostDeviceStart,hostDeviceStop;
cudaEventCreate( &hostDeviceStart );
cudaEventCreate( &hostDeviceStop );
cudaEventRecord( hostDeviceStart, 0 );
float *dev_imgDataBuffer;
cudaError_t errorOne = cudaMalloc((void**)&dev_imgDataBuffer, imgDataBufferSize*sizeof(float));
if(errorOne != cudaSuccess)
{
  printf("CUDA errorOne: %s\n", cudaGetErrorString(errorOne));
}
cudaMemcpy(dev_imgDataBuffer, imgDataBuffer, imgDataBufferSize*sizeof(float), cudaMemcpyHostToDevice);

int *dev_histo;
cudaMalloc((void**)&dev_histo, nbBinsPerDim*nbBinsPerDim*nbBinsPerDim*sizeof(int));

cudaMemset(dev_histo, 0, nbBinsPerDim*nbBinsPerDim*nbBinsPerDim*sizeof(int));
cudaEventRecord( hostDeviceStop, 0 );
cudaEventSynchronize( hostDeviceStop );
float hostDeviceElapsedTime;
cudaEventElapsedTime( &hostDeviceElapsedTime,hostDeviceStart, hostDeviceStop );
printf( "\nTime to transfer host to device:  %3.1f ms\n", hostDeviceElapsedTime );

int numPixPerThread = 1;
int numElemPerThread = numPixPerThread*3;
int histo_size = nbBinsPerDim*nbBinsPerDim*nbBinsPerDim;
double reqNumThreads = (((imgSize - 1)/numPixPerThread)+1);
size_t printBufferSize = 1048576*100;
cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printBufferSize);
cudaEvent_t start,stop;
cudaEventCreate( &start );
cudaEventCreate( &stop );
cudaEventRecord( start, 0 );
histo_kernel<<<26, 512>>>(histo_size, reqNumThreads, numPixPerThread, numElemPerThread, imgDataBufferSize, dev_imgDataBuffer, nbBinsPerDim, binSize, dev_histo);
cudaEventRecord( stop, 0 );
cudaEventSynchronize( stop );
float elapsedTime;
cudaEventElapsedTime( &elapsedTime,start, stop );
printf( "Time to generate:  %3.1f ms\n", elapsedTime );
cudaThreadSynchronize();
cudaError_t error = cudaGetLastError();
if(error != cudaSuccess)
{
  // print the CUDA error message and exit
  printf("CUDA error: %s\n", cudaGetErrorString(error));
  printf("I am inside");
}
//int histo[nbBinsPerDim*nbBinsPerDim*nbBinsPerDim];
cudaEvent_t deviceToHostStart,deviceToHostStop;
cudaEventCreate( &deviceToHostStart );
cudaEventCreate( &deviceToHostStop );
cudaEventRecord( deviceToHostStart, 0 );
cudaMemcpy(count, dev_histo, nbBinsPerDim * nbBinsPerDim * nbBinsPerDim*sizeof(int), cudaMemcpyDeviceToHost);
cudaEventRecord( deviceToHostStop, 0 );
cudaEventSynchronize( deviceToHostStop );
float deviceToHostElapsedTime;
cudaEventElapsedTime( &deviceToHostElapsedTime,deviceToHostStart, deviceToHostStop );
printf( "\nTime to transfer device to host:  %3.1f ms\n", deviceToHostElapsedTime );
cudaFree(dev_histo);
cudaFree(dev_imgDataBuffer);
//return histo;
FILE *histoNonZeroCU;
histoNonZeroCU = fopen("histoNonZeroCU.txt", "a");
for(int i = 0; i < nbBinsPerDim * nbBinsPerDim * nbBinsPerDim; i++){
	if(count[i] != 0)
	fprintf(histoNonZeroCU, "%d) %d \n",i+1, count[i]);
}
fclose(histoNonZeroCU);


}
