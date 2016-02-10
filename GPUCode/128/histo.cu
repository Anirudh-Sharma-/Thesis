#include <stdio.h>
#include "histo.h"

__global__ void histo_kernel(int histo_size, double reqNumThreads, int numPixPerThread, int numElemPerThread, int imgDataBufferSize, float *dev_imgDataBuffer, int nbBinsPerDim, int binSize, int *dev_histo){
    int stride = 0;
	int const part_histo_size = 12288;
	 int stride_offset = part_histo_size;
	 int histo_count = 1;

	__shared__ int temp_histo[part_histo_size];

	while(histo_count <= (((histo_size - 1)/part_histo_size)+1)){
    	temp_histo[threadIdx.x * 96] = 0;
    	temp_histo[threadIdx.x * 96 + 1] = 0;
    	temp_histo[threadIdx.x * 96 + 2] = 0;
    	temp_histo[threadIdx.x * 96 + 3] = 0;
    	temp_histo[threadIdx.x * 96 + 4] = 0;
    	temp_histo[threadIdx.x * 96 + 5] = 0;
    	temp_histo[threadIdx.x * 96 + 6] = 0;
    	temp_histo[threadIdx.x * 96 + 7] = 0;
    	temp_histo[threadIdx.x * 96 + 8] = 0;
    	temp_histo[threadIdx.x * 96 + 9] = 0;
    	temp_histo[threadIdx.x * 96 + 10] = 0;
    	temp_histo[threadIdx.x * 96 + 11] = 0;
    	temp_histo[threadIdx.x * 96 + 12] = 0;
    	temp_histo[threadIdx.x * 96 + 13] = 0;
    	temp_histo[threadIdx.x * 96 + 14] = 0;
    	temp_histo[threadIdx.x * 96 + 15] = 0;
    	temp_histo[threadIdx.x * 96 + 16] = 0;
    	temp_histo[threadIdx.x * 96 + 17] = 0;
    	temp_histo[threadIdx.x * 96 + 18] = 0;
    	temp_histo[threadIdx.x * 96 + 19] = 0;
    	temp_histo[threadIdx.x * 96 + 20] = 0;
    	temp_histo[threadIdx.x * 96 + 21] = 0;
    	temp_histo[threadIdx.x * 96 + 22] = 0;
    	temp_histo[threadIdx.x * 96 + 23] = 0;
    	temp_histo[threadIdx.x * 96 + 24] = 0;
    	temp_histo[threadIdx.x * 96 + 25] = 0;
    	temp_histo[threadIdx.x * 96 + 26] = 0;
    	temp_histo[threadIdx.x * 96 + 27] = 0;
    	temp_histo[threadIdx.x * 96 + 28] = 0;
    	temp_histo[threadIdx.x * 96 + 29] = 0;
    	temp_histo[threadIdx.x * 96 + 30] = 0;
    	temp_histo[threadIdx.x * 96 + 31] = 0;
    	temp_histo[threadIdx.x * 96 + 32] = 0;
    	temp_histo[threadIdx.x * 96 + 33] = 0;
    	temp_histo[threadIdx.x * 96 + 34] = 0;
    	temp_histo[threadIdx.x * 96 + 35] = 0;
    	temp_histo[threadIdx.x * 96 + 36] = 0;
    	temp_histo[threadIdx.x * 96 + 37] = 0;
    	temp_histo[threadIdx.x * 96 + 38] = 0;
    	temp_histo[threadIdx.x * 96 + 39] = 0;
    	temp_histo[threadIdx.x * 96 + 40] = 0;
    	temp_histo[threadIdx.x * 96 + 41] = 0;
    	temp_histo[threadIdx.x * 96 + 42] = 0;
    	temp_histo[threadIdx.x * 96 + 43] = 0;
    	temp_histo[threadIdx.x * 96 + 44] = 0;
    	temp_histo[threadIdx.x * 96 + 45] = 0;
    	temp_histo[threadIdx.x * 96 + 46] = 0;
    	temp_histo[threadIdx.x * 96 + 47] = 0;
    	temp_histo[threadIdx.x * 96 + 48] = 0;
    	temp_histo[threadIdx.x * 96 + 49] = 0;
    	temp_histo[threadIdx.x * 96 + 50] = 0;
    	temp_histo[threadIdx.x * 96 + 51] = 0;
    	temp_histo[threadIdx.x * 96 + 52] = 0;
    	temp_histo[threadIdx.x * 96 + 53] = 0;
    	temp_histo[threadIdx.x * 96 + 54] = 0;
    	temp_histo[threadIdx.x * 96 + 55] = 0;
    	temp_histo[threadIdx.x * 96 + 56] = 0;
    	temp_histo[threadIdx.x * 96 + 57] = 0;
    	temp_histo[threadIdx.x * 96 + 58] = 0;
    	temp_histo[threadIdx.x * 96 + 59] = 0;
    	temp_histo[threadIdx.x * 96 + 60] = 0;
    	temp_histo[threadIdx.x * 96 + 61] = 0;
    	temp_histo[threadIdx.x * 96 + 62] = 0;
    	temp_histo[threadIdx.x * 96 + 63] = 0;
    	temp_histo[threadIdx.x * 96 + 64] = 0;
    	temp_histo[threadIdx.x * 96 + 65] = 0;
    	temp_histo[threadIdx.x * 96 + 66] = 0;
    	temp_histo[threadIdx.x * 96 + 67] = 0;
    	temp_histo[threadIdx.x * 96 + 68] = 0;
    	temp_histo[threadIdx.x * 96 + 69] = 0;
    	temp_histo[threadIdx.x * 96 + 70] = 0;
    	temp_histo[threadIdx.x * 96 + 71] = 0;
    	temp_histo[threadIdx.x * 96 + 72] = 0;
    	temp_histo[threadIdx.x * 96 + 73] = 0;
    	temp_histo[threadIdx.x * 96 + 74] = 0;
    	temp_histo[threadIdx.x * 96 + 75] = 0;
    	temp_histo[threadIdx.x * 96 + 76] = 0;
    	temp_histo[threadIdx.x * 96 + 77] = 0;
    	temp_histo[threadIdx.x * 96 + 78] = 0;
    	temp_histo[threadIdx.x * 96 + 79] = 0;
    	temp_histo[threadIdx.x * 96 + 80] = 0;
    	temp_histo[threadIdx.x * 96 + 81] = 0;
    	temp_histo[threadIdx.x * 96 + 82] = 0;
    	temp_histo[threadIdx.x * 96 + 83] = 0;
    	temp_histo[threadIdx.x * 96 + 84] = 0;
    	temp_histo[threadIdx.x * 96 + 85] = 0;
    	temp_histo[threadIdx.x * 96 + 86] = 0;
    	temp_histo[threadIdx.x * 96 + 87] = 0;
    	temp_histo[threadIdx.x * 96 + 88] = 0;
    	temp_histo[threadIdx.x * 96 + 89] = 0;
    	temp_histo[threadIdx.x * 96 + 90] = 0;
    	temp_histo[threadIdx.x * 96 + 91] = 0;
    	temp_histo[threadIdx.x * 96 + 92] = 0;
    	temp_histo[threadIdx.x * 96 + 93] = 0;
    	temp_histo[threadIdx.x * 96 + 95] = 0;

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

    		atomicAdd(&dev_histo[(threadIdx.x * 96) + stride], temp_histo[threadIdx.x * 96 + 0]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 1) + stride], temp_histo[threadIdx.x * 96 + 1]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 2) + stride], temp_histo[threadIdx.x * 96 + 2]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 3) + stride], temp_histo[threadIdx.x * 96 + 3]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 4) + stride], temp_histo[threadIdx.x * 96 + 4]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 5) + stride], temp_histo[threadIdx.x * 96 + 5]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 6) + stride], temp_histo[threadIdx.x * 96 + 6]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 7) + stride], temp_histo[threadIdx.x * 96 + 7]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 8) + stride], temp_histo[threadIdx.x * 96 + 8]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 9) + stride], temp_histo[threadIdx.x * 96 + 9]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 10) + stride], temp_histo[threadIdx.x * 96 + 10]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 11) + stride], temp_histo[threadIdx.x * 96 + 11]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 12) + stride], temp_histo[threadIdx.x * 96 + 12]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 13) + stride], temp_histo[threadIdx.x * 96 + 13]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 14) + stride], temp_histo[threadIdx.x * 96 + 14]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 15) + stride], temp_histo[threadIdx.x * 96 + 15]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 16) + stride], temp_histo[threadIdx.x * 96 + 16]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 17) + stride], temp_histo[threadIdx.x * 96 + 17]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 18) + stride], temp_histo[threadIdx.x * 96 + 18]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 19) + stride], temp_histo[threadIdx.x * 96 + 19]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 20) + stride], temp_histo[threadIdx.x * 96 + 20]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 21) + stride], temp_histo[threadIdx.x * 96 + 21]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 22) + stride], temp_histo[threadIdx.x * 96 + 22]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 23) + stride], temp_histo[threadIdx.x * 96 + 23]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 24) + stride], temp_histo[threadIdx.x * 96 + 24]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 25) + stride], temp_histo[threadIdx.x * 96 + 25]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 26) + stride], temp_histo[threadIdx.x * 96 + 26]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 27) + stride], temp_histo[threadIdx.x * 96 + 27]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 28) + stride], temp_histo[threadIdx.x * 96 + 28]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 29) + stride], temp_histo[threadIdx.x * 96 + 29]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 30) + stride], temp_histo[threadIdx.x * 96 + 30]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 31) + stride], temp_histo[threadIdx.x * 96 + 31]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 32) + stride], temp_histo[threadIdx.x * 96 + 32]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 33) + stride], temp_histo[threadIdx.x * 96 + 33]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 34) + stride], temp_histo[threadIdx.x * 96 + 34]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 35) + stride], temp_histo[threadIdx.x * 96 + 35]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 36) + stride], temp_histo[threadIdx.x * 96 + 36]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 37) + stride], temp_histo[threadIdx.x * 96 + 37]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 38) + stride], temp_histo[threadIdx.x * 96 + 38]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 39) + stride], temp_histo[threadIdx.x * 96 + 39]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 40) + stride], temp_histo[threadIdx.x * 96 + 40]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 41) + stride], temp_histo[threadIdx.x * 96 + 41]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 42) + stride], temp_histo[threadIdx.x * 96 + 42]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 43) + stride], temp_histo[threadIdx.x * 96 + 43]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 44) + stride], temp_histo[threadIdx.x * 96 + 44]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 45) + stride], temp_histo[threadIdx.x * 96 + 45]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 46) + stride], temp_histo[threadIdx.x * 96 + 46]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 47) + stride], temp_histo[threadIdx.x * 96 + 47]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 48) + stride], temp_histo[threadIdx.x * 96 + 48]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 49) + stride], temp_histo[threadIdx.x * 96 + 49]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 50) + stride], temp_histo[threadIdx.x * 96 + 50]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 51) + stride], temp_histo[threadIdx.x * 96 + 51]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 52) + stride], temp_histo[threadIdx.x * 96 + 52]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 53) + stride], temp_histo[threadIdx.x * 96 + 53]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 54) + stride], temp_histo[threadIdx.x * 96 + 54]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 55) + stride], temp_histo[threadIdx.x * 96 + 55]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 56) + stride], temp_histo[threadIdx.x * 96 + 56]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 57) + stride], temp_histo[threadIdx.x * 96 + 57]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 58) + stride], temp_histo[threadIdx.x * 96 + 58]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 59) + stride], temp_histo[threadIdx.x * 96 + 59]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 60) + stride], temp_histo[threadIdx.x * 96 + 60]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 61) + stride], temp_histo[threadIdx.x * 96 + 61]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 62) + stride], temp_histo[threadIdx.x * 96 + 62]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 63) + stride], temp_histo[threadIdx.x * 96 + 63]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 64) + stride], temp_histo[threadIdx.x * 96 + 64]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 65) + stride], temp_histo[threadIdx.x * 96 + 65]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 66) + stride], temp_histo[threadIdx.x * 96 + 66]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 67) + stride], temp_histo[threadIdx.x * 96 + 67]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 68) + stride], temp_histo[threadIdx.x * 96 + 68]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 69) + stride], temp_histo[threadIdx.x * 96 + 69]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 70) + stride], temp_histo[threadIdx.x * 96 + 70]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 71) + stride], temp_histo[threadIdx.x * 96 + 71]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 72) + stride], temp_histo[threadIdx.x * 96 + 72]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 73) + stride], temp_histo[threadIdx.x * 96 + 73]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 74) + stride], temp_histo[threadIdx.x * 96 + 74]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 75) + stride], temp_histo[threadIdx.x * 96 + 75]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 76) + stride], temp_histo[threadIdx.x * 96 + 76]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 77) + stride], temp_histo[threadIdx.x * 96 + 77]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 78) + stride], temp_histo[threadIdx.x * 96 + 78]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 79) + stride], temp_histo[threadIdx.x * 96 + 79]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 80) + stride], temp_histo[threadIdx.x * 96 + 80]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 81) + stride], temp_histo[threadIdx.x * 96 + 81]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 82) + stride], temp_histo[threadIdx.x * 96 + 82]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 83) + stride], temp_histo[threadIdx.x * 96 + 83]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 84) + stride], temp_histo[threadIdx.x * 96 + 84]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 85) + stride], temp_histo[threadIdx.x * 96 + 85]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 86) + stride], temp_histo[threadIdx.x * 96 + 86]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 87) + stride], temp_histo[threadIdx.x * 96 + 87]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 88) + stride], temp_histo[threadIdx.x * 96 + 88]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 89) + stride], temp_histo[threadIdx.x * 96 + 89]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 90) + stride], temp_histo[threadIdx.x * 96 + 90]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 91) + stride], temp_histo[threadIdx.x * 96 + 91]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 92) + stride], temp_histo[threadIdx.x * 96 + 92]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 93) + stride], temp_histo[threadIdx.x * 96 + 93]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 94) + stride], temp_histo[threadIdx.x * 96 + 94]);
    		atomicAdd(&dev_histo[(threadIdx.x * 96 + 95) + stride], temp_histo[threadIdx.x * 96 + 95]);

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
histo_kernel<<<26, 128>>>(histo_size, reqNumThreads, numPixPerThread, numElemPerThread, imgDataBufferSize, dev_imgDataBuffer, nbBinsPerDim, binSize, dev_histo);
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
