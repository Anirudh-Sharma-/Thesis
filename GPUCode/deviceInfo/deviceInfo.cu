#include <stdio.h>
#include <iostream>


int main( void ) {
cudaDeviceProp prop;
FILE *deviceInfoOutput;
deviceInfoOutput = fopen("deviceInfoOutput.txt", "a");
int count;
cudaGetDeviceCount( &count );
printf("Devices information\n");
fprintf(deviceInfoOutput, "Devices information\n");
printf("Total Cuda devices: %d\n", count);
fprintf(deviceInfoOutput, "Total Cuda devices: %d\n", count);
for (int i=0; i< count; i++) {
	cudaGetDeviceProperties( &prop, i );
//Do something with our device's properties
	printf("----General information for device %d----\n",i);
	fprintf(deviceInfoOutput, "----General information for device %d----\n", i);
	printf("Name:  %s\n", prop.name);
	fprintf(deviceInfoOutput, "Name:  %s\n", prop.name);
	printf("Compute Capability:  %d.%d\n", prop.major, prop.minor);
	fprintf(deviceInfoOutput, "Compute Capability:  %d.%d\n", prop.major, prop.minor);
	printf("Clock Rate:  %d\n", prop.clockRate);
	fprintf(deviceInfoOutput, "Clock Rate:  %d\n", prop.clockRate);
	printf("Device Copy Overlap: ");
	fprintf(deviceInfoOutput, "Device Copy Overlap: ");
	if(prop.deviceOverlap){
		printf("Enabled\n");
		fprintf(deviceInfoOutput, "Enabled\n");
	}
	else{
		printf("Disabled\n");
		fprintf(deviceInfoOutput, "Disabled\n");
	}
	printf("Kernel execution timeout:  ");
	fprintf(deviceInfoOutput, "Kernel execution timeout:  ");
	if (prop.kernelExecTimeoutEnabled){
		printf( "Enabled\n" );
		fprintf(deviceInfoOutput, "Enabled\n");
	}
	else{
		printf( "Disabled\n" );
		fprintf(deviceInfoOutput, "Disabled\n");
	}
	printf( "--- Memory Information for device %d ---\n", i );
	fprintf(deviceInfoOutput, "--- Memory Information for device %d ---\n", i );
	printf(" Total global mem:  %ld\n", prop.totalGlobalMem);
	fprintf(deviceInfoOutput, " Total global mem:  %ld\n", prop.totalGlobalMem);
	printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
	fprintf(deviceInfoOutput, "Total constant Mem:  %ld\n", prop.totalConstMem );
	printf( "Max mem pitch: %ld\n", prop.memPitch );
	fprintf(deviceInfoOutput, "Max mem pitch: %ld\n", prop.memPitch );
	printf( "Texture Alignment:  %ld\n", prop.textureAlignment );
	fprintf(deviceInfoOutput, "Texture Alignment:  %ld\n", prop.textureAlignment );
	printf( "--- MP Information for device %d ---\n", i );
	fprintf(deviceInfoOutput, "--- MP Information for device %d ---\n", i );
	printf( "Multiprocessor count: %d\n", prop.multiProcessorCount );
	fprintf(deviceInfoOutput, "Multiprocessor count: %d\n", prop.multiProcessorCount );
	printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
	fprintf(deviceInfoOutput, "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
	printf( "Registers per mp: %d\n", prop.regsPerBlock );
	fprintf(deviceInfoOutput, "Registers per mp: %d\n", prop.regsPerBlock );
	printf( "Threads in warp: %d\n", prop.warpSize );
	fprintf(deviceInfoOutput, "Threads in warp: %d\n", prop.warpSize );
	printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
	fprintf(deviceInfoOutput, "Max threads per block: %d\n", prop.maxThreadsPerBlock  );
	printf( "Max thread dimensions: (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1],prop.maxThreadsDim[2] );
	fprintf(deviceInfoOutput, "Max thread dimensions: (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1],prop.maxThreadsDim[2] );
	printf( "Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1],prop.maxGridSize[2] );
	fprintf(deviceInfoOutput, "Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1],prop.maxGridSize[2]);
	printf( "\n" );
	fprintf(deviceInfoOutput, "\n" );
}
fclose(deviceInfoOutput);
}
