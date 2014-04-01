/** @file Rawdata.cpp
 *  @brief Used to handle the raw csv data.
 *
 *  Contains the RawData class and defines the basic
 *  datatypes for the project.
 *
 *  @author Iago Lastar (iagolast)
 */
#include "RawData.h"
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
/**
 * Constructor that creates a rawData object.
 *
 * @param data_table this is a matrix of bytes containing the translated csv data.
 * @param ds the number of data samples.
 * @param fs the number of features that each sample has.
 */
RawData::RawData() {
	dataFile = fopen("data.mrmr", "rb");
	calculateDSandFS();
	loadData();
	//FIXME: Descomentar para GPU.
	//mallocGPU();
	//moveGPU();
	calculateVR();
}

RawData::~RawData() {

}

void RawData::freeGPU(){
	cudaFree(d_data);
}

/**
 *
 */
void RawData::destroy() {
	//FIXME: Descomentar para GPU.
	//freeGPU();
	free(valuesRange);
	free(h_data);

}

void RawData::calculateDSandFS() {
	uint featuresSizeBuffer[1];
	uint datasizeBuffer[1];
	fread(datasizeBuffer, sizeof(uint), 1, dataFile);
	fread(featuresSizeBuffer, sizeof(uint), 1, dataFile);
	datasize = datasizeBuffer[0];
	featuresSize = featuresSizeBuffer[0];
}

void RawData::loadData() {
	uint i, j;
	t_data buffer[1];
	//	Reservo espacio para SIZE punteros
	h_data = (t_data*) calloc(featuresSize, sizeof(t_data) * datasize);
	fseek(dataFile, 8, 0);
	for (i = 0; i < datasize; i++) {
		for (j = 0; j < featuresSize; j++) {
			fread(buffer, sizeof(t_data), 1, dataFile);
			h_data[j * datasize + i] = buffer[0];
		}
	}
}

/**
 * Allocs space to keep all data in GPU, if error  program ends.
 */
void RawData::mallocGPU() {
	cudaMalloc((void**) &d_data, datasize * featuresSize * sizeof(t_data));
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("Error allocating data in GPU: %d", err);
		exit(-1);
	}
}

/**
 * Moves the data from host to device, if error  program ends.
 */
void RawData::moveGPU() {
	cudaMemcpy(d_data, h_data, datasize * featuresSize * sizeof(t_data),
			cudaMemcpyHostToDevice);
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("Error allocating data in GPU: %d", err);
		exit(-1);
	}
}

/**
 * Calculates how many different values has each feature.
 */
void RawData::calculateVR() {
	uint i, j;
	t_data dataReaded;
	uint vr;
	valuesRange = (uint*) calloc(featuresSize, sizeof(uint));
	for (i = 0; i < featuresSize; i++) {
		vr = 0;
		for (j = 0; j < datasize; j++) {
			dataReaded = h_data[i * datasize + j];
			if (dataReaded > vr) {
				vr++;
			}
		}
		valuesRange[i] = vr + 1;
	}
}

/**
 *
 */
uint RawData::getDataSize() {
	return datasize;
}

/**
 *
 */
uint RawData::getFeaturesSize() {
	return featuresSize;
}

/**
 * Returns how much values has a features FROM 1 to VALUES;
 */
uint RawData::getValuesRange(uint index) {
	return valuesRange[index];
}

/**
 *
 */
uint * RawData::getValuesRangeArray() {
	return this->valuesRange;
}

/**
 * Returns a vector containing a feature.
 */
t_feature RawData::getFeature(int index) {
	return h_data + index * datasize;
}

/**
 * Returns the GPU vector that contains the feature.
 */
t_feature RawData::getFeatureGPU(int index){
	return d_data + index * datasize;
}
