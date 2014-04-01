/*
 * Histogram.cpp
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#include "Histogram.h"

Histogram::Histogram(RawData & rd) :
		rawData(rd) {

}

Histogram::~Histogram() {
}

t_histogram Histogram::getHistogram(uint index) {
	//TODO: Ajustar parametros gpu, limpiar codigo kernels y limpiar imports GPU.
	uint vr = rawData.getValuesRange(index);
	t_feature h_vector = rawData.getFeature(index); //FIXME: no hace falta en gpu version.
	t_histogram d_acum;
	t_histogram h_acum = (t_histogram) calloc(vr, sizeof(uint));

	//CUDA STUFF
	cudaMalloc((void**) &d_acum, vr * sizeof(uint));
	t_feature d_vector = rawData.getFeatureGPU(index);

	if (vr <= 64) {

		histogram64(d_acum,d_vector,DATASIZE);
	} else {
		histogramNaive(d_vector, d_acum, DATASIZE, blocks);

	}
	/* CPU histogram.
	 for (uint i = 0; i < rawData.getDataSize(); i++) {
	 h_acum[data[i]]++;
	 }
	 //*/
	cudaMemcpy(h_acum, d_acum, vr * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaFree(d_acum);
	return h_acum;
}
