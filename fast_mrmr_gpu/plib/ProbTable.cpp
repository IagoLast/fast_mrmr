/*
 * ProbTable.cpp
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#include "ProbTable.h"
#include "../cuda/cuda_histogram.h"

ProbTable::ProbTable(RawData & rd): rawData(rd) {
	this->datasize = rawData.getDataSize();
	this->featuresSize = rawData.getFeaturesSize();
	this->valuesRange = rawData.getValuesRangeArray();
	this->table = (t_prob_table) malloc(featuresSize * sizeof(t_prob *));
	calculate();
}

ProbTable::~ProbTable() {}

/**
 * Generates a table with all the marginal probabilities for each value for all features.
 *
 * This table is kept in memory so the marginal probability is computed only one time.
 */
void ProbTable::calculate() {
	Histogram histogram = Histogram(rawData);
	int i = 0;
	int j = 0;
	t_prob value = 0;
	t_histogram hist_data;
	for (i = 0; i < featuresSize; ++i) {
		table[i] = (t_prob*) malloc(valuesRange[i] * sizeof(t_prob));
		hist_data = histogram.getHistogram(i);
		for (j = 0; j < valuesRange[i]; ++j) {
			value = (t_prob) hist_data[j] / (t_prob) datasize;
			table[i][j] = value;
		}
	}
}

/**
 * @param featureIndex The index of the feature you want to get the probability.
 * @return The selected feature prob value.
 */
double ProbTable::getProbability(uint index, t_data value) {
	return table[index][value];
}

/**
 * Free prob table data.
 */
void ProbTable::destroy() {
	for (int i = 0; i < featuresSize; i++) {
		free(table[i]);
	}
	closeHistogram64();
	free(table);
}
