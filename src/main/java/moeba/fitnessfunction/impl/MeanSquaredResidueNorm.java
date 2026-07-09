package moeba.fitnessfunction.impl;

import java.util.ArrayList;
import java.util.Arrays;

import moeba.ColumnType;
import moeba.fitnessfunction.IndividualBiclusterFitnessFunction;
import moeba.utils.storage.CacheStorage;

public class MeanSquaredResidueNorm extends IndividualBiclusterFitnessFunction {

    public MeanSquaredResidueNorm(double[][] data, ColumnType[] types, CacheStorage<String, Double> internalCache, String summariseIndividualObjectives) {
        super(data, types, internalCache, summariseIndividualObjectives);
    }

    @Override
    protected double getBiclusterScore(ArrayList<Integer>[] bicluster) {  
        // Get the mean of each row, each column, and the entire bicluster
        int numRows = bicluster[0].size();
        int numCols = bicluster[1].size();
        float[] rowMeans = new float[numRows];
        float[] colMeans = new float[numCols];
        Arrays.fill(colMeans, 0.0f);
        float totalSum = 0.0f;

        for (int i = 0; i < numRows; i++) {
            float rowSum = 0.0f;
            for (int j = 0; j < numCols; j++) {
                double value = data[bicluster[0].get(i)][bicluster[1].get(j)];
                rowSum += value;
                colMeans[j] += value;
            }
            rowMeans[i] = rowSum / numCols;
            totalSum += rowSum;
        }

        float mean = totalSum / (numRows * numCols);
        for (int j = 0; j < numCols; j++) {
            colMeans[j] /= numRows;
        }

        // Calculate Mean Squared Residue
        double score = 0.0;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double value = data[bicluster[0].get(i)][bicluster[1].get(j)];
                score += Math.pow(value - rowMeans[i] - colMeans[j] + mean, 2);
            }
        }

        // Divide by the number of elements in the bicluster
        score /= (bicluster[0].size() * bicluster[1].size());

        // Revert to maximization and normalize between 0 and 1
        return 1 - score / 4;
    }
}
