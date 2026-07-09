package moeba.fitnessfunction.impl;

import java.util.ArrayList;

import moeba.ColumnType;
import moeba.fitnessfunction.IndividualBiclusterFitnessFunction;
import moeba.utils.storage.CacheStorage;

public class BiclusterVarianceNorm extends IndividualBiclusterFitnessFunction {

    public BiclusterVarianceNorm(double[][] data, ColumnType[] types, CacheStorage<String, Double> internalCache, String summariseIndividualObjectives) {
        super(data, types, internalCache, summariseIndividualObjectives);
    }

    @Override
    protected double getBiclusterScore(ArrayList<Integer>[] bicluster) {  
        // Get the mean of the bicluster
        float sum = 0.0f;
        for (int i : bicluster[0]) {
            for (int j : bicluster[1]) {
                sum += data[i][j];
            }
        }
        float mean = sum / (bicluster[0].size() * bicluster[1].size());

        // Sum the squared differences of the bicluster values from the mean
        double score = 0.0;
        for (int i : bicluster[0]) {
            for (int j : bicluster[1]) {
                score += Math.pow(data[i][j] - mean, 2);
            }
        }

        // Revert to maximization and normalize between 0 and 1
        return 1 - 4 * score / (data.length * data[0].length);
    }
}
