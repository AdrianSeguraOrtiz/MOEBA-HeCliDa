package moeba.fitnessfunction.impl;

import java.util.ArrayList;

import moeba.ColumnType;
import moeba.fitnessfunction.IndividualBiclusterFitnessFunction;
import moeba.utils.storage.CacheStorage;

public class RowVarianceNormComp extends IndividualBiclusterFitnessFunction {

    public RowVarianceNormComp(double[][] data, ColumnType[] types, CacheStorage<String, Double> internalCache, String summariseIndividualObjectives) {
        super(data, types, internalCache, summariseIndividualObjectives);
    }

    @Override
    protected double getBiclusterScore(ArrayList<Integer>[] bicluster) {  
        // Get the mean of each row in the bicluster
        float[] means = new float[bicluster[0].size()];
        int cnt = 0;
        for (int i : bicluster[0]) {
            float sum = 0.0f;
            for (int j : bicluster[1]) {
                sum += data[i][j];
            }
            means[cnt++] = sum / bicluster[1].size();
        }

        // Sum the squared differences of the bicluster values from the mean for each row
        double score = 0.0;
        cnt = 0;
        for (int i : bicluster[0]) {
            for (int j : bicluster[1]) {
                score += Math.pow(data[i][j] - means[cnt], 2);
            }
            cnt++;
        }

        // Divide by the number of elements in the bicluster
        score /= (bicluster[0].size() * bicluster[1].size());

        // Normalize between 0 and 1
        return 4 * score;
    }
}
