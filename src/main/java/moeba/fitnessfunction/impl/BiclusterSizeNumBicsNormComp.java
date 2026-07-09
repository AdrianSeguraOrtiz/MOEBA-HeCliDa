package moeba.fitnessfunction.impl;

import java.util.ArrayList;

import moeba.ColumnType;
import moeba.fitnessfunction.GenericBiclusterFitnessFunction;
import moeba.utils.storage.CacheStorage;

public class BiclusterSizeNumBicsNormComp extends GenericBiclusterFitnessFunction {
    private BiclusterSizeNormComp biclusterSizeNormComp;
    private double coherenceWeight;

    public BiclusterSizeNumBicsNormComp(double[][] data, ColumnType[] types, CacheStorage<String, Double> internalCache,
            String summariseIndividualObjectives, double rowsWeight, double coherenceWeight) {
        super(data, types, internalCache, summariseIndividualObjectives);
        this.biclusterSizeNormComp = new BiclusterSizeNormComp(data, types, internalCache, summariseIndividualObjectives, rowsWeight);
        this.coherenceWeight = coherenceWeight;
    }

    @Override
    protected double getBiclusterScore(ArrayList<Integer>[] bicluster, ArrayList<ArrayList<Integer>[]> biclusters) {
        int maxSize = data.length * data[0].length;
        double parcelSize = (double) maxSize / Math.pow(biclusters.size() + 1, 2);
        int biclusterSize = bicluster[0].size() * bicluster[1].size();

        return (1 - coherenceWeight) * biclusterSizeNormComp.getNormalizedSize(bicluster) + coherenceWeight * (1 - Math.min(1, Math.abs(parcelSize - biclusterSize) / parcelSize));
    }
    
}
