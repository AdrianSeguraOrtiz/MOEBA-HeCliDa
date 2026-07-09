package moeba.fitnessfunction.impl;

import java.util.ArrayList;

import moeba.ColumnType;
import moeba.fitnessfunction.IndividualBiclusterFitnessFunction;
import moeba.utils.storage.CacheStorage;

public class BiclusterSizeNormComp extends IndividualBiclusterFitnessFunction {
    private double rowsWeight;
    private double colsWeight;

    public BiclusterSizeNormComp(double[][] data, ColumnType[] types, CacheStorage<String, Double> internalCache, String summariseIndividualObjectives, double rowsWeight) {
        super(data, types, internalCache, summariseIndividualObjectives);
        this.rowsWeight = rowsWeight;
        this.colsWeight = 1 - rowsWeight;
    }

    public double getNormalizedSize(ArrayList<Integer>[] bicluster) {
        return this.rowsWeight * ((double) bicluster[0].size() / data.length) + this.colsWeight * ((double) bicluster[1].size() / data[0].length);
    }

    protected double getBiclusterScore(ArrayList<Integer>[] bicluster) {
        return getNormalizedSize(bicluster);
    }
}
