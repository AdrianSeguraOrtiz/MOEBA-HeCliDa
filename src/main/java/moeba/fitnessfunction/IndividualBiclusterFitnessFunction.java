package moeba.fitnessfunction;

import java.util.ArrayList;

import moeba.ColumnType;
import moeba.utils.storage.CacheStorage;

public abstract class IndividualBiclusterFitnessFunction extends BiclusterFitnessFunction {

    public IndividualBiclusterFitnessFunction(double[][] data, ColumnType[] types, CacheStorage<String, Double> internalCache,
            String summariseIndividualObjectives) {
        super(data, types, internalCache, summariseIndividualObjectives);
    }

    protected double getBiclusterScore(ArrayList<ArrayList<Integer>[]> biclusters, int i) {
        return this.getBiclusterScore(biclusters.get(i));
    }

    protected abstract double getBiclusterScore(ArrayList<Integer>[] bicluster);
}
