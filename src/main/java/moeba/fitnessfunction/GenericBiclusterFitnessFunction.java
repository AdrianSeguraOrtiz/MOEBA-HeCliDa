package moeba.fitnessfunction;

import java.util.ArrayList;

import moeba.ColumnType;
import moeba.utils.storage.CacheStorage;

public abstract class GenericBiclusterFitnessFunction extends BiclusterFitnessFunction {

    public GenericBiclusterFitnessFunction(double[][] data, ColumnType[] types,
            CacheStorage<String, Double> internalCache, String summariseIndividualObjectives) {
        super(data, types, internalCache, summariseIndividualObjectives);
    }

    protected abstract double getBiclusterScore(ArrayList<Integer>[] bicluster, ArrayList<ArrayList<Integer>[]> biclusters);

    protected double getBiclusterScore(ArrayList<ArrayList<Integer>[]> biclusters, int i) {
        if (biclusters.size() == 1) return 0.0;
        ArrayList<ArrayList<Integer>[]> biclustersCopy = new ArrayList<>(biclusters);
        biclustersCopy.remove(i);
        return this.getBiclusterScore(biclusters.get(i), biclustersCopy);
    }
    
}
