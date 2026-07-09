package moeba.fitnessfunction;

import java.util.ArrayList;

import moeba.ColumnType;

public abstract class FitnessFunction {
    protected double[][] data;
    protected ColumnType[] types;
    protected RunnableFunc func;

    public interface RunnableFunc {
        double run(ArrayList<ArrayList<Integer>[]> biclusters);
    }

    public FitnessFunction(double[][] data, ColumnType[] types) {
        this.data = data;
        this.types = types;
    }

    public abstract double run(ArrayList<ArrayList<Integer>[]> biclusters);

    public ArrayList<ArrayList<Integer>[]> deleteSmallOrEmptyBiclusters(ArrayList<ArrayList<Integer>[]> biclusters) {
        biclusters.removeIf(b -> b[0].size() <= 1 || b[1].size() <= 1);
        return biclusters;
    }
    
}
