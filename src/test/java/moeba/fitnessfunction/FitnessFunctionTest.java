package moeba.fitnessfunction;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;

import org.junit.jupiter.api.Test;
import moeba.ColumnType;
import moeba.fitnessfunction.impl.BiclusterSizeNumBicsNormComp;
import moeba.fitnessfunction.impl.BiclusterSizeNormComp;
import moeba.fitnessfunction.impl.BiclusterVarianceNorm;
import moeba.fitnessfunction.impl.DistanceBetweenBiclustersNormComp;
import moeba.fitnessfunction.impl.MeanSquaredResidueNorm;
import moeba.fitnessfunction.impl.RowVarianceNormComp;

public class FitnessFunctionTest {

    final double[][] data = new double[][] {
        {0.5, 0.2, 0.7, 0.3},
        {0.1, 0.4, 0.6, 0.8},
        {0.7, 0.8, 0.5, 0.1},
        {0.2, 0.3, 0.8, 0.6}
    };
    final ColumnType[] types = new ColumnType[] {
        ColumnType.numeric(),
        ColumnType.numeric(),
        ColumnType.numeric(),
        ColumnType.numeric()
    };
    final double epsilon = 0.01;
    
    @SuppressWarnings("unchecked")
    @Test
    public void testBiclusterSizeNormCompOneBicluster() {
        FitnessFunction f = new BiclusterSizeNormComp(data, types, null, "Mean", 0.5);

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b = new ArrayList[]{new ArrayList<>(Arrays.asList(1, 2)), new ArrayList<>(Arrays.asList(0, 1))};
        biclusters.add(b);

        double expected = 1.0 - (0.5*(2.0/4.0) + 0.5*(2.0/4.0));
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testBiclusterSizeNormCompTwoBiclustersMean() {
        FitnessFunction f = new BiclusterSizeNormComp(data, types, null, "Mean", 0.5);

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b1 = new ArrayList[]{new ArrayList<>(Arrays.asList(1, 2)), new ArrayList<>(Arrays.asList(0, 1))};
        ArrayList<Integer>[] b2 = new ArrayList[]{new ArrayList<>(Arrays.asList(0, 2, 3)), new ArrayList<>(Arrays.asList(1, 3))};
        biclusters.add(b1);
        biclusters.add(b2);

        double expected = 1.0 - (
            (0.5*(2.0/4.0) + 0.5*(2.0/4.0)) + (0.5*(3.0/4.0) + 0.5*(2.0/4.0))
        ) / 2;
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testBiclusterSizeNormCompTwoBiclustersHarmonicMean() {
        FitnessFunction f = new BiclusterSizeNormComp(data, types, null, "HarmonicMean", 0.5);

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b1 = new ArrayList[]{new ArrayList<>(Arrays.asList(1, 2)), new ArrayList<>(Arrays.asList(0, 1))};
        ArrayList<Integer>[] b2 = new ArrayList[]{new ArrayList<>(Arrays.asList(0, 2, 3)), new ArrayList<>(Arrays.asList(1, 3))};
        biclusters.add(b1);
        biclusters.add(b2);

        double expected = 1.0 - 2.0 / (
            1.0 / (0.5*(2.0/4.0) + 0.5*(2.0/4.0)) +
            1.0 / (0.5*(3.0/4.0) + 0.5*(2.0/4.0))
        );
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testBiclusterSizeNormCompTwoBiclustersGeometricMean() {
        FitnessFunction f = new BiclusterSizeNormComp(data, types, null, "GeometricMean", 0.5);

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b1 = new ArrayList[]{new ArrayList<>(Arrays.asList(1, 2)), new ArrayList<>(Arrays.asList(0, 1))};
        ArrayList<Integer>[] b2 = new ArrayList[]{new ArrayList<>(Arrays.asList(0, 2, 3)), new ArrayList<>(Arrays.asList(1, 3))};
        biclusters.add(b1);
        biclusters.add(b2);

        double expected = 1.0 - Math.sqrt(
            (0.5*(2.0/4.0) + 0.5*(2.0/4.0)) * (0.5*(3.0/4.0) + 0.5*(2.0/4.0))
        );
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testBiclusterSizeWeightedNormCompOneBicluster() {
        FitnessFunction f = new BiclusterSizeNormComp(data, types, null, "Mean", 0.75);

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b = new ArrayList[]{new ArrayList<>(Arrays.asList(1, 2, 3)), new ArrayList<>(Arrays.asList(1, 2))};
        biclusters.add(b);

        double expected = 1.0 - (0.75*(3.0/4.0) + 0.25*(2.0/4.0));
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testBiclusterVarianceNormOneBicluster() {
        FitnessFunction f = new BiclusterVarianceNorm(data, types, null, "Mean");

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b = new ArrayList[]{new ArrayList<>(Arrays.asList(2, 3)), new ArrayList<>(Arrays.asList(2, 3))};
        biclusters.add(b);

        double expected = 1.0 - (
            1.0 - 4 * (
                Math.pow(0.5-0.5, 2) + Math.pow(0.1-0.5, 2) +
                Math.pow(0.8-0.5, 2) + Math.pow(0.6-0.5, 2)
            ) / 16.0
        );
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testRowVarianceNormCompOneBicluster() {
        FitnessFunction f = new RowVarianceNormComp(data, types, null, "Mean");

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b = new ArrayList[]{new ArrayList<>(Arrays.asList(2, 3)), new ArrayList<>(Arrays.asList(2, 3))};
        biclusters.add(b);

        double expected = 1.0 - (
            4 * (
                Math.pow(0.5-0.3, 2) + Math.pow(0.1-0.3, 2) +
                Math.pow(0.8-0.7, 2) + Math.pow(0.6-0.7, 2)
            ) / 4.0
        );
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testMSRNormOneBicluster() {
        FitnessFunction f = new MeanSquaredResidueNorm(data, types, null, "Mean");

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b = new ArrayList[]{new ArrayList<>(Arrays.asList(2, 3)), new ArrayList<>(Arrays.asList(2, 3))};
        biclusters.add(b);

        double expected = (
            Math.pow(0.5-0.3-0.65+0.5, 2) + Math.pow(0.1-0.3-0.35+0.5, 2) +
            Math.pow(0.8-0.7-0.65+0.5, 2) + Math.pow(0.6-0.7-0.35+0.5, 2)
        ) / 4.0 / 4.0;
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testDistanceBetweenBiclustersNormCompMean() {
        FitnessFunction f = new DistanceBetweenBiclustersNormComp(data, types, null, "Mean");

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b1 = new ArrayList[]{new ArrayList<>(Arrays.asList(0, 1)), new ArrayList<>(Arrays.asList(0, 1))};
        ArrayList<Integer>[] b2 = new ArrayList[]{new ArrayList<>(Arrays.asList(2, 3)), new ArrayList<>(Arrays.asList(2, 3))};
        biclusters.add(b1);
        biclusters.add(b2);

        double dy2b = (Math.pow(0.7-0.3, 2) + Math.pow(0.8-0.3, 2)) / 2.0;
        double dy2cb = (Math.pow(0.5-0.8, 2) + Math.pow(0.1-0.6, 2)) / 2.0;
        double fvy2 = dy2b / (dy2b + dy2cb);

        double dy3b = (Math.pow(0.2-0.3, 2) + Math.pow(0.3-0.3, 2)) / 2.0;
        double dy3cb = (Math.pow(0.5-0.8, 2) + Math.pow(0.1-0.6, 2)) / 2.0;
        double fvy3 = dy3b / (dy3b + dy3cb);

        double fvb1 = (fvy2 + fvy3) / 2.0;

        double dy0b = (Math.pow(0.7-0.65, 2) + Math.pow(0.3-0.35, 2)) / 2.0;
        double dy0cb = (Math.pow(0.5-0.1, 2) + Math.pow(0.2-0.4, 2)) / 2.0;
        double fvy0 = dy0b / (dy0b + dy0cb);
        
        double dy1b = (Math.pow(0.6-0.65, 2) + Math.pow(0.8-0.35, 2)) / 2.0;
        double dy1cb = (Math.pow(0.5-0.1, 2) + Math.pow(0.2-0.4, 2)) / 2.0;
        double fvy1 = dy1b / (dy1b + dy1cb);

        double fvb2 = (fvy0 + fvy1) / 2.0;

        double expected = 1.0 - (fvb1 + fvb2) / 2.0;
        assertEquals(expected, f.run(biclusters), epsilon);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testBiclusterSizeCoverNormCompMean() {
        FitnessFunction f = new BiclusterSizeNumBicsNormComp(data, types, null, "Mean", 0.5, 0.5);

        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        ArrayList<Integer>[] b1 = new ArrayList[]{new ArrayList<>(Arrays.asList(1, 2)), new ArrayList<>(Arrays.asList(0, 1))};
        ArrayList<Integer>[] b2 = new ArrayList[]{new ArrayList<>(Arrays.asList(0, 3)), new ArrayList<>(Arrays.asList(2, 3))};
        biclusters.add(b1);
        biclusters.add(b2);
        double b1Score = 0.5 * (0.5*(2.0/4.0) + 0.5*(2.0/4.0)) + 0.5 * (1 - ((4.0 - 4.0) / 4.0));
        double b2Score = 0.5 * (0.5*(2.0/4.0) + 0.5*(2.0/4.0)) + 0.5 * (1 - ((4.0 - 4.0) / 4.0));

        double expected = 1.0 - (b1Score + b2Score) / 2;
        assertEquals(expected, f.run(biclusters), epsilon);
    }
}
