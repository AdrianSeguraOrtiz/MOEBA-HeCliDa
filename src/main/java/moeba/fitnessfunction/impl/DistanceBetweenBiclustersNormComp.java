package moeba.fitnessfunction.impl;

import java.util.ArrayList;

import moeba.ColumnType;
import moeba.fitnessfunction.GenericBiclusterFitnessFunction;
import moeba.utils.storage.CacheStorage;

public class DistanceBetweenBiclustersNormComp extends GenericBiclusterFitnessFunction {

    public DistanceBetweenBiclustersNormComp(double[][] data, ColumnType[] types,
            CacheStorage<String, Double> internalCache, String summariseIndividualObjectives) {
        super(data, types, internalCache, summariseIndividualObjectives);
    }

    @Override
    protected double getBiclusterScore(ArrayList<Integer>[] bicluster, ArrayList<ArrayList<Integer>[]> biclusters) {
        double totalScore = 0.0;

        // Paso 0: Encontrar el bicluster más cercano (mayor cantidad de columnas compartidas)
        ArrayList<Integer>[] closestBicluster = findClosestBicluster(bicluster, biclusters);

        // Paso 1: Calcular el promedio de cada bicluster
        double[] meanBicluster = calculateMean(bicluster);
        double[] meanClosestBicluster = calculateMean(closestBicluster);

        // Paso 2: Para cada fila en el bicluster más cercano
        for (Integer row : closestBicluster[0]) {

            // Paso 2.0: Cargar los datos de la fila
            // Para las columnas del bicluster más cercano
            double[] yiClosestBic = new double[closestBicluster[1].size()];
            for (int j = 0; j < closestBicluster[1].size(); j++) {
                yiClosestBic[j] = data[row][closestBicluster[1].get(j)];
            }
            // Para las columnas del bicluster evaluado
            double[] yiBic = new double[bicluster[1].size()];
            for (int j = 0; j < bicluster[1].size(); j++) {
                yiBic[j] = data[row][bicluster[1].get(j)];
            }

            // Paso 2.1: Distancia de yiBic al promedio del bicluster evaluado
            double distanceBiclusterMean = calculateDistanceToBiclusterMean(yiBic, meanBicluster);

            // Paso 2.2: Distancia de yi al promedio ajustado del bicluster cercano
            double[] adjustedMeanClosestBicluster = calculateAdjustedMean(meanClosestBicluster, yiClosestBic, bicluster[0].size());
            double distanceClosestBicAdjustedMean = calculateDistanceToBiclusterMean(yiClosestBic, adjustedMeanClosestBicluster);

            // Paso 2.3: Calcular el valor de encaje
            double fitScore = calculateFitScore(distanceBiclusterMean, distanceClosestBicAdjustedMean);

            // Acumular el puntaje
            totalScore += fitScore;
        }

        // Paso 3: Promedio de los valores
        return totalScore / closestBicluster[0].size();
    }

    // Encuentra el bicluster más cercano basado en el mayor número de columnas compartidas
    private ArrayList<Integer>[] findClosestBicluster(ArrayList<Integer>[] bicluster, ArrayList<ArrayList<Integer>[]> biclusters) {
        ArrayList<Integer>[] closestBicluster = biclusters.get(0);
        int maxSharedColumns = 0;

        for (ArrayList<Integer>[] otherBicluster : biclusters) {
            int sharedColumns = countSharedColumns(bicluster[1], otherBicluster[1]);
            if (sharedColumns > maxSharedColumns) {
                maxSharedColumns = sharedColumns;
                closestBicluster = otherBicluster;
            }
        }
        return closestBicluster;
    }

    // Cuenta las columnas compartidas entre dos conjuntos de columnas
    private int countSharedColumns(ArrayList<Integer> cols1, ArrayList<Integer> cols2) {
        int shared = 0;
        for (Integer col : cols1) {
            if (cols2.contains(col)) {
                shared++;
            }
        }
        return shared;
    }

    // Calcula el promedio de las columnas del bicluster
    private double[] calculateMean(ArrayList<Integer>[] bicluster) {
        double[] mean = new double[bicluster[1].size()];
        for (int j = 0; j < bicluster[1].size(); j++) {
            double sum = 0.0;
            for (Integer row : bicluster[0]) {
                sum += data[row][bicluster[1].get(j)];
            }
            mean[j] = sum / bicluster[0].size();
        }

        return mean;
    }

    // Calcula la distancia entre la fila yi y el promedio de las columnas del bicluster
    private double calculateDistanceToBiclusterMean(double[] yi, double[] meanBicluster) {
        double sum = 0.0;
        for (int j = 0; j < yi.length; j++) {
            sum += Math.pow(yi[j] - meanBicluster[j], 2);
        }
        return sum / yi.length;
    }

    // Ajusta el promedio de un bicluster al quitar una fila yi
    private double[] calculateAdjustedMean(double[] meanBicluster, double[] yi, int n) {
        double[] adjustedMean = new double[meanBicluster.length];
        for (int j = 0; j < meanBicluster.length; j++) {
            adjustedMean[j] = (n * meanBicluster[j] - yi[j]) / (n - 1);
        }
        return adjustedMean;
    }

    // Calcula el valor de ajuste basado en las distancias a los promedios
    private double calculateFitScore(double distanceToCurrentMean, double distanceToAdjustedMean) {
        return distanceToCurrentMean / (distanceToCurrentMean + distanceToAdjustedMean);
    }
}
