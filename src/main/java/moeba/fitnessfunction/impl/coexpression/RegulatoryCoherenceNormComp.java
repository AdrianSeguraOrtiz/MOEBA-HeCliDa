package moeba.fitnessfunction.impl.coexpression;

import java.util.ArrayList;

import moeba.ColumnType;
import moeba.fitnessfunction.GlobalFitnessFunction;
import moeba.utils.coexpression.GeneRegulatoryNetwork;

/**
 * The RegulatoryCoherenceNormComp class is a global fitness function designed to measure the modularity 
 * of the partition into communities, where the communities are represented by the rows of biclusters. 
 * This partition is evaluated on the regulatory network inferred by Genie3 based on the input gene expression data.
 * 
 * Reason: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247671
 */
public class RegulatoryCoherenceNormComp extends GlobalFitnessFunction {
    private GeneRegulatoryNetwork regNetwork; // The gene regulatory network inferred from gene expression data

    /**
     * Constructor that initializes the fitness function with gene expression data and types for the data.
     * It also infers the gene regulatory network using the provided data.
     * 
     * @param data A 2D double array representing the gene expression data, where rows correspond to genes
     *             and columns correspond to conditions or samples.
     * @param types An array representing the semantic data types of the columns.
     */
    public RegulatoryCoherenceNormComp(double[][] data, ColumnType[] types) {
        super(data, types);
        this.regNetwork = new GeneRegulatoryNetwork(data);  // Create the gene regulatory network
        this.func = biclusters -> getRegulatoryCoherence(biclusters);  // Define the fitness function
    }

    /**
     * Maps each gene (row) to its corresponding bicluster index.
     * 
     * @param biclusters A list of biclusters, where each bicluster contains two arrays:
     *                   one for the gene indices (rows) and one for the condition indices (columns).
     * @return An integer array where the value at each index represents the bicluster index the gene belongs to.
     */
    private int[] getRowBiclusters(ArrayList<ArrayList<Integer>[]> biclusters) {
        int[] rowBiclusters = new int[regNetwork.getNumNodes()];
        for (int i = 0; i < biclusters.size(); i++) {
            for (int j = 0; j < biclusters.get(i)[0].size(); j++) {
                rowBiclusters[biclusters.get(i)[0].get(j)] = i + 1; // Assign bicluster index (1-based)
            }
        }
        return rowBiclusters;
    }

    /**
     * Calculates the regulatory coherence of the biclusters by comparing the regulatory relationships
     * within the genes grouped in the same bicluster. The coherence is measured based on the gene regulatory network.
     * 
     * @param biclusters A list of biclusters, where each bicluster contains two arrays:
     *                   one for the gene indices (rows) and one for the condition indices (columns).
     * @return A double value representing the normalized regulatory coherence. The value is between 0 and 1,
     *         where 0 represents the best coherence and 1 represents the worst.
     */
    private double getRegulatoryCoherence(ArrayList<ArrayList<Integer>[]> biclusters) {
        int[] rowBiclusters = getRowBiclusters(biclusters);  // Get the bicluster assignments for genes
        float sum = 0;

        // Loop through all pairs of genes in the network and accumulate coherence based on regulatory relationships
        for (int i = 0; i < regNetwork.getNumNodes(); i++) {
            for (int j = 0; j < regNetwork.getNumNodes(); j++) {
                // Only consider gene pairs in the same bicluster and skip unassigned genes (bicluster index = 0)
                if (rowBiclusters[i] == rowBiclusters[j] && rowBiclusters[i] != 0) {
                    sum += regNetwork.getConfidence(i, j) 
                           - (regNetwork.getOutDegree(i) * regNetwork.getInDegree(j)) / regNetwork.getTotalWeight();
                }
            }
        }

        // Calculate the modularity-like score based on the accumulated sum and total weight of the network
        double mod = sum / regNetwork.getTotalWeight();

        // Normalize the modularity value to a range between 0 (best) and 1 (worst)
        return 1 - (mod + 1) / 2;
    }
}
