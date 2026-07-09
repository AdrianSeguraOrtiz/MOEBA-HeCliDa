package moeba.representationwrapper.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.Map;
import java.util.LinkedHashMap;

import moeba.StaticUtils;
import moeba.fitnessfunction.FitnessFunction;
import moeba.fitnessfunction.GenericBiclusterFitnessFunction;
import moeba.fitnessfunction.GlobalFitnessFunction;
import moeba.fitnessfunction.IndividualBiclusterFitnessFunction;
import moeba.operator.crossover.generic.GenericCrossover;
import moeba.operator.crossover.generic.biclusterbinary.BiclusterBinaryCrossover;
import moeba.operator.crossover.generic.biclusterbinary.impl.BicUniformCrossover;
import moeba.operator.crossover.generic.cellbinary.CellBinaryCrossover;
import moeba.operator.crossover.generic.cellbinary.impl.CellUniformCrossover;
import moeba.operator.crossover.generic.rowbiclustermixed.RowBiclusterMixedCrossover;
import moeba.operator.crossover.generic.rowbiclustermixed.impl.GroupedBasedCrossover;
import moeba.operator.crossover.generic.rowpermutation.RowPermutationCrossover;
import moeba.operator.crossover.generic.rowpermutation.impl.CycleCrossover;
import moeba.operator.crossover.generic.rowpermutation.impl.EdgeRecombinationCrossover;
import moeba.operator.crossover.generic.rowpermutation.impl.PartiallyMappedCrossover;
import moeba.operator.mutation.generic.GenericMutation;
import moeba.operator.mutation.generic.biclusterbinary.BiclusterBinaryMutation;
import moeba.operator.mutation.generic.biclusterbinary.impl.BicUniformMutation;
import moeba.operator.mutation.generic.cellbinary.CellBinaryMutation;
import moeba.operator.mutation.generic.cellbinary.impl.CellUniformMutation;
import moeba.operator.mutation.generic.rowpermutation.RowPermutationMutation;
import moeba.operator.mutation.generic.rowpermutation.impl.SwapMutation;
import moeba.representationwrapper.RepresentationWrapper;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.solution.integersolution.IntegerSolution;
import org.uma.jmetal.util.binarySet.BinarySet;

public class GenericRepresentationWrapper extends RepresentationWrapper {
    private static final String DEFAULT_CROSSOVER_OPERATOR = "GroupedBasedCrossover;CellUniformCrossover";
    private static final String DEFAULT_MUTATION_OPERATOR = "SwapMutation;BicUniformMutation;CellUniformMutation";
    private static final Set<Class<? extends FitnessFunction>> SUPPORTED_FITNESS_FUNCTION_TYPES =
        Set.<Class<? extends FitnessFunction>>of(
            IndividualBiclusterFitnessFunction.class,
            GenericBiclusterFitnessFunction.class,
            GlobalFitnessFunction.class
        );

    private float genericInitialMinPercBics;
    private float genericInitialMaxPercBics;
    private String summariseIndividualObjectives;
    private Random random;

    public GenericRepresentationWrapper(int numRows, int numColumns, float genericInitialMinPercBics, float genericInitialMaxPercBics, String summariseIndividualObjectives) {
        super(numRows, numColumns);
        this.genericInitialMinPercBics = genericInitialMinPercBics;
        this.genericInitialMaxPercBics = genericInitialMaxPercBics;
        this.summariseIndividualObjectives = summariseIndividualObjectives;
        this.random = new Random();
    }

    public CompositeSolution buildComposition(IntegerSolution integerSolution, BinarySolution binarySolution) {

        // Ensure that the initial number of biclusters is varied within an acceptable range
        binarySolution.variables().get(0).clear();
        float limit = random.nextFloat()*(genericInitialMaxPercBics - genericInitialMinPercBics) + genericInitialMinPercBics;

        // Ensure that the integer part is a permutation
        List<Integer> rowIndexes = IntStream.rangeClosed(0, super.numRows - 1).boxed().collect(Collectors.toList());
        Collections.shuffle(rowIndexes);

        // Take advantage of the loop to perform both operations simultaneously
        for (int i = 0; i < super.numRows; i++) {
            integerSolution.variables().set(i, rowIndexes.get(i));
            if (random.nextFloat() < limit) {
                binarySolution.variables().get(0).set(i);
            }
        }

        return new CompositeSolution(Arrays.asList(integerSolution, binarySolution));
    }

    @Override
    public int getNumIntVariables() {
        return super.numRows; 
    }

    @Override
    public int getNumBinaryVariables() {
        return 1 + super.numColumns;
    }

    @Override
    public int getLowerIntegerBound() {
        return 0;
    }

    @Override
    public int getUpperIntegerBound() {
        return super.numRows - 1;
    }

    @Override
    public int getNumBitsPerVariable() {
        return super.numRows;
    }

    @Override
    public String getSummariseMethod() {
        return summariseIndividualObjectives;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ArrayList<ArrayList<Integer>[]> getBiclustersFromRepresentation(CompositeSolution solution) {
        
        // Initialize the result list
        ArrayList<ArrayList<Integer>[]> res = new ArrayList<>();
        
        // Extract integer and binary variables from the composite solution
        List<Integer> integerVariables = ((IntegerSolution) solution.variables().get(0)).variables();
        List<BinarySet> binaryVariables = ((BinarySolution) solution.variables().get(1)).variables();
        
        // Initialize rows, cols, minRows, minRow and precalculatedSums
        ArrayList<Integer> rows = new ArrayList<>();
        ArrayList<Integer> cols = new ArrayList<>();
        ArrayList<Integer> minRows = new ArrayList<>();
        int minRow = super.numRows;
        int[][] precalculatedSums = new int[super.numColumns][super.numRows + 1];
        
        // Calculate precalculatedSums
        for (int j = 0; j < super.numColumns; j++) {
            precalculatedSums[j][0] = 0;
            for (int i = 1; i <= super.numRows; i++) {
                precalculatedSums[j][i] = precalculatedSums[j][i - 1] + (binaryVariables.get(j+1).get(integerVariables.get(i-1)) ? 1 : 0);
            }
        }

        // Extract biclusters
        for (int i = 0; i < super.numRows; i++) {
            int row = integerVariables.get(i);
            rows.add(row);
            if (row < minRow) minRow = row;
            if (binaryVariables.get(0).get(i) || i == super.numRows - 1) {
                for (int j = 0; j < super.numColumns; j++) {
                    if (((float) (precalculatedSums[j][i + 1] - precalculatedSums[j][i - rows.size() + 1]) / rows.size()) > 0.5) {
                        cols.add(j);
                    }
                }
                
                // Create and add bicluster to the result list
                ArrayList<Integer>[] bicluster = new ArrayList[2];
                Collections.sort(rows);
                bicluster[0] = new ArrayList<>(rows);
                Collections.sort(cols);
                bicluster[1] = new ArrayList<>(cols);
                res.add(bicluster);

                // Clear rows and cols for next iteration
                rows.clear();
                cols.clear();

                // Add minRow to list and reset for next iteration
                minRows.add(minRow);
                minRow = super.numRows;
            }
        }

        // Clean unused variables
        rows = null;
        cols = null;
        precalculatedSums = null;

        // Sort list of biclusters depending on the smallest row
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < minRows.size(); i++) {
            indexes.add(i);
        }
        Collections.sort(indexes, Comparator.comparing(minRows::get));
        ArrayList<ArrayList<Integer>[]> resOrdered = new ArrayList<>();
        for (int index : indexes) {
            resOrdered.add(res.get(index));
        }

        // Clean unused variables
        res = null;

        // Merge biclusters with same columns
        // TODO: Hacer esto en fase de reparacion / busqueda local
        //mergeBiclustersSameColumns(resOrdered, solution);

        return resOrdered;
    }

    @Override
    public String[] getVarLabels() {
        String[] varLabels = new String[2 * super.numRows + super.numRows * super.numColumns];
        for (int i = 0; i < super.numRows; i++) {
            varLabels[i] = "R" + i;
        }
        for (int i = 0; i < super.numRows; i++) {
            varLabels[i + super.numRows] = "P" + i;
        }
        for (int i = 2*super.numRows; i < varLabels.length; i++) {
            varLabels[i] = "Cell-R" + (i % super.numRows) + "-C" + ((i / super.numRows) - 2);
        }
        return varLabels;
    }

    @Override
    public String getDefaultCrossoverOperator() {
        return DEFAULT_CROSSOVER_OPERATOR;
    }

    @Override
    public String getDefaultMutationOperator() {
        return DEFAULT_MUTATION_OPERATOR;
    }

    @Override
    public Set<Class<? extends FitnessFunction>> getSupportedFitnessFunctionTypes() {
        return SUPPORTED_FITNESS_FUNCTION_TYPES;
    }

    @Override
    public CrossoverOperator<CompositeSolution> getCrossoverFromString(String strCrossoverOperator, double crossoverProbability, int numApproxCrossovers) {
        CrossoverOperator<CompositeSolution> res;
        String[] listStrCrossovers = strCrossoverOperator.split(";");

        if (listStrCrossovers.length == 2) {
            RowBiclusterMixedCrossover rowBiclusterMixedCrossover = getRowBiclusterMixedCrossoverFromString(listStrCrossovers[0], numApproxCrossovers);
            CellBinaryCrossover cellBinaryCrossover = getCellBinaryCrossoverFromString(listStrCrossovers[1]);
            res = new GenericCrossover(crossoverProbability, rowBiclusterMixedCrossover, cellBinaryCrossover);
        } else if (listStrCrossovers.length == 3) {
            RowPermutationCrossover rowPermutationCrossover = getRowPermutationCrossoverFromString(listStrCrossovers[0]);
            BiclusterBinaryCrossover biclusterBinaryCrossover = getBiclusterBinaryCrossoverFromString(listStrCrossovers[1]);
            CellBinaryCrossover cellBinaryCrossover = getCellBinaryCrossoverFromString(listStrCrossovers[2]);
            res = new GenericCrossover(crossoverProbability, rowPermutationCrossover, biclusterBinaryCrossover, cellBinaryCrossover);
        } else {
            throw new RuntimeException("The number of crossover operators is not supported for GENERIC representation.");
        }
        return res;
    }

    @Override
    public MutationOperator<CompositeSolution> getMutationFromString(String strMutationOperator, String mutationProbability, int numApproxMutations) {
        MutationOperator<CompositeSolution> res;
        String[] listStrMutations = strMutationOperator.split(";");

        if (listStrMutations.length == 3) {
            RowPermutationMutation rowPermutationMutation = getRowPermutationMutationFromString(listStrMutations[0]);
            BiclusterBinaryMutation biclusterBinaryMutation = getBiclusterBinaryMutationFromString(listStrMutations[1]);
            CellBinaryMutation cellBinaryMutation = getCellBinaryMutationFromString(listStrMutations[2]);
            res = new GenericMutation(mutationProbability, numApproxMutations, rowPermutationMutation, biclusterBinaryMutation, cellBinaryMutation);
        } else {
            throw new RuntimeException("The number of mutation operators is not supported for the GENERIC representation.");
        }

        return res;
    }

    public RowPermutationCrossover getRowPermutationCrossoverFromString(String str) {
        RowPermutationCrossover res;
        switch (str.toLowerCase()) {
            case "cyclecrossover":
                res = new CycleCrossover();
                break;
            case "edgerecombinationcrossover":
                res = new EdgeRecombinationCrossover();
                break;
            case "partiallymappedcrossover":
                res = new PartiallyMappedCrossover();
                break;
            default:
                throw new RuntimeException("The row permutation crossover " + str + " is not implemented.");
        }
        return res;
    }

    public BiclusterBinaryCrossover getBiclusterBinaryCrossoverFromString(String str) {
        BiclusterBinaryCrossover res;
        switch (str.toLowerCase()) {
            case "bicuniformcrossover":
                res = new BicUniformCrossover();
                break;
            default:
                throw new RuntimeException(
                        "The bicluster binary crossover " + str + " is not implemented.");
        }
        return res;
    }

    public CellBinaryCrossover getCellBinaryCrossoverFromString(String str) {
        CellBinaryCrossover res;
        switch (str.toLowerCase()) {
            case "celluniformcrossover":
                res = new CellUniformCrossover();
                break;
            default:
                throw new RuntimeException(
                        "The cell binary crossover " + str + " is not implemented.");
        }
        return res;
    }

    public static RowBiclusterMixedCrossover getRowBiclusterMixedCrossoverFromString(String str, int numApproxCrossovers) {
        RowBiclusterMixedCrossover res;

        if (str.toLowerCase().startsWith("groupedbasedcrossover")) {
            Map<String, String> subParams = StaticUtils.getSubParams("groupedbasedcrossover", str);
            res = new GroupedBasedCrossover(
                numApproxCrossovers, 
                Float.parseFloat(StaticUtils.getOne("groupedbasedcrossover", subParams, "shuffleend", "0.75")),
                Float.parseFloat(StaticUtils.getOne("groupedbasedcrossover", subParams, "dynamicstartamount", "0.25"))
            );
        } else {
            throw new RuntimeException("The row bicluster mixed crossover " + str + " is not implemented.");
        }
        
        return res;
    }

    public static BiclusterBinaryMutation getBiclusterBinaryMutationFromString(String str) {
        BiclusterBinaryMutation res;
        switch (str.toLowerCase()) {
            case "bicuniformmutation":
                res = new BicUniformMutation();
                break;
            default:
                throw new RuntimeException(
                        "The bicluster binary mutation " + str + " is not implemented.");
        }
        return res;
    }

    public static CellBinaryMutation getCellBinaryMutationFromString(String str) {
        CellBinaryMutation res;
        switch (str.toLowerCase()) {
            case "celluniformmutation":
                res = new CellUniformMutation();
                break;
            default:
                throw new RuntimeException(
                        "The cell binary mutation " + str + " is not implemented.");
        }
        return res;
    }

    public static RowPermutationMutation getRowPermutationMutationFromString(String str) {
        RowPermutationMutation res;
        switch (str.toLowerCase()) {
            case "swapmutation":
                res = new SwapMutation();
                break;
            default:
                throw new RuntimeException(
                        "The row permutation mutation " + str + " is not implemented.");
        }
        return res;
    }

    @SuppressWarnings("unchecked")
    public void mergeBiclustersSameColumns(ArrayList<ArrayList<Integer>[]> biclusters, CompositeSolution solution) {

        // Mapa para agrupar filas por sus columnas correspondientes
        Map<String, ArrayList<Integer>> map = new LinkedHashMap<>();

        // For each bicluster, add its rows to the map with the columns as key
        for (ArrayList<Integer>[] bicluster : biclusters) {
            String key = bicluster[1].toString(); // column list as string
            map.computeIfAbsent(key, k -> new ArrayList<>())
                .addAll(bicluster[0]); // add rows to column list
        }

        // Clear old biclusters list
        biclusters.clear();

        // For each group of rows with the same columns, create a new bicluster
        for (Map.Entry<String, ArrayList<Integer>> entry : map.entrySet()) {
            // Get rows and sort them
            ArrayList<Integer> rows = entry.getValue();
            Collections.sort(rows);

            // Reconstruct columns from the key
            String key = entry.getKey();
            ArrayList<Integer> cols = key.length() <= 2 ? new ArrayList<>() : new ArrayList<>(Arrays.asList(key.substring(1, key.length() - 1).split(", ")))
                .stream().map(Integer::parseInt)
                .collect(Collectors.toCollection(ArrayList::new));

            // Create and add the new bicluster
            ArrayList<Integer>[] bicluster = new ArrayList[2];
            bicluster[0] = rows;
            bicluster[1] = cols;
            biclusters.add(bicluster);
        }
    }
    
}
