package moeba.fitnessfunction;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import moeba.ColumnType;
import moeba.Representation;
import moeba.StaticUtils;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;


@Command(name = "FastEvaluationRunner", description = "Executes fitness functions quickly for a specific proposal", mixinStandardHelpOptions = true, showDefaultValues = true, sortOptions = false)
public class FastEvaluationRunner implements Runnable {

    @Option(names = {"--input-dataset"}, description = "Path to the input CSV dataset on which you want to perform biclustering", required = true)
    private File inputDataset;

    @Option(names = {"--input-column-types"}, description = "Path to the input JSON file which maps each CSV column name to a semantic column type", required = true)
    private File inputColumnTypes;

    @Option(names = {"--solution-translated"}, description = "Path to the input CSV file with solution translated biclusters.", required = true)
    private File solutionTranslatedFile;

    @Option(names = {"--representation"}, description = "Representation as a string. Possible values: GENERIC, INDIVIDUAL", defaultValue = "GENERIC")
    private Representation representation;

    @Option(names = {"--str-fitness-functions"}, 
            description = "Objectives to optimize separated by semicolon. Possible values: \n" + //
                "\t- General purpose objectives (Any representation): BiclusterSizeNormComp, BiclusterVarianceNorm, RowVarianceNormComp, MeanSquaredResidueNorm \n" + //
                "\t- General purpose objectives (GENERIC representation): BiclusterSizeNumBicsNormComp, DistanceBetweenBiclustersNormComp \n" + //
                "\t- Co-Expression objectives (GENERIC representation): RegulatoryCoherenceNormComp \n" + //
                "In case any objective requires additional parameters, they shall be specified in brackets in the following way ObjectiveName(parameter1=value, parameter2=value, ...)", 
            defaultValue = "BiclusterSizeNormComp;MeanSquaredResidueNorm")
    private String strFitnessFormulas;

    @Option(names = {"--summarise-individual-objectives"}, description = "Way to summarise the overall quality of the solutions from the individual quality of their biclusters. Only for GENERIC, SPECIFIC or DYNAMIC representation. Possible values: Mean, HarmonicMean, GeometricMean", defaultValue = "Mean")
    private String summariseIndividualObjectives;

    @Option(names = {"--num-threads"}, description = "Number of threads. Default: All")
    private int numThreads = Runtime.getRuntime().availableProcessors();

    @Option(names = {"--output-file"}, description = "Path to the output CSV file to save results.", defaultValue = "FastFUN.csv")
    private File outputFile;

    @Override
    public void run() {

        // Read input dataset
        String[][] data = null;
        try {
            data = StaticUtils.csvToStringMatrix(inputDataset);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Read column names from input CSV file
        String[] columnNames = null;
        try {
            BufferedReader br = new BufferedReader(new FileReader(inputDataset));
            columnNames = br.readLine().split(",");
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Read column types
        ColumnType[] types = null;
        try {
            types = StaticUtils.jsonToColumnTypes(inputColumnTypes, columnNames);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Convert data to numeric matrix in base of types
        double[][] numericData = StaticUtils.dataToNumericMatrix(data, types, numThreads);
        data = null;

        // Get fitness functions
        String[] strFitnessFunctions = strFitnessFormulas.split(";");
        FitnessFunction[] fitnessFunctions = new FitnessFunction[strFitnessFunctions.length];
        for (int i = 0; i < strFitnessFunctions.length; i++) {
            fitnessFunctions[i] = StaticUtils.getFitnessFunctionFromString(strFitnessFunctions[i], numericData, types, null, summariseIndividualObjectives);
        }

        // Load inferred biclusters from CSV file
        ArrayList<ArrayList<ArrayList<Integer>[]>> biclusters = StaticUtils.loadBiclusters(solutionTranslatedFile);

        // Flatten inferred biclusters if representation is INDIVIDUAL
        if (representation == Representation.INDIVIDUAL) {
            ArrayList<ArrayList<ArrayList<Integer>[]>> individualBiclusters = new ArrayList<>();
            for (ArrayList<ArrayList<Integer>[]> genericBiclusters : biclusters) {
                for (ArrayList<Integer>[] bicluster : genericBiclusters) {
                    ArrayList<ArrayList<Integer>[]> individualBicluster = new ArrayList<>();
                    individualBicluster.add(bicluster);
                    individualBiclusters.add(individualBicluster);
                }
            }
            biclusters = individualBiclusters;
            individualBiclusters = null;
        }

        // Run evaluation
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            for (int i = 0; i < strFitnessFunctions.length; i++) {
                writer.print("\"" + strFitnessFunctions[i] + "\"");
                if (i < strFitnessFunctions.length - 1) writer.print(",");
            }
            writer.println();
            for (ArrayList<ArrayList<Integer>[]> bicluster : biclusters) {
                for (int j = 0; j < fitnessFunctions.length; j++) {
                    writer.print(fitnessFunctions[j].run(bicluster));
                    if (j < fitnessFunctions.length - 1) writer.print(",");
                }
                writer.println();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public static void main(String[] args) {
        CommandLine commandLine = new CommandLine(new FastEvaluationRunner());
        commandLine.execute(args);
    }

}
