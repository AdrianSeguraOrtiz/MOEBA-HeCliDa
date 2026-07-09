package moeba;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.ehcache.CacheManager;
import org.ehcache.config.builders.CacheManagerBuilder;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.selection.impl.BinaryTournamentSelection;
import org.uma.jmetal.operator.selection.impl.NaryTournamentSelection;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.util.AbstractAlgorithmRunner;
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import moeba.StaticUtils.AlgorithmResult;
import moeba.representationwrapper.RepresentationWrapper;
import moeba.utils.observer.ProblemObserver;
import moeba.utils.observer.ProblemObserver.ObserverInterface;
import moeba.utils.output.SolutionListTranslatedVAR;
import moeba.utils.output.SolutionListVARWithHeader;
import moeba.utils.storage.CacheStorage;
import moeba.utils.storage.impl.HybridCache;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

@Command(name = "RunnerMOEBA", description = "Multi-Objective Evolutionary Biclustering Algorithm (MOEBA) for Heterogeneous Clinical Data (HeCliDa) with progressive representation for self-determination on the number of clusters", mixinStandardHelpOptions = true, showDefaultValues = true, sortOptions = false)
public class Runner extends AbstractAlgorithmRunner implements Runnable {

    private static final String AUTO = "AUTO";

    @Option(names = {"--input-dataset"}, description = "Path to the input CSV dataset on which you want to perform biclustering", required = true)
    private File inputDataset;

    @Option(names = {"--input-column-types"}, description = "Path to the input JSON file which specifies the names of the columns in order and the type of data of each of them", required = true)
    private File inputColumnTypes;

    @Option(names = {"--representation"}, description = "Representation as a string. Possible values: GENERIC, SPECIFIC, INDIVIDUAL, DYNAMIC", defaultValue = "GENERIC")
    private Representation representation;

    @Option(names = {"--specific-num-biclusters"}, description = "Number of biclusters. Only for SPECIFIC representation", defaultValue = "-1")
    private int specificNumBiclusters;

    @Option(names = {"--generic-initial-min-num-bics"}, description = "Initial minimum number of biclusters. Only for GENERIC representation. Default: 5%% of the number of rows", defaultValue = "-1")
    private int genericInitialMinNumBics;

    @Option(names = {"--generic-initial-max-num-bics"}, description = "Initial maximum number of biclusters. Only for GENERIC representation. Default: 25%% of the number of rows", defaultValue = "-1")
    private int genericInitialMaxNumBics;

    @Option(names = {"--str-fitness-functions"}, 
            description = "Objectives to optimize separated by semicolon. Possible values: \n" + //
                "\t- General purpose objectives (Any representation): BiclusterSizeNormComp, BiclusterVarianceNorm, RowVarianceNormComp, MeanSquaredResidueNorm \n" + //
                "\t- General purpose objectives (GENERIC or SPECIFIC representation): BiclusterSizeNumBicsNormComp, DistanceBetweenBiclustersNormComp \n" + //
                "\t- Co-Expression objectives (GENERIC or SPECIFIC representation): RegulatoryCoherenceNormComp \n" + //
                "In case any objective requires additional parameters, they shall be specified in brackets in the following way ObjectiveName(parameter1=value, parameter2=value, ...)", 
            defaultValue = "BiclusterSizeNormComp;MeanSquaredResidueNorm")
    private String strFitnessFormulas;

    @Option(names = {"--summarise-individual-objectives"}, 
            description = "Way to summarise the overall quality of the solutions from the individual quality of their biclusters. Only for GENERIC, SPECIFIC or DYNAMIC representation. Possible values: Mean, HarmonicMean, GeometricMean \n" + //
                "If this value has already been specified for any objective, this parameter will be ignored for that case",
            defaultValue = "Mean")
    private String summariseIndividualObjectives;

    @Option(names = {"--population-size"}, description = "Population size", defaultValue = "100")
    private int populationSize;

    @Option(names = {"--max-evaluations"}, description = "Max number of evaluations", defaultValue = "25000")
    private int maxEvaluations;

    @Option(names = {"--str-algorithm"}, 
            description = "Algorithm as a string. Possible values: \n" + //
                "\t- Single Objective: GA-AsyncParallel, GA-SingleThread \n" + //
                "\t- Multi Objective: NSGAII-AsyncParallel, NSGAII-SingleThread, MOEAD-SingleThread, SMS-EMOA-SingleThread, MOCell-SingleThread, SPEA2-SingleThread, IBEA-SingleThread, NSGAIII-SingleThread, MOSA-SingleThread \n" + //
                "\t- Many Objective: NSGAII-ExternalFile-AsyncParallel \n" + //
                "In case any algorithm requires additional parameters, they shall be specified in brackets in the following way AlgorithmName(parameter1=value, parameter2=value, ...)", 
            defaultValue = "NSGAII-AsyncParallel")
    private String strAlgorithm;

    @Option(names = {"--crossover-probability"}, description = "Crossover probability", defaultValue = "0.9")
    private double crossoverProbability;

    @Option(names = {"--mutation-probability"}, description = "Mutation probability. If a progressive mutation is desired, a range must be defined by using '->' (e.g. 0.3->0.05).", defaultValue = "0.1")
    private String mutationProbability;

    @Option(names = {"--crossover-operator"}, 
            description = "Crossover operator. The following are configuration templates according to each type of representation:\n" + //
                "\t- GENERIC: RowPermutationCrossover;BiclusterBinaryCrossover;CellBinaryCrossover or RowBiclusterMixedCrossover;CellBinaryCrossover\n" + //
                "\t- SPECIFIC: ...\n" + //
                "\t- INDIVIDUAL: RowColBinaryCrossover\n" + //
                "\t- DYNAMIC: GENERIC-SPECIFIC\n" + //
                "Use AUTO to select a representation-aware default. In case any operator requires additional parameters, they shall be specified in brackets in the following way OperatorName(parameter1=value, parameter2=value, ...)",
            defaultValue = AUTO)
    private String strCrossoverOperator;

    @Option(names = {"--mutation-operator"}, 
            description = "Mutation operator. Same explanation as for the crossover operator:\n" + //
                "\t- GENERIC: RowPermutationMutation;BiclusterBinaryMutation;CellBinaryMutation\n" + //
                "\t- SPECIFIC: ...\n" + //
                "\t- INDIVIDUAL: RowColBinaryMutation\n" + //
                "\t- DYNAMIC: GENERIC-SPECIFIC\n" + //
                "Use AUTO to select a representation-aware default.", 
            defaultValue = AUTO)
    private String strMutationOperator;

    @Option(names = {"--have-external-cache"}, description = "Whether the external cache is used")
    private boolean haveExternalCache;

    @Option(names = {"--have-internal-cache"}, description = "Whether the internal cache is used")
    private boolean haveInternalCache;

    @Option(names = {"--observers"}, description = "List of observers separated by semicolon. Possible values: BiclusterCountObserver, FitnessEvolutionMinObserver, FitnessEvolutionAvgObserver, FitnessEvolutionMaxObserver, NumEvaluationsObserver, ExternalCacheObserver, InternalCacheObserver. Use AUTO to select representation-aware defaults.", defaultValue = AUTO)
    private String strObservers;

    @Option(names = {"--num-threads"}, description = "Number of threads. Default: All")
    private int numThreads = Runtime.getRuntime().availableProcessors();

    @Option(names = {"--output-folder"}, description = "Output folder")
    private String outputFolder;

    // Store solutions
    private List<CompositeSolution> solutions;

    // Store observers
    private ObserverInterface[] observers;

    @SuppressWarnings("unchecked")
    @Override
    public void run() {
        // Config sort. NOTE: https://github.com/jMetal/jMetal/issues/446
        System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");

        // If the representation is not SPECIFIC the number of biclusters cant be set
        if (this.representation != Representation.SPECIFIC && this.specificNumBiclusters != -1) {
            throw new IllegalArgumentException("No se puede fijar el número de biclusters para la representación " + this.representation);
        }

        // If the representation is not GENERIC the initial number of biclusters cant be set
        if (this.representation != Representation.GENERIC && (this.genericInitialMinNumBics != -1 || this.genericInitialMaxNumBics != -1)) {
            throw new IllegalArgumentException("No se puede fijar el rango inicial de biclusters para la representación " + this.representation);
        }

        // If the representation is INDIVIDUAL summariseIndividualObjectives must be Mean
        if (this.representation == Representation.INDIVIDUAL && !this.summariseIndividualObjectives.equals("Mean")) {
            throw new IllegalArgumentException("No se puede fijar la suma de objetivos individuales para la representación " + this.representation);
        }

        String resolvedObservers = resolveObservers();

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
        Class<?>[] types = null;
        try {
            types = StaticUtils.jsonToClassArray(inputColumnTypes, columnNames);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Convert data to numeric matrix in base of types
        double[][] numericData = StaticUtils.dataToNumericMatrix(data, types, numThreads);
        data = null;

        // Create Hybrid Caches Manager
        BasicConfigurator.configure();
        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.OFF);
        CacheManager hybridCacheManager = CacheManagerBuilder.newCacheManagerBuilder().build();
        hybridCacheManager.init();

        // Evolución central con representación genérica
        // 1. Array de funciones de fitness
        String[] fitnessFunctions = strFitnessFormulas.split(";");

        // 2. Caché externa
        CacheStorage<String, Double[]> externalCache = haveExternalCache ? new HybridCache<>(hybridCacheManager, "ExternalCache", String.class, Double[].class, 1000) : null;

        // 3. Cachés internas
        CacheStorage<String, Double>[] internalCaches = null;
        if (haveInternalCache) {
            internalCaches = new CacheStorage[fitnessFunctions.length];
            for (int i = 0; i < internalCaches.length; i++) {
                internalCaches[i] = new HybridCache<>(hybridCacheManager, fitnessFunctions[i] + "Cache", String.class, Double.class, 1000);
            }
        }

        // 4. Observadores
        String[] strObserversArray = resolvedObservers.split(";");
        this.observers = new ObserverInterface[strObserversArray.length];
        for (int i = 0; i < this.observers.length; i++) {
            this.observers[i] = StaticUtils.getObserverFromString(strObserversArray[i], populationSize, fitnessFunctions, maxEvaluations / populationSize, externalCache, internalCaches, null);
        }

        // Problem
        float genericInitialMinPercBics = genericInitialMinNumBics != -1 ? (float) genericInitialMinNumBics / numericData.length : 0.05f;
        float genericInitialMaxPercBics = genericInitialMaxNumBics != -1 ? (float) genericInitialMaxNumBics / numericData.length : 0.25f;
        RepresentationWrapper representationWrapper = StaticUtils.getRepresentationWrapperFromRepresentation(representation, numericData.length, numericData[0].length, specificNumBiclusters, genericInitialMinPercBics, genericInitialMaxPercBics, summariseIndividualObjectives);
        String resolvedCrossoverOperator = resolveCrossoverOperator(representationWrapper);
        String resolvedMutationOperator = resolveMutationOperator(representationWrapper);
        Problem problem = new ProblemObserver(numericData, types, fitnessFunctions, externalCache, internalCaches, representationWrapper, this.observers);

        // Operators
        // 1. Crossover
        CrossoverOperator<CompositeSolution> crossover = representationWrapper.getCrossoverFromString(resolvedCrossoverOperator, crossoverProbability, (int) Math.round(maxEvaluations * crossoverProbability));
        
        // 2. Mutation
        MutationOperator<CompositeSolution> mutation = representationWrapper.getMutationFromString(resolvedMutationOperator, mutationProbability, maxEvaluations);

        // 3. Selection
        NaryTournamentSelection<CompositeSolution> selection = new BinaryTournamentSelection<>(new RankingAndCrowdingDistanceComparator<>());

        // Algorithm
        AlgorithmResult<CompositeSolution> result = StaticUtils.executeEvolutionaryAlgorithm(
                problem,
                populationSize,
                maxEvaluations,
                strAlgorithm,
                selection,
                crossover,
                mutation,
                numThreads
        );

        // Store population
        this.solutions = result.population;

        // Write output to files if outputFolder is specified
        if (outputFolder != null) {
            // Create output folder
            try {
                Files.createDirectories(Paths.get(outputFolder));
            } catch (IOException ioe) {
                throw new RuntimeException(ioe);
            }

            // Write the evolution of fitness values to an output txt file
            for (ObserverInterface observer : this.observers) {
                observer.writeToFile(outputFolder + "/" + observer.getClass().getSimpleName() + ".csv");
            }

            // Write the data of the last population (pareto front approximation)
            new SolutionListVARWithHeader(result.population, strFitnessFormulas.split(";"), representationWrapper.getVarLabels())
                    .setVarFileOutputContext(new DefaultFileOutputContext(outputFolder + "/VAR.csv", ","))
                    .setFunFileOutputContext(new DefaultFileOutputContext(outputFolder + "/FUN.csv", ","))
                    .print();

            // Write translated VAR
            new SolutionListTranslatedVAR(representationWrapper)
                .printTranslatedVAR(outputFolder + "/VAR-translated.csv", result.population);
        }


        System.out.println("Threads used: " + numThreads);
        System.out.println("Total execution time: " + result.computingTime + "ms");
            
        if (numThreads > 1) {
            System.exit(0);
        }
    }

    public static void main(String[] args) {
        CommandLine commandLine = new CommandLine(new Runner());
        commandLine.execute(args);
    }

    private String resolveCrossoverOperator(RepresentationWrapper representationWrapper) {
        return isAuto(strCrossoverOperator) ? representationWrapper.getDefaultCrossoverOperator() : strCrossoverOperator;
    }

    private String resolveMutationOperator(RepresentationWrapper representationWrapper) {
        return isAuto(strMutationOperator) ? representationWrapper.getDefaultMutationOperator() : strMutationOperator;
    }

    private String resolveObservers() {
        return isAuto(strObservers) ? getDefaultObservers(representation, haveExternalCache, haveInternalCache) : strObservers;
    }

    private static boolean isAuto(String value) {
        return value == null || AUTO.equalsIgnoreCase(value.trim());
    }

    static String getDefaultObservers(Representation representation, boolean haveExternalCache, boolean haveInternalCache) {
        List<String> observerNames = new ArrayList<>();
        observerNames.add("FitnessEvolutionMinObserver");

        if (representation == Representation.GENERIC) {
            observerNames.add("BiclusterCountObserver");
        }
        if (haveExternalCache) {
            observerNames.add("ExternalCacheObserver");
        }
        if (haveInternalCache) {
            observerNames.add("InternalCacheObserver");
        }

        return String.join(";", observerNames);
    }

    public List<CompositeSolution> getSolutions() {
        return solutions;
    }

    public ObserverInterface[] getObservers() {
        return observers;
    }
}
