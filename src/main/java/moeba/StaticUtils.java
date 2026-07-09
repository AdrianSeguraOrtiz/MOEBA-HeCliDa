package moeba;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiFunction;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.ArrayList;
import java.util.Enumeration;

import moeba.algorithm.AsyncMultiThreadGAParents;
import moeba.algorithm.AsyncMultiThreadNSGAIIParents;
import moeba.algorithm.AsyncMultiThreadNSGAIIParentsExternalFile;
import moeba.fitnessfunction.FitnessFunction;
import moeba.fitnessfunction.GenericBiclusterFitnessFunction;
import moeba.fitnessfunction.GlobalFitnessFunction;
import moeba.fitnessfunction.IndividualBiclusterFitnessFunction;
import moeba.fitnessfunction.impl.BiclusterSizeNormComp;
import moeba.fitnessfunction.impl.BiclusterSizeNumBicsNormComp;
import moeba.fitnessfunction.impl.BiclusterVarianceNorm;
import moeba.fitnessfunction.impl.MeanSquaredResidueNorm;
import moeba.fitnessfunction.impl.RowVarianceNormComp;
import moeba.fitnessfunction.impl.coexpression.RegulatoryCoherenceNormComp;
import moeba.parameterization.ParameterizationExercise;
import moeba.fitnessfunction.impl.DistanceBetweenBiclustersNormComp;
import moeba.representationwrapper.RepresentationWrapper;
import moeba.representationwrapper.impl.GenericRepresentationWrapper;
import moeba.representationwrapper.impl.IndividualRepresentationWrapper;
import moeba.representationwrapper.impl.SpecificRepresentationWrapper;
import moeba.utils.observer.ProblemObserver.ObserverInterface;
import moeba.utils.observer.impl.BiclusterCountObserver;
import moeba.utils.observer.impl.ExternalCacheObserver;
import moeba.utils.observer.impl.FitnessEvolutionAvgObserver;
import moeba.utils.observer.impl.FitnessEvolutionMaxObserver;
import moeba.utils.observer.impl.FitnessEvolutionMinObserver;
import moeba.utils.observer.impl.InternalCacheObserver;
import moeba.utils.observer.impl.NumEvaluationsObserver;
import moeba.utils.observer.impl.ParameterizationFunVarCleanerObserver;
import moeba.utils.storage.CacheStorage;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.algorithm.multiobjective.ibea.IBEA;
import org.uma.jmetal.algorithm.multiobjective.mocell.MOCell;
import org.uma.jmetal.algorithm.multiobjective.mosa.MOSA;
import org.uma.jmetal.algorithm.multiobjective.mosa.cooling.impl.Exponential;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder;
import org.uma.jmetal.algorithm.multiobjective.nsgaiii.NSGAIII;
import org.uma.jmetal.algorithm.multiobjective.nsgaiii.NSGAIIIBuilder;
import org.uma.jmetal.algorithm.multiobjective.smsemoa.SMSEMOA;
import org.uma.jmetal.algorithm.multiobjective.spea2.SPEA2;
import org.uma.jmetal.example.AlgorithmRunner;
import org.uma.jmetal.experimental.componentbasedalgorithm.algorithm.multiobjective.moead.MOEAD;
import org.uma.jmetal.experimental.componentbasedalgorithm.algorithm.singleobjective.geneticalgorithm.GeneticAlgorithm;
import org.uma.jmetal.experimental.componentbasedalgorithm.catalogue.replacement.impl.MuPlusLambdaReplacement;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.selection.impl.NaryTournamentSelection;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.util.SolutionListUtils;
import org.uma.jmetal.util.aggregativefunction.AggregativeFunction;
import org.uma.jmetal.util.aggregativefunction.impl.PenaltyBoundaryIntersection;
import org.uma.jmetal.util.aggregativefunction.impl.Tschebyscheff;
import org.uma.jmetal.util.aggregativefunction.impl.WeightedSum;
import org.uma.jmetal.util.archive.BoundedArchive;
import org.uma.jmetal.util.archive.impl.CrowdingDistanceArchive;
import org.uma.jmetal.util.archive.impl.GenericBoundedArchive;
import org.uma.jmetal.util.archive.impl.HypervolumeArchive;
import org.uma.jmetal.util.archive.impl.SpatialSpreadDeviationArchive;
import org.uma.jmetal.util.comparator.DominanceComparator;
import org.uma.jmetal.util.comparator.ObjectiveComparator;
import org.uma.jmetal.util.densityestimator.impl.CrowdingDistanceDensityEstimator;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.legacy.qualityindicator.impl.hypervolume.impl.PISAHypervolume;
import org.uma.jmetal.util.neighborhood.Neighborhood;
import org.uma.jmetal.util.neighborhood.impl.C25;
import org.uma.jmetal.util.neighborhood.impl.C9;
import org.uma.jmetal.util.neighborhood.impl.L13;
import org.uma.jmetal.util.neighborhood.impl.L25;
import org.uma.jmetal.util.neighborhood.impl.L5;
import org.uma.jmetal.util.termination.Termination;
import org.uma.jmetal.util.termination.impl.TerminationByEvaluations;
import org.uma.jmetal.problem.Problem;

@SuppressWarnings("deprecation")
public final class StaticUtils {

    private static class ObjectivesParams {
        public double[][] data;
        public Class<?>[] types;
        public CacheStorage<String, Double> cache;
        public String summariseIndividualObjectives;

        public ObjectivesParams(double[][] data, Class<?>[] types, CacheStorage<String, Double> cache, String summariseIndividualObjectives) {
            this.data = data;
            this.types = types;
            this.cache = cache;
            this.summariseIndividualObjectives = summariseIndividualObjectives;
        }
    }

    public static final class ObjectiveDefinition {
        private final String identifier;
        private final Class<? extends FitnessFunction> fitnessFunctionType;
        private final BiFunction<String, ObjectivesParams, FitnessFunction> factory;

        private ObjectiveDefinition(
            String identifier,
            Class<? extends FitnessFunction> fitnessFunctionType,
            BiFunction<String, ObjectivesParams, FitnessFunction> factory
        ) {
            this.identifier = identifier;
            this.fitnessFunctionType = fitnessFunctionType;
            this.factory = factory;
        }

        public String getIdentifier() {
            return identifier;
        }

        public Class<? extends FitnessFunction> getFitnessFunctionType() {
            return fitnessFunctionType;
        }

        private boolean matches(String str) {
            return str.toLowerCase().startsWith(identifier);
        }

        public FitnessFunction create(String str, double[][] data, Class<?>[] types, CacheStorage<String, Double> cache, String summariseIndividualObjectives) {
            return create(str, new ObjectivesParams(data, types, cache, summariseIndividualObjectives));
        }

        private FitnessFunction create(String str, ObjectivesParams op) {
            return factory.apply(str, op);
        }
    }

    static final Map<String, ObjectiveDefinition> OBJECTIVES_MAP = new LinkedHashMap<>();
    static {
        OBJECTIVES_MAP.put("biclustersizenormcomp", new ObjectiveDefinition("biclustersizenormcomp", IndividualBiclusterFitnessFunction.class, (str, op) -> {
            Map<String, String> subParams = getSubParams("biclustersizenormcomp", str);
            String sumIndObjs = StaticUtils.getOne("biclustersizenormcomp", subParams, "summariseindividualobjectives", op.summariseIndividualObjectives);
            Double rowsWeight = Double.parseDouble(StaticUtils.getOne("biclustersizenormcomp", subParams, "rowsweight", "0.5"));
            return new BiclusterSizeNormComp(op.data, op.types, op.cache, sumIndObjs, rowsWeight);
        }));

        OBJECTIVES_MAP.put("biclustervariancenorm", new ObjectiveDefinition("biclustervariancenorm", IndividualBiclusterFitnessFunction.class, (str, op) -> {
            Map<String, String> subParams = getSubParams("biclustervariancenorm", str);
            String sumIndObjs = StaticUtils.getOne("biclustervariancenorm", subParams, "summariseindividualobjectives", op.summariseIndividualObjectives);
            return new BiclusterVarianceNorm(op.data, op.types, op.cache, sumIndObjs);
        }));

        OBJECTIVES_MAP.put("rowvariancenormcomp", new ObjectiveDefinition("rowvariancenormcomp", IndividualBiclusterFitnessFunction.class, (str, op) -> {
            Map<String, String> subParams = getSubParams("rowvariancenormcomp", str);
            String sumIndObjs = StaticUtils.getOne("rowvariancenormcomp", subParams, "summariseindividualobjectives", op.summariseIndividualObjectives);
            return new RowVarianceNormComp(op.data, op.types, op.cache, sumIndObjs);
        }));

        OBJECTIVES_MAP.put("meansquaredresiduenorm", new ObjectiveDefinition("meansquaredresiduenorm", IndividualBiclusterFitnessFunction.class, (str, op) -> {
            Map<String, String> subParams = getSubParams("meansquaredresiduenorm", str);
            String sumIndObjs = StaticUtils.getOne("meansquaredresiduenorm", subParams, "summariseindividualobjectives", op.summariseIndividualObjectives);
            return new MeanSquaredResidueNorm(op.data, op.types, op.cache, sumIndObjs);
        }));

        OBJECTIVES_MAP.put("distancebetweenbiclustersnormcomp", new ObjectiveDefinition("distancebetweenbiclustersnormcomp", GenericBiclusterFitnessFunction.class, (str, op) -> {
            Map<String, String> subParams = getSubParams("distancebetweenbiclustersnormcomp", str);
            String sumIndObjs = StaticUtils.getOne("distancebetweenbiclustersnormcomp", subParams, "summariseindividualobjectives", op.summariseIndividualObjectives);
            return new DistanceBetweenBiclustersNormComp(op.data, op.types, op.cache, sumIndObjs);
        }));

        OBJECTIVES_MAP.put("regulatorycoherencenormcomp", new ObjectiveDefinition("regulatorycoherencenormcomp", GlobalFitnessFunction.class, (str, op) -> {
            return new RegulatoryCoherenceNormComp(op.data, op.types);
        }));

        OBJECTIVES_MAP.put("biclustersizenumbicsnormcomp", new ObjectiveDefinition("biclustersizenumbicsnormcomp", GenericBiclusterFitnessFunction.class, (str, op) -> {
            Map<String, String> subParams = getSubParams("biclustersizenumbicsnormcomp", str);
            String sumIndObjs = StaticUtils.getOne("biclustersizenumbicsnormcomp", subParams, "summariseindividualobjectives", op.summariseIndividualObjectives);
            Double rowsWeight = Double.parseDouble(StaticUtils.getOne("biclustersizenumbicsnormcomp", subParams, "rowsweight", "0.5"));
            double coherenceWeight = Double.parseDouble(StaticUtils.getOne("biclustersizenumbicsnormcomp", subParams, "coherenceweight", "0.5"));
            return new BiclusterSizeNumBicsNormComp(op.data, op.types, op.cache, sumIndObjs, rowsWeight, coherenceWeight);
        }));
    }

    /**
     * Returns the objective definition matching a given identifier string.
     *
     * @param str the identifier string for the fitness function
     * @return the objective definition
     * @throws RuntimeException if the fitness function is not implemented
     */
    public static ObjectiveDefinition getObjectiveDefinitionFromString(String str) {
        for (ObjectiveDefinition definition : OBJECTIVES_MAP.values()) {
            if (definition.matches(str)) {
                return definition;
            }
        }
        throw new RuntimeException("Fitness function not implemented: " + str);
    }

    /**
     * Returns a FitnessFunction object based on a given identifier string.
     *
     * @param str the identifier string for the fitness function
     * @param data the 2D array of data
     * @param types the array of data types
     * @param cache the internal cache of the fitness function
     * @param summariseIndividualObjectives the way to summarise the overall quality of the solutions from the individual quality of their biclusters
     * @return a FitnessFunction object
     * @throws RuntimeException if the fitness function is not implemented
     */
    public static FitnessFunction getFitnessFunctionFromString(String str, double[][] data, Class<?>[] types, CacheStorage<String, Double> cache, String summariseIndividualObjectives) {
        return getObjectiveDefinitionFromString(str).create(str, data, types, cache, summariseIndividualObjectives);
    }

    /**
     * Returns an observer from its string representation.
     *
     * @param str string representation of the desired observer
     * @param populationSize the size of the population in the genetic algorithm
     * @param fitnessFunctions an array of strings representing the fitness functions
     * @param numGenerations the number of generations of the genetic algorithm
     * @param externalCache a map of external cache data
     * @param internalCaches an array of internal caches
     * @param exercise the parameterization exercise
     * @return an observer
     * @throws RuntimeException if the observer is not implemented
     */
    public static ObserverInterface getObserverFromString(String str, int populationSize, String[] fitnessFunctions, int numGenerations, CacheStorage<String, Double[]> externalCache, CacheStorage<String, Double>[] internalCaches, ParameterizationExercise exercise) {
        ObserverInterface res;
        switch (str.toLowerCase()) {
            case "biclustercountobserver":
                res = new BiclusterCountObserver(populationSize, numGenerations);
                break;
            case "externalcacheobserver":
                res = new ExternalCacheObserver(populationSize, externalCache);
                break;
            case "fitnessevolutionminobserver":
                res = new FitnessEvolutionMinObserver(populationSize, fitnessFunctions.length);
                break;
            case "fitnessevolutionmaxobserver":
                res = new FitnessEvolutionMaxObserver(populationSize, fitnessFunctions.length);
                break;
            case "fitnessevolutionavgobserver":
                res = new FitnessEvolutionAvgObserver(populationSize, fitnessFunctions.length);
                break;
            case "internalcacheobserver":
                res = new InternalCacheObserver(populationSize, fitnessFunctions, internalCaches);
                break;
            case "numevaluationsobserver":
                res = new NumEvaluationsObserver(populationSize);
                break;
            case "parameterizationfunvarcleanerobserver":
                res = new ParameterizationFunVarCleanerObserver(exercise, fitnessFunctions.length);
                break;
            default:
                throw new RuntimeException("The observer " + str + " is not implemented.");
        }
        return res;
    }


    /**
     * Returns a new representation wrapper instance based on the given representation.
     * 
     * @param rep The representation to create a wrapper for.
     * @param numRows The number of rows in the dataset.
     * @param numCols The number of columns in the dataset.
     * @param specificNumBiclusters The number of biclusters in the specific representation.
     * @param genericInitialMinPercBics The initial minimum percentage of biclusters in the generic representation.
     * @param genericInitialMaxPercBics The initial maximum percentage of biclusters in the generic representation.
     * @param summariseIndividualObjectives The way to summarise the overall quality of the solutions from the individual quality of their biclusters.
     * @return A new representation wrapper instance.
     * @throws RuntimeException If the given representation is not implemented.
     */
    public static RepresentationWrapper getRepresentationWrapperFromRepresentation(Representation rep, int numRows, int numCols, int specificNumBiclusters, float genericInitialMinPercBics, float genericInitialMaxPercBics, String summariseIndividualObjectives) {
        RepresentationWrapper res;
        switch (rep) {
            case GENERIC:
                res = new GenericRepresentationWrapper(numRows, numCols, genericInitialMinPercBics, genericInitialMaxPercBics, summariseIndividualObjectives);
                break;
            case SPECIFIC:
                res = new SpecificRepresentationWrapper(numRows, numCols, specificNumBiclusters, summariseIndividualObjectives);
                break;
            case INDIVIDUAL:
                res = new IndividualRepresentationWrapper(numRows, numCols);
                break;
            default:
                throw new RuntimeException("The representation " + rep.name() + " is not implemented.");
        }
        return res;
    }

    /**
     * Converts a CSV file to a bidimensional array of strings.
     * 
     * @param inputDataset The input CSV file to read
     * @return A bidimensional array of strings, where each row corresponds to a line in the CSV file and each column
     *         corresponds to a value in that line.
     * @throws IOException If there is an error reading the input CSV file
     */
    public static String[][] csvToStringMatrix(File inputDataset) throws IOException {
        // Read all lines from the CSV file sequentially
        List<String> lines = Files.readAllLines(inputDataset.toPath());
        
        // Process lines in parallel to convert them to an array of objects
        String[][] matrix = lines.parallelStream()
            .skip(1) // Ignore the first line as it contains the heading
            .map(line -> line.split(",")) // Assumes comma as CSV separator
            .toArray(String[][]::new); // Convert list to a bidimensional array of strings
        
        return matrix;
    }

    /**
     * Converts a JSON file to an array of Class objects based on the provided column names.
     * 
     * @param inputJsonFile The input JSON file to read column types from
     * @param columnNames The names of the columns to map to Class objects
     * @return An array of Class objects corresponding to the column names
     * @throws IOException If there is an error reading the input JSON file
     * @throws IllegalArgumentException If a column name is not found in the JSON file or if an unsupported type is encountered
     */
    public static Class<?>[] jsonToClassArray(File inputJsonFile, String[] columnNames) throws IOException, IllegalArgumentException {
        // Read column types from the input JSON file
        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, String> columnTypes = objectMapper.readValue(inputJsonFile, new TypeReference<Map<String, String>>() {});

        // Define mapping of column type strings to Class objects
        Map<String, Class<?>> typeMapping = new HashMap<>();
        typeMapping.put("string", String.class);
        typeMapping.put("int", Integer.class);
        typeMapping.put("double", Double.class);
        typeMapping.put("float", Float.class);
        typeMapping.put("float64", Float.class);
        typeMapping.put("boolean", Boolean.class);

        // Initialize an array to hold the Class objects for the columns
        Class<?>[] columnClasses = new Class<?>[columnNames.length];

        // Map column types to Class objects based on the provided column names
        for (int i = 0; i < columnNames.length; i++) {
            String columnName = columnNames[i];
            String type = columnTypes.get(columnName);
            if (type == null) {
                throw new IllegalArgumentException("Column '" + columnName + "' not found in the JSON file.");
            }
            
            Class<?> columnClass = typeMapping.get(type.toLowerCase());
            if (columnClass != null) {
                columnClasses[i] = columnClass;
            } else {
                throw new IllegalArgumentException("Unsupported type: " + type);
            }
        }

        return columnClasses;
    }

    /**
     * Converts a data matrix with string values to a matrix with numeric values.
     * Supported types: string, float, double, int, boolean.
     * Strings are converted to categorical values, with each unique string value
     * being assigned a unique numeric value.
     *
     * @param data The data matrix with string values
     * @param types The types of each column in the data matrix
     * @param numThreads The number of threads to use for parallel processing
     * @return A matrix with numeric values
     */
    public static double[][] dataToNumericMatrix(String[][] data, Class<?>[] types, int numThreads) {
        double[][] numericData = new double[data.length][data[0].length];
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        // Convert each column in parallel
        for (int j = 0; j < data[0].length; j++) {
            final int colIndex = j;
            executor.execute(() -> {
                // Determine how to convert the data in this column based on its type
                if (types[colIndex] == Float.class || types[colIndex] == Double.class
                        || types[colIndex] == Integer.class) {
                    // Column is numeric, directly parse the values as doubles
                    for (int i = 0; i < data.length; i++) {
                        numericData[i][colIndex] = Double.parseDouble(data[i][colIndex]);
                    }
                } else if (types[colIndex] == Boolean.class) {
                    // Column is boolean, map "Yes" to 1.0 and other values to 0.0
                    for (int i = 0; i < data.length; i++) {
                        numericData[i][colIndex] = data[i][colIndex].equalsIgnoreCase("Yes") ? 1.0 : 0.0;
                    }
                } else if (types[colIndex] == String.class) {
                    // Column is string, convert to categorical values
                    Map<String, Double> categoryToNumber = new HashMap<>();
                    double categoryIndex = 0.0;
                    for (int i = 0; i < data.length; i++) {
                        String category = data[i][colIndex];
                        if (!categoryToNumber.containsKey(category)) {
                            categoryToNumber.put(category, categoryIndex);
                            categoryIndex++;
                        }
                        numericData[i][colIndex] = categoryToNumber.get(category);
                    }
                }
            });
        }
        executor.shutdown();
        return numericData;
    }

    /**
     * Represents the result of an evolutionary algorithm execution.
     */
    public static class AlgorithmResult<S extends Solution<?>> {
        // Execution time of the algorithm
        public long computingTime;
        // Population at the last iteration of the algorithm
        public List<S> population;

        /**
         * Constructs an instance of AlgorithmResult.
         * 
         * @param computingTime The total execution time of the algorithm
         * @param population The population at the last iteration of the algorithm
         */
        public AlgorithmResult(long computingTime, List<S> population) {
            this.computingTime = computingTime;
            this.population = population;
        }
    }

    /**
     * Executes an evolutionary algorithm based on the specified parameters.
     * 
     * @param problem The problem to be solved by the evolutionary algorithm
     * @param populationSize The size of the population
     * @param maxEvaluations The maximum number of evaluations
     * @param strAlgorithm The name of the algorithm to execute
     * @param selection The selection operator
     * @param crossover The crossover operator
     * @param mutation The mutation operator
     * @param numThreads The number of threads to use for parallel execution
     * @return An AlgorithmResult containing the total execution time and the final population
     * @throws IllegalArgumentException If the specified algorithm is not supported for the problem type
     */
    public static AlgorithmResult<CompositeSolution> executeEvolutionaryAlgorithm(
            Problem<CompositeSolution> problem,
            int populationSize,
            int maxEvaluations,
            String strAlgorithm,
            NaryTournamentSelection<CompositeSolution> selection,
            CrossoverOperator<CompositeSolution> crossover,
            MutationOperator<CompositeSolution> mutation,
            int numThreads) {

        long computingTime;
        List<CompositeSolution> population;

        // Defines the termination condition for the algorithm
        Termination termination = new TerminationByEvaluations(maxEvaluations);
        
        // Set offspring population size to be equal to the population size
        int offspringPopulationSize = populationSize;

        // Executes the algorithm based on the problem's number of objectives and specified algorithm name
        if (problem.getNumberOfObjectives() == 1) {
            // Single-objective problem logic
            if (strAlgorithm.startsWith("GA-SingleThread")) {
                // Instantiates and executes a single-threaded genetic algorithm
                GeneticAlgorithm<CompositeSolution> algorithm = new GeneticAlgorithm<>(
                        problem,
                        populationSize,
                        offspringPopulationSize,
                        selection,
                        crossover,
                        mutation,
                        termination);

                algorithm.run();
                computingTime = algorithm.getTotalComputingTime();
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());

            } else if (strAlgorithm.startsWith("GA-AsyncParallel")) {
                // Instantiates and executes an asynchronous parallel genetic algorithm
                long initTime = System.currentTimeMillis();

                AsyncMultiThreadGAParents<CompositeSolution> algorithm = new AsyncMultiThreadGAParents<>(
                        numThreads,
                        problem,
                        populationSize,
                        crossover,
                        mutation,
                        selection,
                        new MuPlusLambdaReplacement<>(new ObjectiveComparator<>(0)),
                        termination);

                algorithm.run();
                long endTime = System.currentTimeMillis();
                computingTime = endTime - initTime;
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());

            } else {
                throw new IllegalArgumentException("The algorithm " + strAlgorithm + " is not available for single-objective problems.");
            }
        } else {
            // Multi-objective problem logic
            if (strAlgorithm.startsWith("NSGAII-SingleThread")) {
                // Instantiates and executes a single-threaded NSGA-II algorithm
                Algorithm<List<CompositeSolution>> algorithm = new NSGAIIBuilder<>(problem, crossover, mutation, populationSize)
                        .setSelectionOperator(selection)
                        .setMaxEvaluations(maxEvaluations)
                        .build();

                AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm).execute();
                computingTime = algorithmRunner.getComputingTime();
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());

            } else if (strAlgorithm.startsWith("NSGAII-AsyncParallel")) {
                // Instantiates and executes an asynchronous parallel NSGA-II algorithm
                long initTime = System.currentTimeMillis();

                AsyncMultiThreadNSGAIIParents<CompositeSolution> algorithm = new AsyncMultiThreadNSGAIIParents<>(
                        numThreads,
                        problem,
                        populationSize,
                        crossover,
                        mutation,
                        termination);

                algorithm.run();
                long endTime = System.currentTimeMillis();
                computingTime = endTime - initTime;
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());

            } else if (strAlgorithm.startsWith("NSGAII-ExternalFile-AsyncParallel")) {
                // Instantiates and executes an asynchronous parallel NSGA-II algorithm with external file support
                long initTime = System.currentTimeMillis();

                AsyncMultiThreadNSGAIIParentsExternalFile<CompositeSolution> algorithm = new AsyncMultiThreadNSGAIIParentsExternalFile<>(
                        numThreads,
                        problem,
                        populationSize,
                        crossover,
                        mutation,
                        termination);

                algorithm.run();
                long endTime = System.currentTimeMillis();
                computingTime = endTime - initTime;
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());

            } else if (strAlgorithm.startsWith("MOEAD-SingleThread")) {
                // Get the directory where the weight vector files are located
                String weightVectorDirectory = null;
                try {
                    weightVectorDirectory = createTempDirectoryFromResource("weightVectorFiles/moead/").toString();
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (URISyntaxException e) {
                    e.printStackTrace();
                }

                // Check if the weight vector file exists
                File weightVectorFile = new File(weightVectorDirectory + "/W" + problem.getNumberOfObjectives() + "D_" + populationSize + ".dat");
                if (!weightVectorFile.exists()) {
                    System.out.println("The weight vector file " + weightVectorFile.getAbsolutePath() + " does not exist.");
                    return new AlgorithmResult<>(0, new ArrayList<>());
                }

                // Get subparameters
                Map<String, String> subParams = StaticUtils.getSubParams("MOEAD-SingleThread", strAlgorithm);
                AggregativeFunction aggregativeFunction = new Tschebyscheff();
                for (Map.Entry<String, String> entry : subParams.entrySet()) {
                    if (entry.getKey().equals("aggregativefunction")) {
                        switch (entry.getValue()) {
                            case "tschebyscheff":
                                aggregativeFunction = new Tschebyscheff();
                                break;
                            case "weightedsum":
                                aggregativeFunction = new WeightedSum();
                                break;
                            case "penaltyboundaryintersection":
                                aggregativeFunction = new PenaltyBoundaryIntersection();
                                break;
                            default:
                                throw new IllegalArgumentException("The aggregative function " + entry.getValue() + " is not implemented.");
                        }
                    }
                }

                // Instantiates and executes a single-threaded MOEAD algorithm
                MOEAD<CompositeSolution> algorithm = new MOEAD<CompositeSolution>(
                    problem, 
                    populationSize, 
                    mutation, 
                    crossover, 
                    aggregativeFunction, 
                    Double.parseDouble(StaticUtils.getOne("MOEAD-SingleThread", subParams, "neighborhoodselectionprobability", "0.1")),
                    Integer.parseInt(StaticUtils.getOne("MOEAD-SingleThread", subParams, "maximumnumberofreplacedsolutions", "2")),
                    Integer.parseInt(StaticUtils.getOne("MOEAD-SingleThread", subParams, "neighborhoodsize", "20")),
                    weightVectorDirectory, 
                    termination
                );

                algorithm.run();
                computingTime = algorithm.getTotalComputingTime();
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());
            
            } else if (strAlgorithm.startsWith("SMS-EMOA-SingleThread")) {
                // Instantiates and executes a single-threaded SMS-EMOA algorithm
                long initTime = System.currentTimeMillis();

                SMSEMOA<CompositeSolution> algorithm = new SMSEMOA<CompositeSolution>(
                    problem,
                    maxEvaluations,
                    populationSize,
                    populationSize,
                    crossover,
                    mutation,
                    selection,
                    new DominanceComparator<>(),
                    new PISAHypervolume<>()
                );


                algorithm.run();
                long endTime = System.currentTimeMillis();
                computingTime = endTime - initTime;
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());
            
            } else if (strAlgorithm.startsWith("MOCell-SingleThread")) {
                // Instantiates and executes a single-threaded MOCell algorithm
                long initTime = System.currentTimeMillis();

                // Adjust population size
                populationSize = nearestPerfectSquare(populationSize);

                // Get subparameters
                Map<String, String> subParams = StaticUtils.getSubParams("MOCell-SingleThread", strAlgorithm);
                BoundedArchive<CompositeSolution> archive = new CrowdingDistanceArchive<>(populationSize);
                Neighborhood<CompositeSolution> neighborhood = new C9<>((int)Math.sqrt(populationSize), (int)Math.sqrt(populationSize));

                for (Map.Entry<String, String> entry : subParams.entrySet()) {
                    if (entry.getKey().equals("archive")) {
                        switch (entry.getValue()) {
                            case "crowdingdistancearchive":
                                archive = new CrowdingDistanceArchive<>(populationSize);
                                break;
                            case "hypervolumearchive":
                                archive = new HypervolumeArchive<>(populationSize, new PISAHypervolume<>());
                                break;
                            case "spatialspreaddeviationarchive":
                                archive = new SpatialSpreadDeviationArchive<>(populationSize);
                                break;
                            default:
                                throw new IllegalArgumentException("The archive " + entry.getValue() + " is not implemented.");
                        }
                    } else if (entry.getKey().equals("neighborhood")) {
                        switch (entry.getValue()) {
                            case "c9":
                                neighborhood = new C9<>((int)Math.sqrt(populationSize), (int)Math.sqrt(populationSize));
                                break;
                            case "c25":
                                neighborhood = new C25<>((int)Math.sqrt(populationSize), (int)Math.sqrt(populationSize));
                                break;
                            case "l5":
                                neighborhood = new L5<>((int)Math.sqrt(populationSize), (int)Math.sqrt(populationSize));
                                break;
                            case "l13":
                                neighborhood = new L13<>((int)Math.sqrt(populationSize), (int)Math.sqrt(populationSize));
                                break;
                            case "l25":
                                neighborhood = new L25<>((int)Math.sqrt(populationSize), (int)Math.sqrt(populationSize));
                                break;
                            default:
                                throw new IllegalArgumentException("The neighborhood " + entry.getValue() + " is not implemented.");
                        }
                    }
                }

                MOCell<CompositeSolution> algorithm = new MOCell<CompositeSolution>(
                    problem,
                    maxEvaluations,
                    populationSize,
                    archive,
                    neighborhood,
                    crossover,
                    mutation,
                    selection,
                    new SequentialSolutionListEvaluator<>()
                );

                algorithm.run();
                long endTime = System.currentTimeMillis();
                computingTime = endTime - initTime;
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());
            
            } else if (strAlgorithm.startsWith("SPEA2-SingleThread")) {
                // Instantiates and executes a single-threaded SPEA2 algorithm
                long initTime = System.currentTimeMillis();

                // Get subparameters
                Map<String, String> subParams = StaticUtils.getSubParams("SPEA2-SingleThread", strAlgorithm);

                SPEA2<CompositeSolution> algorithm = new SPEA2<CompositeSolution>(
                   problem,
                   maxEvaluations / populationSize,
                   populationSize,
                   crossover,
                   mutation,
                   selection,
                   new SequentialSolutionListEvaluator<>(),
                   Integer.parseInt(StaticUtils.getOne("SPEA2-SingleThread", subParams, "k", "1"))
                );

                algorithm.run();
                long endTime = System.currentTimeMillis();
                computingTime = endTime - initTime;
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());
            
            } else if (strAlgorithm.startsWith("IBEA-SingleThread")) {
                // Instantiates and executes a single-threaded IBEA algorithm
                long initTime = System.currentTimeMillis();

                IBEA<CompositeSolution> algorithm = new IBEA<CompositeSolution>(
                    problem,
                    populationSize,
                    populationSize,
                    maxEvaluations,
                    selection,
                    crossover,
                    mutation
                );

                algorithm.run();
                long endTime = System.currentTimeMillis();
                computingTime = endTime - initTime;
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());
            
            } else if (strAlgorithm.startsWith("NSGAIII-SingleThread")) {
                // Get subparameters
                Map<String, String> subParams = StaticUtils.getSubParams("NSGAIII-SingleThread", strAlgorithm);
                int numberOfDivisions = Integer.parseInt(StaticUtils.getOne("NSGAIII-SingleThread", subParams, "numberofdivisions", "12"));
                
                // Instantiates and executes a single-threaded NSGAIII algorithm
                NSGAIII<CompositeSolution> algorithm = new NSGAIIIBuilder<>(problem)
                    .setCrossoverOperator(crossover)
                    .setMutationOperator(mutation)
                    .setSelectionOperator(selection)
                    .setPopulationSize(populationSize)
                    .setMaxIterations(maxEvaluations / (int) CombinatoricsUtils.binomialCoefficient(numberOfDivisions + problem.getNumberOfObjectives() - 1, problem.getNumberOfObjectives() - 1))
                    .setNumberOfDivisions(numberOfDivisions)
                    .build();

                AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm).execute();
                computingTime = algorithmRunner.getComputingTime();
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());
            
            } else if (strAlgorithm.startsWith("MOSA-SingleThread")) {
                // Instantiates and executes a single-threaded MOSA algorithm
                long initTime = System.currentTimeMillis();

                // Get subparameters
                Map<String, String> subParams = StaticUtils.getSubParams("MOSA-SingleThread", strAlgorithm);

                MOSA<CompositeSolution> algorithm = new MOSA<CompositeSolution>(
                    problem,
                    maxEvaluations,
                    new GenericBoundedArchive<>(populationSize, new CrowdingDistanceDensityEstimator<>()),
                    mutation,
                    Double.parseDouble(StaticUtils.getOne("MOSA-SingleThread", subParams, "initialtemperature", "1.0")), 
                    new Exponential(0.95)
                );  

                algorithm.run();
                long endTime = System.currentTimeMillis();
                computingTime = endTime - initTime;
                population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());
            
            } else {
                throw new IllegalArgumentException("The algorithm " + strAlgorithm + " is not available for multi-objective problems.");
            }
        }

        return new AlgorithmResult<>(computingTime, population);
    } 

    /**
     * Finds the nearest perfect square to a given number.
     * 
     * @param number The number to find the nearest perfect square for.
     * @return The nearest perfect square.
     */
    public static int nearestPerfectSquare(int number) {
        // Calculate the floor of the square root of the number
        int lower = (int) Math.floor(Math.sqrt(number));
        
        // Calculate the square of the lower value
        int lowerSquare = lower * lower;
        
        // Calculate the square of the lower value plus one
        int upper = lower + 1;
        int upperSquare = upper * upper;

        // Compare which perfect square is closer to the number
        // and return the square of the closer value
        if ((number - lowerSquare) < (upperSquare - number)) {
            return lowerSquare;
        } else {
            return upperSquare;
        }
    }

    /**
     * Converts a bicluster to a string representation
     * 
     * @param bicArray the bicluster
     * @return the string representation of the bicluster
     */
    public static String biclusterToString(ArrayList<Integer>[] bicArray) {
        StringBuilder builder = new StringBuilder();
        builder.append("(rows: [");
        for (int i = 0; i < bicArray[0].size(); i++) {
            builder.append(bicArray[0].get(i).toString());
            if (i < bicArray[0].size() - 1) {
                builder.append(" ");
            }
        }
        builder.append("] cols: [");
        for (int i = 0; i < bicArray[1].size(); i++) {
            builder.append(bicArray[1].get(i).toString());
            if (i < bicArray[1].size() - 1) {
                builder.append(" ");
            }
        }
        builder.append("])");
        return builder.toString();
    }

    /**
     * Converts a list of biclusters to a string representation. The string representation
     * is a comma separated list of the string representation of each bicluster.
     * 
     * @param biclusters The list of biclusters to convert to a string
     * @return The string representation of the list of biclusters
     */
    public static String biclustersToString(ArrayList<ArrayList<Integer>[]> biclusters) {
        StringBuilder builder = new StringBuilder();
        for (ArrayList<Integer>[] bicluster : biclusters) {
            builder.append(biclusterToString(bicluster));
            builder.append(", ");
        }
        String res = builder.toString();
        return res.substring(0, res.length() - 2); // Remove last comma and space
    }

    /**
     * Converts a string representation of biclusters into a list of biclusters.
     * Each bicluster is represented as an ArrayList of two ArrayLists of Integers,
     * where the first list represents rows and the second list represents columns.
     *
     * @param biclustersString The string representation of biclusters.
     * @return The ArrayList of biclusters.
     */
    @SuppressWarnings("unchecked")
    public static ArrayList<ArrayList<Integer>[]> stringToBiclusters(String biclustersString) {
        ArrayList<ArrayList<Integer>[]> biclusters = new ArrayList<>();
        Pattern bicPattern = Pattern.compile("Bicluster\\d+: \\(rows: \\[([\\d ]+)\\] cols: \\[([\\d ]+)\\]\\)");
        Matcher matcher = bicPattern.matcher(biclustersString);

        while (matcher.find()) {
            ArrayList<Integer>[] bicArray = new ArrayList[2];
            bicArray[0] = stringToIntegerList(matcher.group(1));
            bicArray[1] = stringToIntegerList(matcher.group(2));
            biclusters.add(bicArray);
        }

        return biclusters;
    }

    /**
     * Converts a space-separated string of numbers into an ArrayList of Integers.
     *
     * @param numbers String containing space-separated numbers.
     * @return ArrayList of Integers parsed from the string.
     */
    private static ArrayList<Integer> stringToIntegerList(String numbers) {
        ArrayList<Integer> list = new ArrayList<>();
        String[] tokens = numbers.trim().split(" ");
        for (String token : tokens) {
            if (!token.isEmpty()) {
                list.add(Integer.parseInt(token));
            }
        }
        return list;
    }


    /**
     * Loads biclusters from a specified CSV file.
     * This method reads a file line by line, converts each line into a list of biclusters,
     * and returns a list of these biclusters. Each bicluster is represented as an ArrayList of ArrayLists of arrays of Integers.
     *
     * @param file The file to read biclusters from.
     * @return A list containing the biclusters loaded from the file.
     * @throws IOException if an I/O error occurs reading from the file.
     */
    public static ArrayList<ArrayList<ArrayList<Integer>[]>> loadBiclusters(File file) {
        ArrayList<ArrayList<ArrayList<Integer>[]>> biclusters = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                biclusters.add(StaticUtils.stringToBiclusters(line));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return biclusters;
    }

    /**
     * Loads gold standard biclusters from a specified CSV file.
     * This method reads the first line of the file and converts it into a list of biclusters,
     * which is returned. This list is intended to serve as the "gold standard" for validation.
     *
     * @param file The file to read the gold standard biclusters from.
     * @return A list of gold standard biclusters.
     * @throws IOException if an I/O error occurs reading from the file.
     */
    public static ArrayList<ArrayList<Integer>[]> loadGoldStandardBiclusters(File file) {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line = reader.readLine();
            return StaticUtils.stringToBiclusters(line);
        } catch (IOException e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Creates a temporary directory and copies the resources from the specified resource path into it.
     * The resource path can be a directory or a JAR file. If the resource path is a JAR file,
     * the resources are extracted from the JAR file and copied into the temporary directory.
     *
     * @param resourcePath The path of the resources to be copied.
     * @return The path of the temporary directory.
     * @throws IOException if an I/O error occurs while creating the temporary directory or copying the resources.
     * @throws URISyntaxException if the resource path is not a valid URI.
     */
    public static Path createTempDirectoryFromResource(String resourcePath) throws IOException, URISyntaxException {
        // Create a temporary directory
        Path tempDir = Files.createTempDirectory("temp_resources_");

        // Get the URL of the resource
        URL resourceUrl = StaticUtils.class.getClassLoader().getResource(resourcePath);
        if (resourceUrl == null) {
            throw new IOException("Resource path not found: " + resourcePath);
        }

        if (resourceUrl.getProtocol().equals("jar")) {
            // If the resources are inside a JAR file
            String jarPath = resourceUrl.getPath().substring(5, resourceUrl.getPath().indexOf("!"));

            // Extract the resources from the JAR file and copy them into the temporary directory
            try (JarFile jar = new JarFile(jarPath)) {
                Enumeration<JarEntry> entries = jar.entries();
                while (entries.hasMoreElements()) {
                    JarEntry entry = entries.nextElement();
                    if (entry.getName().startsWith(resourcePath) && !entry.isDirectory()) {
                        File tempFile = new File(tempDir.toFile(),
                                entry.getName().substring(resourcePath.length()));
                        if (!tempFile.getParentFile().exists()) {
                            tempFile.getParentFile().mkdirs();
                        }
                        try (InputStream inputStream = jar.getInputStream(entry);
                             FileOutputStream outputStream = new FileOutputStream(tempFile)) {
                            byte[] buffer = new byte[1024];
                            int bytesRead;
                            while ((bytesRead = inputStream.read(buffer)) != -1) {
                                outputStream.write(buffer, 0, bytesRead);
                            }
                        }
                    }
                }
            }
        } else {
            // If the resources are in the file system
            Path sourcePath = Paths.get(resourceUrl.toURI());
            Files.walk(sourcePath)
                 .forEach(source -> {
                     Path destination = tempDir.resolve(sourcePath.relativize(source).toString());
                     try {
                         Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
                     } catch (IOException e) {
                         e.printStackTrace();
                     }
                 });
        }
        return tempDir;
    }

    /**
     * Extracts sub-parameters from a given parameter string.
     * 
     * @param param The parameter string to extract sub-parameters from.
     * @param input The input string containing the sub-parameters.
     * @return A map of sub-parameters with their corresponding values.
     */
    public static Map<String, String> getSubParams(String param, String input) {
        Map<String, String> subParams = new HashMap<>();
        param = param.toLowerCase();
        input = input.toLowerCase();
        
        if (input.matches(param + "((.*))")) {
            String[] strSubparams = input.split("[()=, ]");
            
            // Iterate through the array in pairs, with each pair representing a sub-parameter and its value
            for (int i = 1; i < strSubparams.length; i+=2) {
                subParams.put(strSubparams[i], strSubparams[i+1]);
            }
        }

        return subParams;
    }

    /**
     * Retrieves the value of a specific key from a map of sub-parameters, or returns a default value if the key is not found.
     * 
     * @param param        The parameter string to extract sub-parameters from. This is included in the warning message.
     * @param subParams    The map of sub-parameters to retrieve the value from.
     * @param key          The key of the sub-parameter to retrieve the value from.
     * @param defaultValue The default value to return if the key is not found.
     * @return             The value of the sub-parameter if it exists, otherwise the default value.
     */
    public static String getOne(String param, Map<String, String> subParams, String key, String defaultValue) {
        if (subParams.containsKey(key)) {
            return subParams.get(key);
        } else {
            // If the key is not found, print a warning message and return the default value
            System.out.println("Warning: " + key + " not found in sub-parameters of " + param + ". Using default value: " + defaultValue);
            return defaultValue;
        }
    }
}
