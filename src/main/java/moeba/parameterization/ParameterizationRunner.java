package moeba.parameterization;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.uma.jmetal.experimental.componentbasedalgorithm.algorithm.singleobjective.geneticalgorithm.GeneticAlgorithm;
import org.uma.jmetal.experimental.componentbasedalgorithm.catalogue.replacement.Replacement;
import org.uma.jmetal.experimental.componentbasedalgorithm.catalogue.replacement.impl.MuPlusLambdaReplacement;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.crossover.impl.CompositeCrossover;
import org.uma.jmetal.operator.crossover.impl.IntegerSBXCrossover;
import org.uma.jmetal.operator.crossover.impl.SBXCrossover;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.mutation.impl.CompositeMutation;
import org.uma.jmetal.operator.mutation.impl.IntegerPolynomialMutation;
import org.uma.jmetal.operator.mutation.impl.PolynomialMutation;
import org.uma.jmetal.operator.selection.impl.BinaryTournamentSelection;
import org.uma.jmetal.operator.selection.impl.NaryTournamentSelection;
import org.uma.jmetal.util.SolutionListUtils;
import org.uma.jmetal.util.comparator.ObjectiveComparator;
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import org.uma.jmetal.util.termination.Termination;
import org.uma.jmetal.util.termination.impl.TerminationByEvaluations;

import moeba.Representation;
import moeba.StaticUtils;
import moeba.StaticUtils.AlgorithmResult;
import moeba.algorithm.AsyncMultiThreadGAParents;
import moeba.parameterization.operator.ParameterizationCrossover;
import moeba.parameterization.operator.ParameterizationMutation;
import moeba.parameterization.problem.ParameterizationProblem;
import moeba.parameterization.problem.impl.CEProblem;
import moeba.representationwrapper.RepresentationWrapper;
import moeba.utils.observer.ProblemObserver.ObserverInterface;
import moeba.utils.observer.impl.NumEvaluationsObserver;
import moeba.utils.observer.impl.ParameterizationFunVarCleanerObserver;
import moeba.utils.output.SolutionListTranslatedVAR;
import moeba.utils.output.SolutionListVARWithHeader;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

@Command(name = "RunnerParameterization", description = "Run the MOEBA parameterization", mixinStandardHelpOptions = true, showDefaultValues = true, sortOptions = false)
public class ParameterizationRunner implements Runnable {

    @Option(names = {"--input-benchmark-folder"}, description = "Path to the input benchmark folder", required = true)
    private File inputBenchmarkFolder;

    @Option(names = {"--internal-evaluations"}, description = "Number of internal evaluations", required = true)
    private int internalEvaluations;

    @Option(names = {"--unsupervised-conf-file"}, description = "Path to the unsupervised configuration file", required = true)
    private File unsupervisedConfFile;

    @Option(names = {"--external-unsupervised-evaluations"}, description = "Number of external unsupervised evaluations", required = true)
    private int externalUnsupervisedEvaluations;

    @Option(names = {"--external-unsupervised-population-size"}, description = "External unsupervised population size", required = true)
    private int externalUnsupervisedPopulationSize;

    @Option(names = {"--supervised-conf-file"}, description = "Path to the supervised configuration file", required = true)
    private File supervisedConfFile;

    @Option(names = {"--external-supervised-evaluations"}, description = "Number of external supervised evaluations", required = true)
    private int externalSupervisedEvaluations;

    @Option(names = {"--external-supervised-population-size"}, description = "External supervised population size", required = true)
    private int externalSupervisedPopulationSize;

    @Option(names = {"--num-threads"}, description = "Number of threads. Default: All")
    private int numThreads = Runtime.getRuntime().availableProcessors();

    @Option(names = {"--output-folder"}, description = "Path to the output folder", defaultValue = "./MOEBA-Parameterization/")
    private String outputFolder;

    @Override
    public void run() {
        // Config sort. NOTE: https://github.com/jMetal/jMetal/issues/446
        System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");

        // Read input benchmark folder
        Set<String> validPrefixes = detectValidPrefixes(inputBenchmarkFolder);
        System.out.println("Number of problems detected: " + validPrefixes.size());
        System.out.println("Prefixes: " + String.join(", ", validPrefixes));

        // Get parameterization exercises
        ParameterizationExercise unsupervisedParameterizationExercise = new ParameterizationExercise(unsupervisedConfFile, externalUnsupervisedEvaluations, externalUnsupervisedPopulationSize, numThreads/validPrefixes.size());
        ParameterizationExercise supervisedParameterizationExercise = new ParameterizationExercise(supervisedConfFile, externalSupervisedEvaluations, externalSupervisedPopulationSize, unsupervisedParameterizationExercise.numThreads == 1 ? numThreads/validPrefixes.size() : 1);
        
        // Get parameterization problem
        String staticConf = "--max-evaluations=" + internalEvaluations + " --num-threads=1" + " --observers=FitnessEvolutionMinObserver";
        ObserverInterface[] externalSupervisedObservers = supervisedParameterizationExercise.populationSize == 1 ? new ObserverInterface[0] : new ObserverInterface[]{new NumEvaluationsObserver(externalSupervisedPopulationSize), new ParameterizationFunVarCleanerObserver(supervisedParameterizationExercise, 1)};
        String externalUnsupervisedObservers = unsupervisedParameterizationExercise.populationSize == 1 ? "" : "ParameterizationFunVarCleanerObserver";
        CEProblem ceproblem = new CEProblem(supervisedParameterizationExercise, staticConf, externalSupervisedObservers, externalUnsupervisedObservers, validPrefixes.toArray(new String[validPrefixes.size()]), unsupervisedParameterizationExercise, outputFolder + "/ce-tmp/");

        // Run parameterization
        AlgorithmResult<ParameterizationSolution> result = ParameterizationRunner.executeParameterizationAlgorithm(ceproblem);

        // Save solutions recursively
        // 1. Create output folder
        try {
            Files.createDirectories(Paths.get(outputFolder));
        } catch (IOException ioe) {
            throw new RuntimeException(ioe);
        }

        // 2. Get CE solutions
        ParameterizationSolution ceSolution = result.population.get(0);
        double ceFUN = ceSolution.objectives()[0];
        String ceVAR = Arrays.toString(ParameterizationProblem.preprocessArguments(supervisedParameterizationExercise.getArgsFromSolution(ceSolution).split(" ")));
        saveToFile(outputFolder, "ceSolution.txt", "ceFUN: " + ceFUN + "\n" + "ceVAR: " + ceVAR);
        for (ObserverInterface observer : externalSupervisedObservers) {
            observer.writeToFile(outputFolder + "/ce-" + observer.getClass().getSimpleName() + ".csv");
        }

        // 3. Get HV solutions
        ParameterizationSolution hvSolution = ceSolution.subPopulations.get(0).get(0);
        double hvFUN = hvSolution.objectives()[0];
        String hvVAR = Arrays.toString(ParameterizationProblem.preprocessArguments(unsupervisedParameterizationExercise.getArgsFromSolution(hvSolution).split(" ")));
        saveToFile(outputFolder, "hvSolution.txt", "hvFUN: " + hvFUN + "\n" + "hvVAR: " + hvVAR);
        for (ObserverInterface observer : ceSolution.subObservers.get(0)) {
            observer.writeToFile(outputFolder + "/hv-" + observer.getClass().getSimpleName() + ".csv");
        }

        // 4. Get MOEBA solutions
        String representation = supervisedParameterizationExercise.getValueOfArg("--representation", ceSolution);
        List<String> objectives = new ArrayList<>();
        int cntO = 0;
        String value = "";
        while (value != null) {
            value = supervisedParameterizationExercise.getValueOfArg("--comb--str-fitness-functions--" + cntO, ceSolution);
            if (value != null && !value.equals("''")) {
                for(String o : value.split(";")) {
                    objectives.add(o);
                }
            }
            cntO++;
        }

        String[] prefixes = validPrefixes.toArray(new String[validPrefixes.size()]);
        for (int i = 0; i < prefixes.length; i++) {
            
            String tag = Paths.get(prefixes[i]).getFileName().toString();
            int tagIndex = hvSolution.tags.indexOf(tag);
            List<ParameterizationSolution> moebaSolutions = hvSolution.subPopulations.get(tagIndex);
            RepresentationWrapper wrapper = StaticUtils.getRepresentationWrapperFromRepresentation(Representation.valueOf(representation), ceproblem.numRows[i], ceproblem.numCols[i], 0, 0, 0, null);

            // Write the data of the moeba population
            new SolutionListVARWithHeader(moebaSolutions, objectives.toArray(new String[0]), wrapper.getVarLabels())
                    .setVarFileOutputContext(new DefaultFileOutputContext(outputFolder + "/VAR-" + tag + ".csv", ","))
                    .setFunFileOutputContext(new DefaultFileOutputContext(outputFolder + "/FUN-" + tag + ".csv", ","))
                    .print();

            // Write translated VAR
            new SolutionListTranslatedVAR(wrapper)
                .printTranslatedVAR(outputFolder + "/VAR-translated-" + tag + ".csv", moebaSolutions);

            // Write observers registered data
            for (ObserverInterface observer : hvSolution.subObservers.get(tagIndex)) {
                observer.writeToFile(outputFolder + "/moeba-" + observer.getClass().getSimpleName() + "-" + tag + ".csv");
            }
        }

        System.out.println("Threads used: " + numThreads);
        System.out.println("Total execution time: " + result.computingTime + "ms");
            
        System.exit(0);

    }

    /**
     * Saves the given content to a file in the specified folder.
     *
     * @param folder   the folder where the file will be saved
     * @param fileName the name of the file
     * @param content  the content to be saved
     */
    private static void saveToFile(String folder, String fileName, String content) {
        try {
            // Get the path of the file to be created
            Path filePath = Paths.get(folder, fileName);

            // Write the content to the file
            Files.write(filePath, content.getBytes());
        } catch (IOException e) {
            // Print the stack trace if an error occurs
            e.printStackTrace();
        }
    }

    /**
     * Executes the parameterization algorithm.
     *
     * @param problem the parameterization problem
     * @return the result of the parameterization algorithm
     */
    public static AlgorithmResult<ParameterizationSolution> executeParameterizationAlgorithm(ParameterizationProblem problem) {

        // Get the exercise
        ParameterizationExercise exercise = problem.getParameterizationExercise();

        // Create the crossover operator.
        // The crossover operator is a composite crossover that combines
        // SBX crossover and integer SBX crossover.
        CrossoverOperator<ParameterizationSolution> crossover =
                new ParameterizationCrossover(new CompositeCrossover(Arrays.asList(
                        new SBXCrossover(1.0, 20.0),
                        new IntegerSBXCrossover(1.0, 20.0)
                )));

        // Create the mutation operator.
        // The mutation operator is a composite mutation that combines
        // polynomial mutation and integer polynomial mutation.
        MutationOperator<ParameterizationSolution> mutation =
                new ParameterizationMutation(new CompositeMutation(Arrays.asList(
                        new PolynomialMutation(0.1, 20.0),
                        new IntegerPolynomialMutation(0.1, 2.0)
                )));

        // Execute the evolutionary algorithm.
        // The algorithm is a genetic algorithm (GA) running in a single or parallel thread.
        long computingTime;
        List<ParameterizationSolution> population;
        Termination termination = new TerminationByEvaluations(exercise.evaluations);
        Replacement<ParameterizationSolution> replacement = new MuPlusLambdaReplacement<>(new ObjectiveComparator<>(0));
        NaryTournamentSelection<ParameterizationSolution> selection = new BinaryTournamentSelection<>(new RankingAndCrowdingDistanceComparator<>());

        if (exercise.numThreads == 1) {
            // Instantiates and executes a single-threaded genetic algorithm
            GeneticAlgorithm<ParameterizationSolution> algorithm = new GeneticAlgorithm<>(
                    problem,
                    exercise.populationSize,
                    exercise.populationSize,
                    selection,
                    crossover,
                    mutation,
                    termination);

            algorithm.run();
            computingTime = algorithm.getTotalComputingTime();
            population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());

        } else {
            // Instantiates and executes an asynchronous parallel genetic algorithm
            long initTime = System.currentTimeMillis();

            AsyncMultiThreadGAParents<ParameterizationSolution> algorithm = new AsyncMultiThreadGAParents<>(
                    exercise.numThreads,
                    problem,
                    exercise.populationSize,
                    crossover,
                    mutation,
                    selection,
                    replacement,
                    exercise.evaluations);

            algorithm.run();
            long endTime = System.currentTimeMillis();
            computingTime = endTime - initTime;
            population = SolutionListUtils.getNonDominatedSolutions(algorithm.getResult());
        } 

        // Return the result of the parameterization algorithm.
        return new AlgorithmResult<>(computingTime, population);
    }


    /**
     * Detects the valid prefixes in the input benchmark folder.
     *
     * @param benchFolder the input benchmark folder
     * @return a set of valid prefixes
     * @throws IllegalArgumentException if the input benchmark folder is not found
     */
    public static Set<String> detectValidPrefixes(File benchFolder) {

        // Check if the input benchmark folder exists
        if (!benchFolder.isDirectory()) {
            throw new IllegalArgumentException("Input benchmark folder not found: " + benchFolder);
        }

        // Map to store the prefixes and the found file types
        Map<String, Set<String>> prefixMap = new HashMap<>();

        // List all the files in the directory
        File[] files = benchFolder.listFiles();

        // Iterate over each file and group them by prefix
        for (File file : files) {
            if (file.isFile()) {
                String fileName = file.getName();

                // Check if the file name ends with "-data.csv"
                if (fileName.endsWith("-data.csv")) {
                    String prefix = fileName.substring(0, fileName.length() - "-data.csv".length());
                    prefixMap.computeIfAbsent(prefix, k -> new HashSet<>()).add("data");
                }

                // Check if the file name ends with "-translated.csv"
                else if (fileName.endsWith("-translated.csv")) {
                    String prefix = fileName.substring(0, fileName.length() - "-translated.csv".length());
                    prefixMap.computeIfAbsent(prefix, k -> new HashSet<>()).add("translated");
                }

                // Check if the file name ends with "-types.json"
                else if (fileName.endsWith("-types.json")) {
                    String prefix = fileName.substring(0, fileName.length() - "-types.json".length());
                    prefixMap.computeIfAbsent(prefix, k -> new HashSet<>()).add("types");
                }
            }
        }

        // Find all the prefixes that have all the three types of files
        Set<String> validPrefixes = new HashSet<>();
        for (Map.Entry<String, Set<String>> entry : prefixMap.entrySet()) {
            if (entry.getValue().contains("data") && entry.getValue().contains("translated") && entry.getValue().contains("types")) {
                validPrefixes.add(benchFolder + "/" + entry.getKey());
            }
        }

        return validPrefixes;
    }


    public static void main(String[] args) {
        CommandLine commandLine = new CommandLine(new ParameterizationRunner());
        commandLine.execute(args);
    }
    
}
