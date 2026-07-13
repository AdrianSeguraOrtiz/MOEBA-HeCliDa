package moeba.algorithm;

import org.uma.jmetal.experimental.componentbasedalgorithm.catalogue.replacement.Replacement;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.selection.SelectionOperator;
import org.uma.jmetal.parallel.asynchronous.multithreaded.Master;
import org.uma.jmetal.parallel.asynchronous.task.ParallelTask;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.util.errorchecking.Check;
import org.uma.jmetal.util.observable.Observable;
import org.uma.jmetal.util.observable.impl.DefaultObservable;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.util.termination.Termination;
import org.uma.jmetal.util.termination.impl.TerminationByEvaluations;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

/**
 * Implements an asynchronous multi-threaded genetic algorithm (GA) with parallel evaluation of parents.
 * This class extends the Master class provided by the jMetal framework to support parallel execution of tasks
 * in a genetic algorithm context. It handles the creation, execution, and updating of tasks representing
 * genetic operations such as mutation and crossover in an asynchronous manner.
 *
 * @param <S> Solution type that extends the Solution interface, representing the type of solutions the GA will work with.
 */
public class AsyncMultiThreadGAParents<S extends Solution<?>>
    extends Master<ParallelTask<S>, List<S>> {
  private final Problem<S> problem;
  private final CrossoverOperator<S> crossover;
  private final MutationOperator<S> mutation;
  private final SelectionOperator<List<S>, S> selection;
  private final Replacement<S> replacement;
  private List<S> population = new ArrayList<>();
  private final int populationSize;
  private int evaluations = 0;
  private int submittedTasks = 0;
  private final int maximumScheduledEvaluations;
  private final Termination termination;
  private long initTime;

  private final Map<String, Object> attributes;
  private final Observable<Map<String, Object>> observable;
  private final List<Thread> workers = new ArrayList<>();
  private final AtomicReference<Throwable> workerFailure = new AtomicReference<>();
  private final ParallelTask<S> stopTask = new StopTask<>();

  private static final class StopTask<S> implements ParallelTask<S> {
    @Override
    public S getContents() {
      return null;
    }

    @Override
    public long getIdentifier() {
      return Long.MIN_VALUE;
    }
  }

  /**
   * Constructs an AsyncMultiThreadGAParents object with the specified parameters.
   *
   * @param numberOfCores   The number of cores to use for parallel task execution.
   * @param problem         The problem to be solved by the GA.
   * @param populationSize  The size of the population.
   * @param crossover       The crossover operator to be used.
   * @param mutation        The mutation operator to be used.
   * @param selection       The selection operator to be used for selecting parents.
   * @param replacement     The replacement strategy to be used for creating the new population.
   * @param maximumEvaluations Maximum number of solutions to evaluate.
   */
  public AsyncMultiThreadGAParents(
      int numberOfCores,
      Problem<S> problem,
      int populationSize,
      CrossoverOperator<S> crossover,
      MutationOperator<S> mutation,
      SelectionOperator<List<S>, S> selection,
      Replacement<S> replacement,
      int maximumEvaluations) {
    this(
        numberOfCores,
        problem,
        populationSize,
        crossover,
        mutation,
        selection,
        replacement,
        new TerminationByEvaluations(maximumEvaluations),
        maximumEvaluations
    );
  }

  /**
   * Creates an asynchronous algorithm with a general jMetal termination condition.
   * Evaluations already running when the condition is met are allowed to finish.
   */
  public AsyncMultiThreadGAParents(
      int numberOfCores,
      Problem<S> problem,
      int populationSize,
      CrossoverOperator<S> crossover,
      MutationOperator<S> mutation,
      SelectionOperator<List<S>, S> selection,
      Replacement<S> replacement,
      Termination termination) {
    this(
        numberOfCores,
        problem,
        populationSize,
        crossover,
        mutation,
        selection,
        replacement,
        termination,
        Integer.MAX_VALUE
    );
  }

  private AsyncMultiThreadGAParents(
      int numberOfCores,
      Problem<S> problem,
      int populationSize,
      CrossoverOperator<S> crossover,
      MutationOperator<S> mutation,
      SelectionOperator<List<S>, S> selection,
      Replacement<S> replacement,
      Termination termination,
      int maximumScheduledEvaluations) {
    super(numberOfCores);
    Check.that(numberOfCores > 0, "The number of cores must be positive");
    Check.that(populationSize > 0, "The population size must be positive");
    Check.that(maximumScheduledEvaluations > 0, "The maximum number of evaluations must be positive");
    Check.notNull(termination);
    this.problem = problem;
    this.crossover = crossover;
    this.mutation = mutation;
    this.populationSize = populationSize;
    this.maximumScheduledEvaluations = maximumScheduledEvaluations;
    this.termination = termination;
    this.selection = selection;
    this.replacement = replacement;

    attributes = new HashMap<>();
    observable = new DefaultObservable<>("Observable");
  }

  /**
   * Creates worker threads for parallel task execution.
   *
   * @param numberOfCores The number of worker threads to create.
   * @param problem       The problem to be solved, used in the evaluation of solutions.
   */
  private void createWorkers(int numberOfCores, Problem<S> problem) {
    IntStream.range(0, numberOfCores).forEach(i -> {
      Thread worker = new Thread(
          () -> {
            while (!Thread.currentThread().isInterrupted()) {
              ParallelTask<S> task;
              try {
                task = pendingTaskQueue.take();
              } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                return;
              }
              if (task == stopTask) {
                return;
              }

              try {
                problem.evaluate(task.getContents());
              } catch (Throwable failure) {
                workerFailure.compareAndSet(null, failure);
              }

              completedTaskQueue.add(ParallelTask.create(task.getIdentifier(), task.getContents()));
            }
          },
          "moeba-async-worker-" + i
      );
      worker.setDaemon(true);
      workers.add(worker);
      worker.start();
    });
  }

  /**
   * Generates a unique identifier for a task.
   *
   * @return An integer representing the task identifier.
   */
  private int createTaskIdentifier() {
    return JMetalRandom.getInstance().nextInt(0, 1000000000);
  }

  /**
   * Initializes the progress attributes at the beginning of the algorithm's execution.
   */
  @Override
  public void initProgress() {
    attributes.put("EVALUATIONS", evaluations);
    attributes.put("POPULATION", population);
    attributes.put("COMPUTING_TIME", System.currentTimeMillis() - initTime);

    observable.setChanged();
    observable.notifyObservers(attributes);
  }

  /**
   * Updates progress attributes during the algorithm's execution.
   */
  @Override
  public void updateProgress() {
    attributes.put("EVALUATIONS", evaluations);
    attributes.put("POPULATION", population);
    attributes.put("COMPUTING_TIME", System.currentTimeMillis() - initTime);
    attributes.put("BEST_SOLUTION", population.get(0));

    observable.setChanged();
    observable.notifyObservers(attributes);
  }

  /**
   * Creates the initial set of tasks representing the initial population.
   *
   * @return A list of ParallelTask objects representing the initial solutions to be evaluated.
   */
  @Override
  public List<ParallelTask<S>> createInitialTasks() {
    List<S> initialPopulation = new ArrayList<>();
    List<ParallelTask<S>> initialTaskList = new ArrayList<>();
    IntStream.range(0, populationSize)
        .forEach(i -> initialPopulation.add(problem.createSolution()));
    initialPopulation.forEach(
        solution -> {
          int taskId = JMetalRandom.getInstance().nextInt(0, 1000);
          initialTaskList.add(ParallelTask.create(taskId, solution));
        });

    return initialTaskList;
  }

  /**
   * Submits the initial tasks for execution.
   *
   * @param initialTaskList The list of initial tasks to be submitted.
   */
  @Override
  public void submitInitialTasks(List<ParallelTask<S>> initialTaskList) {
    int initialTasksToSubmit = Math.min(
        Math.min(numberOfCores, initialTaskList.size()),
        maximumScheduledEvaluations - submittedTasks
    );
    for (int i = 0; i < initialTasksToSubmit; i++) {
      submitTask(initialTaskList.remove(0));
    }
  }

  @Override
  public ParallelTask<S> waitForComputedTask() {
    ParallelTask<S> task;
    try {
      task = completedTaskQueue.take();
    } catch (InterruptedException exception) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted while waiting for a parallel evaluation", exception);
    }
    Throwable failure = workerFailure.get();
    if (failure != null) {
      throw new IllegalStateException("Parallel solution evaluation failed", failure);
    }
    return task;
  }

  /**
   * Processes a computed task, updating the population with the new solution.
   *
   * @param task The computed ParallelTask containing a solution.
   */
  @Override
  public void processComputedTask(ParallelTask<S> task) {
    evaluations++;
    if (population.size() < populationSize) {
      population.add(task.getContents());
    } else {
      List<S> offspringPopulation = new ArrayList<>(1);
      offspringPopulation.add(task.getContents());

      population = replacement.replace(population, offspringPopulation);
      Check.that(population.size() == populationSize, "The population size is incorrect");
    }
  }

  /**
   * Submits a task for execution.
   *
   * @param task The ParallelTask to be submitted.
   */
  @Override
  public void submitTask(ParallelTask<S> task) {
    if (submittedTasks < maximumScheduledEvaluations) {
      pendingTaskQueue.add(task);
      submittedTasks++;
    }
  }

  /**
   * Creates a new task for execution. This method is used when creating tasks dynamically during the algorithm's execution.
   *
   * @return A new ParallelTask object.
   */
  @Override
  public ParallelTask<S> createNewTask() {
    int numberOfParents = crossover.getNumberOfRequiredParents();
    if (population.size() >= numberOfParents) {
      List<S> parents = new ArrayList<>(numberOfParents);
      for (int i = 0; i < numberOfParents; i++) {
        parents.add(selection.execute(population));
      }

      List<S> offspring = crossover.execute(parents);

      mutation.execute(offspring.get(0));

      return ParallelTask.create(createTaskIdentifier(), offspring.get(0));
    } else {
      return ParallelTask.create(createTaskIdentifier(), problem.createSolution());
    }
  }

  /**
   * Checks whether the termination condition of the algorithm has not been met.
   *
   * @return True if the termination condition is not met, false otherwise.
   */
  @Override
  public boolean stoppingConditionIsNotMet() {
    return !termination.isMet(attributes);
  }

  /**
   * Starts the execution of the asynchronous multi-threaded genetic algorithm.
   */
  @Override
  public void run() {
    initTime = System.currentTimeMillis();
    try {
      createWorkers(numberOfCores, problem);
      List<ParallelTask<S>> initialTasks = createInitialTasks();
      submitInitialTasks(initialTasks);
      initProgress();
      while (stoppingConditionIsNotMet()) {
        processComputedTask(waitForComputedTask());
        updateProgress();
        if (stoppingConditionIsNotMet() && submittedTasks < maximumScheduledEvaluations) {
          if (thereAreInitialTasksPending(initialTasks)) {
            submitTask(getInitialTask(initialTasks));
          } else {
            submitTask(createNewTask());
          }
        }
      }
    } finally {
      stopWorkers();
    }
  }

  private void stopWorkers() {
    pendingTaskQueue.clear();
    for (int i = 0; i < workers.size(); i++) {
      pendingTaskQueue.add(stopTask);
    }
    boolean interrupted = false;
    for (Thread worker : workers) {
      boolean joined = false;
      while (!joined) {
        try {
          worker.join();
          joined = true;
        } catch (InterruptedException exception) {
          interrupted = true;
        }
      }
    }
    pendingTaskQueue.clear();
    completedTaskQueue.clear();
    workers.clear();
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  /**
   * Returns the result of the genetic algorithm, which is the current population.
   *
   * @return A list of solutions representing the current population.
   */
  @Override
  public List<S> getResult() {
    return population;
  }

  /**
   * Gets the observable object for this algorithm.
   *
   * @return The Observable object that allows observers to track changes.
   */
  public Observable<Map<String, Object>> getObservable() {
    return observable;
  }
}
