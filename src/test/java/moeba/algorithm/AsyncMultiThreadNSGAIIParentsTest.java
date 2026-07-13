package moeba.algorithm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import moeba.ColumnType;
import moeba.fitnessfunction.FitnessFunction;
import moeba.fitnessfunction.impl.BiclusterSizeNormComp;
import moeba.representationwrapper.impl.IndividualRepresentationWrapper;
import moeba.utils.observer.ProblemObserver;
import org.junit.jupiter.api.Test;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;

class AsyncMultiThreadNSGAIIParentsTest {
    private static final int POPULATION_SIZE = 2;
    private static final int MAXIMUM_EVALUATIONS = 5;

    @Test
    void honorsEvaluationBudgetWithoutReevaluatingInitialSolutionsAndStopsWorkers() {
        IndividualRepresentationWrapper wrapper = new IndividualRepresentationWrapper(4, 4);
        CountingObserver observer = new CountingObserver();
        ProblemObserver problem = problem(wrapper, observer);
        CountingCrossover crossover = new CountingCrossover(crossover(wrapper));
        AsyncMultiThreadNSGAIIParents<CompositeSolution> algorithm = new AsyncMultiThreadNSGAIIParents<>(
            2,
            problem,
            POPULATION_SIZE,
            crossover,
            mutation(wrapper),
            MAXIMUM_EVALUATIONS
        );

        algorithm.run();

        assertEquals(MAXIMUM_EVALUATIONS, observer.count.get());
        assertEquals(MAXIMUM_EVALUATIONS, observer.uniqueSolutions.size());
        assertTrue(crossover.count.get() > 0);
        assertEquals(POPULATION_SIZE, algorithm.getResult().size());
        assertNoWorkersRunning();
    }

    @Test
    void propagatesEvaluationFailuresAndStopsWorkers() {
        IndividualRepresentationWrapper wrapper = new IndividualRepresentationWrapper(4, 4);
        ProblemObserver failingProblem = new ProblemObserver(
            objectives(),
            null,
            null,
            wrapper,
            new ProblemObserver.ObserverInterface[0]
        ) {
            @Override
            public CompositeSolution evaluate(CompositeSolution solution) {
                throw new IllegalArgumentException("evaluation failed");
            }
        };
        AsyncMultiThreadNSGAIIParents<CompositeSolution> algorithm = new AsyncMultiThreadNSGAIIParents<>(
            2,
            failingProblem,
            POPULATION_SIZE,
            crossover(wrapper),
            mutation(wrapper),
            MAXIMUM_EVALUATIONS
        );

        IllegalStateException exception = assertThrows(IllegalStateException.class, algorithm::run);

        assertEquals("evaluation failed", exception.getCause().getMessage());
        assertNoWorkersRunning();
    }

    @Test
    void honorsBudgetSmallerThanWorkerCount() {
        IndividualRepresentationWrapper wrapper = new IndividualRepresentationWrapper(4, 4);
        CountingObserver observer = new CountingObserver();
        AsyncMultiThreadNSGAIIParents<CompositeSolution> algorithm = new AsyncMultiThreadNSGAIIParents<>(
            4,
            problem(wrapper, observer),
            POPULATION_SIZE,
            crossover(wrapper),
            mutation(wrapper),
            1
        );

        algorithm.run();

        assertEquals(1, observer.count.get());
        assertEquals(1, algorithm.getResult().size());
        assertNoWorkersRunning();
    }

    private static ProblemObserver problem(
        IndividualRepresentationWrapper wrapper,
        ProblemObserver.ObserverInterface observer
    ) {
        return new ProblemObserver(
            objectives(),
            null,
            null,
            wrapper,
            new ProblemObserver.ObserverInterface[] {observer}
        );
    }

    private static FitnessFunction[] objectives() {
        double[][] data = new double[4][4];
        ColumnType[] types = new ColumnType[] {
            ColumnType.numeric(),
            ColumnType.numeric(),
            ColumnType.numeric(),
            ColumnType.numeric()
        };
        return new FitnessFunction[] {
            new BiclusterSizeNormComp(data, types, null, null, 0.5),
            new BiclusterSizeNormComp(data, types, null, null, 0.75)
        };
    }

    private static CrossoverOperator<CompositeSolution> crossover(IndividualRepresentationWrapper wrapper) {
        return wrapper.getCrossoverFromString(
            wrapper.getDefaultCrossoverOperator(),
            0.9,
            MAXIMUM_EVALUATIONS
        );
    }

    private static MutationOperator<CompositeSolution> mutation(IndividualRepresentationWrapper wrapper) {
        return wrapper.getMutationFromString(
            wrapper.getDefaultMutationOperator(),
            "0.1",
            MAXIMUM_EVALUATIONS
        );
    }

    private static void assertNoWorkersRunning() {
        assertFalse(Thread.getAllStackTraces().keySet().stream()
            .anyMatch(thread -> thread.isAlive() && thread.getName().startsWith("moeba-async-worker-")));
    }

    private static final class CountingObserver implements ProblemObserver.ObserverInterface {
        private final AtomicInteger count = new AtomicInteger();
        private final Set<CompositeSolution> uniqueSolutions = Collections.newSetFromMap(new IdentityHashMap<>());

        @Override
        public void register(CompositeSolution result) {
            count.incrementAndGet();
            uniqueSolutions.add(result);
        }

        @Override
        public void writeToFile(String file) {
            // No output is needed for this evaluation-budget test.
        }
    }

    private static final class CountingCrossover implements CrossoverOperator<CompositeSolution> {
        private final CrossoverOperator<CompositeSolution> delegate;
        private final AtomicInteger count = new AtomicInteger();

        private CountingCrossover(CrossoverOperator<CompositeSolution> delegate) {
            this.delegate = delegate;
        }

        @Override
        public List<CompositeSolution> execute(List<CompositeSolution> source) {
            count.incrementAndGet();
            return delegate.execute(source);
        }

        @Override
        public double getCrossoverProbability() {
            return delegate.getCrossoverProbability();
        }

        @Override
        public int getNumberOfRequiredParents() {
            return delegate.getNumberOfRequiredParents();
        }

        @Override
        public int getNumberOfGeneratedChildren() {
            return delegate.getNumberOfGeneratedChildren();
        }
    }
}
