package moeba.utils.observer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

import moeba.ColumnType;
import moeba.fitnessfunction.FitnessFunction;
import moeba.fitnessfunction.impl.BiclusterSizeNormComp;
import moeba.representationwrapper.impl.IndividualRepresentationWrapper;
import org.junit.jupiter.api.Test;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;

class ProblemObserverTest {

    @Test
    void notificationsAreSerializedDuringParallelEvaluation() throws Exception {
        double[][] data = new double[4][4];
        ColumnType[] types = new ColumnType[] {
            ColumnType.numeric(),
            ColumnType.numeric(),
            ColumnType.numeric(),
            ColumnType.numeric()
        };
        ConcurrentAccessObserver observer = new ConcurrentAccessObserver();
        ProblemObserver problem = new ProblemObserver(
            new FitnessFunction[] {new BiclusterSizeNormComp(data, types, null, null, 0.5)},
            null,
            null,
            new IndividualRepresentationWrapper(4, 4),
            new ProblemObserver.ObserverInterface[] {observer}
        );
        List<CompositeSolution> solutions = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            solutions.add(problem.createSolution());
        }

        ExecutorService executor = Executors.newFixedThreadPool(8);
        try {
            List<Future<?>> evaluations = new ArrayList<>();
            for (CompositeSolution solution : solutions) {
                evaluations.add(executor.submit(() -> problem.evaluate(solution)));
            }
            for (Future<?> evaluation : evaluations) {
                evaluation.get();
            }
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }

        assertEquals(solutions.size(), observer.count.get());
        assertFalse(observer.concurrentAccessDetected.get());
    }

    private static final class ConcurrentAccessObserver implements ProblemObserver.ObserverInterface {
        private final AtomicBoolean registering = new AtomicBoolean();
        private final AtomicBoolean concurrentAccessDetected = new AtomicBoolean();
        private final AtomicInteger count = new AtomicInteger();

        @Override
        public void register(CompositeSolution result) {
            if (!registering.compareAndSet(false, true)) {
                concurrentAccessDetected.set(true);
            }
            LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(1));
            count.incrementAndGet();
            registering.set(false);
        }

        @Override
        public void writeToFile(String file) {
            // No output is needed for this concurrency test.
        }
    }
}
