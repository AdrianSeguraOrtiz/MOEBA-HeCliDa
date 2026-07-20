package moeba;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.selection.impl.BinaryTournamentSelection;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.solution.compositesolution.CompositeSolution;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.bounds.Bounds;
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator;

class ConstrainedAlgorithmExecutionTest {
    private static final int POPULATION_SIZE = 4;
    private static final int MAX_EVALUATIONS = 8;
    private static final List<String> AUDITED_ALGORITHMS = Arrays.asList(
        "NSGAII-SingleThread",
        "MOCell-SingleThread",
        "SPEA2-SingleThread",
        "NSGAIII-SingleThread(numberofdivisions=3)"
    );

    @Test
    void auditedAlgorithmsKeepFeasibleSolutionsAheadOfBetterObjectives() {
        for (String algorithm : AUDITED_ALGORITHMS) {
            DeterministicConstrainedProblem problem =
                new DeterministicConstrainedProblem(true);

            List<CompositeSolution> result = execute(algorithm, problem);

            assertEquals(MAX_EVALUATIONS, problem.evaluations(), algorithm);
            assertFalse(result.isEmpty(), algorithm);
            assertTrue(
                result.stream().allMatch(solution -> solution.constraints()[0] == 0.0),
                algorithm
            );
            assertTrue(
                result.stream().allMatch(solution -> solution.objectives()[0] == 0.2),
                algorithm
            );
        }
    }

    @Test
    void auditedAlgorithmsPreferTheSmallestViolationWhenNoneIsFeasible() {
        for (String algorithm : AUDITED_ALGORITHMS) {
            DeterministicConstrainedProblem problem =
                new DeterministicConstrainedProblem(false);

            List<CompositeSolution> result = execute(algorithm, problem);

            assertFalse(result.isEmpty(), algorithm);
            assertTrue(
                result.stream().allMatch(solution -> solution.constraints()[0] == -0.1),
                algorithm
            );
        }
    }

    @Test
    void spea2RejectsUnsupportedNeighborhoodParameter() {
        DeterministicConstrainedProblem problem = new DeterministicConstrainedProblem(true);

        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> execute("SPEA2-SingleThread(k=2)", problem)
        );

        assertTrue(exception.getMessage().contains("only applies k=1"));
    }

    private static List<CompositeSolution> execute(
            String algorithm,
            Problem<CompositeSolution> problem) {
        return StaticUtils.executeEvolutionaryAlgorithm(
            problem,
            POPULATION_SIZE,
            MAX_EVALUATIONS,
            algorithm,
            new BinaryTournamentSelection<>(new RankingAndCrowdingDistanceComparator<>()),
            new CopyCrossover(),
            new NoOpMutation(),
            1
        ).population;
    }

    private static final class CopyCrossover
            implements CrossoverOperator<CompositeSolution> {
        @Override
        public List<CompositeSolution> execute(List<CompositeSolution> parents) {
            return Arrays.asList(
                new CompositeSolution(parents.get(0)),
                new CompositeSolution(parents.get(1))
            );
        }

        @Override
        public double getCrossoverProbability() {
            return 1.0;
        }

        @Override
        public int getNumberOfRequiredParents() {
            return 2;
        }

        @Override
        public int getNumberOfGeneratedChildren() {
            return 2;
        }
    }

    private static final class NoOpMutation
            implements MutationOperator<CompositeSolution> {
        @Override
        public CompositeSolution execute(CompositeSolution solution) {
            return solution;
        }

        @Override
        public double getMutationProbability() {
            return 0.0;
        }
    }

    private static final class DeterministicConstrainedProblem
            implements Problem<CompositeSolution> {
        private final boolean includesFeasibleSolutions;
        private final AtomicInteger created = new AtomicInteger();
        private final AtomicInteger evaluated = new AtomicInteger();

        private DeterministicConstrainedProblem(boolean includesFeasibleSolutions) {
            this.includesFeasibleSolutions = includesFeasibleSolutions;
        }

        @Override
        public int getNumberOfVariables() {
            return 1;
        }

        @Override
        public int getNumberOfObjectives() {
            return 2;
        }

        @Override
        public int getNumberOfConstraints() {
            return 1;
        }

        @Override
        public String getName() {
            return "DeterministicConstrainedProblem";
        }

        @Override
        public CompositeSolution evaluate(CompositeSolution solution) {
            evaluated.incrementAndGet();
            int identifier = (int) component(solution).variables().get(0).doubleValue();
            if (includesFeasibleSolutions) {
                evaluateMixedPopulation(solution, identifier);
            } else {
                solution.objectives()[0] = identifier == 0 ? 1.0 : 0.0;
                solution.objectives()[1] = identifier == 0 ? 1.0 : 0.0;
                solution.constraints()[0] = -0.1 * (identifier + 1);
            }
            return solution;
        }

        private static void evaluateMixedPopulation(
                CompositeSolution solution,
                int identifier) {
            if (identifier < 2) {
                solution.objectives()[0] = 0.0;
                solution.objectives()[1] = 0.0;
                solution.constraints()[0] = -0.1 * (identifier + 1);
                return;
            }
            double objective = identifier == 2 ? 0.2 : 0.8;
            solution.objectives()[0] = objective;
            solution.objectives()[1] = objective;
            solution.constraints()[0] = 0.0;
        }

        @Override
        public CompositeSolution createSolution() {
            DefaultDoubleSolution component = new DefaultDoubleSolution(
                getNumberOfObjectives(),
                getNumberOfConstraints(),
                Collections.singletonList(Bounds.create(0.0, 3.0))
            );
            component.variables().set(0, (double) Math.floorMod(created.getAndIncrement(), 4));
            return new CompositeSolution(Collections.<Solution<?>>singletonList(component));
        }

        private static DoubleSolution component(CompositeSolution solution) {
            return (DoubleSolution) solution.variables().get(0);
        }

        private int evaluations() {
            return evaluated.get();
        }
    }
}
