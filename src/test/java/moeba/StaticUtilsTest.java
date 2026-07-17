package moeba;

import org.junit.jupiter.api.Test;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;

import moeba.constraint.ConstraintFunction;
import moeba.fitnessfunction.FitnessFunction;
import moeba.fitnessfunction.impl.BiclusterSizeNormComp;
import moeba.representationwrapper.impl.IndividualRepresentationWrapper;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.comparator.DominanceComparator;
import static moeba.StaticUtils.csvToStringMatrix;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class StaticUtilsTest {

    @Test
    public void testcsvToStringMatrix() throws IOException {
        File inputDataset = File.createTempFile("testcsvToStringMatrix", ".csv");
        inputDataset.deleteOnExit();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(inputDataset))) {
            bw.write("header1,header2,header3\n");
            bw.write("value11,value12,value13\n");
            bw.write("value21,value22,value23\n");
        }

        String[][] matrix = csvToStringMatrix(inputDataset);

        assertEquals(2, matrix.length);
        assertEquals(3, matrix[0].length);
        assertEquals("value11", matrix[0][0]);
        assertEquals("value12", matrix[0][1]);
        assertEquals("value13", matrix[0][2]);
        assertEquals("value21", matrix[1][0]);
        assertEquals("value22", matrix[1][1]);
        assertEquals("value23", matrix[1][2]);
    }

    @Test
    public void testResolveAlgorithmForSingleThreadUsesSingleThreadEquivalentAndReportsChange() {
        PrintStream originalOut = System.out;
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));

            String resolvedGa = StaticUtils.resolveAlgorithmForThreadCount("GA-AsyncParallel", 1);
            String resolvedNsgaii = StaticUtils.resolveAlgorithmForThreadCount("NSGAII-AsyncParallel(foo=bar)", 1);

            assertEquals("GA-SingleThread", resolvedGa);
            assertEquals("NSGAII-SingleThread(foo=bar)", resolvedNsgaii);
        } finally {
            System.setOut(originalOut);
        }

        String message = new String(output.toByteArray(), StandardCharsets.UTF_8);
        assertTrue(message.contains("Algorithm GA-AsyncParallel was requested with 1 thread."));
        assertTrue(message.contains("Executing GA-SingleThread instead."));
        assertTrue(message.contains("Algorithm NSGAII-AsyncParallel was requested with 1 thread."));
        assertTrue(message.contains("Executing NSGAII-SingleThread instead."));
    }

    @Test
    public void testResolveAlgorithmKeepsAsyncWhenMoreThanOneThreadIsAvailable() {
        assertEquals(
            "NSGAII-AsyncParallel",
            StaticUtils.resolveAlgorithmForThreadCount("NSGAII-AsyncParallel", 2)
        );
    }

    @Test
    public void testResolveAlgorithmRejectsExternalFileAsyncWithOneThread() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> StaticUtils.resolveAlgorithmForThreadCount("NSGAII-ExternalFile-AsyncParallel", 1)
        );

        assertTrue(exception.getMessage().contains("requires --num-threads > 1"));
    }

    @Test
    void constrainedProblemsRejectAlgorithmsWithoutAuditedConstraintSupport() {
        double[][] data = new double[4][4];
        ColumnType[] types = new ColumnType[] {
            ColumnType.numeric(), ColumnType.numeric(), ColumnType.numeric(), ColumnType.numeric()
        };
        Problem problem = new Problem(
            new FitnessFunction[] {new BiclusterSizeNormComp(data, types, null, null, 0.5)},
            new ConstraintFunction[] {biclusters -> 0.0},
            null,
            null,
            new IndividualRepresentationWrapper(4, 4)
        );

        StaticUtils.validateConstraintSupport(problem, "NSGAII-AsyncParallel");
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> StaticUtils.validateConstraintSupport(problem, "MOCell-SingleThread")
        );

        assertTrue(exception.getMessage().contains("supported only by NSGA-II"));
    }

    @Test
    void jMetalConstrainedDominanceOrdersFeasibilityViolationAndParetoQuality() {
        DominanceComparator<DefaultDoubleSolution> comparator = new DominanceComparator<>();
        DefaultDoubleSolution feasible = point(new double[] {0.9, 0.9}, 0.0);
        DefaultDoubleSolution infeasible = point(new double[] {0.1, 0.1}, -0.1);
        DefaultDoubleSolution lowerViolation = point(new double[] {0.9, 0.9}, -0.1);
        DefaultDoubleSolution higherViolation = point(new double[] {0.1, 0.1}, -0.2);
        DefaultDoubleSolution paretoDominant = point(new double[] {0.2, 0.4}, 0.0);
        DefaultDoubleSolution paretoDominated = point(new double[] {0.3, 0.5}, 0.0);
        DefaultDoubleSolution paretoTradeoff = point(new double[] {0.1, 0.8}, 0.0);

        assertTrue(comparator.compare(feasible, infeasible) < 0);
        assertTrue(comparator.compare(lowerViolation, higherViolation) < 0);
        assertTrue(comparator.compare(paretoDominant, paretoDominated) < 0);
        assertEquals(0, comparator.compare(paretoDominant, paretoTradeoff));
    }

    private static DefaultDoubleSolution point(double[] objectives, double constraint) {
        DefaultDoubleSolution solution = new DefaultDoubleSolution(
            objectives.length,
            1,
            Collections.emptyList()
        );
        System.arraycopy(objectives, 0, solution.objectives(), 0, objectives.length);
        solution.constraints()[0] = constraint;
        return solution;
    }

    @Test
    public void testJsonToColumnTypes() throws IOException {
        File inputJsonFile = writeJson(
            "testJsonToColumnTypes",
            "{"
                + "\"numeric_col\":{\"type\":\"numeric\"},"
                + "\"boolean_col\":{\"type\":\"boolean\"},"
                + "\"nominal_col\":{\"type\":\"categorical_nominal\"},"
                + "\"ordinal_col\":{\"type\":\"categorical_ordinal\",\"order\":[\"low\",\"medium\",\"high\"]}"
                + "}"
        );
        String[] columnNames = {"numeric_col", "boolean_col", "nominal_col", "ordinal_col"};

        ColumnType[] columnTypes = StaticUtils.jsonToColumnTypes(inputJsonFile, columnNames);

        assertEquals(4, columnTypes.length);
        assertEquals(ColumnKind.NUMERIC, columnTypes[0].getKind());
        assertEquals(ColumnKind.BOOLEAN, columnTypes[1].getKind());
        assertEquals(ColumnKind.CATEGORICAL_NOMINAL, columnTypes[2].getKind());
        assertEquals(ColumnKind.CATEGORICAL_ORDINAL, columnTypes[3].getKind());
        assertIterableEquals(Arrays.asList("low", "medium", "high"), columnTypes[3].getOrdinalValues());
    }

    @Test
    public void testJsonToColumnTypesRejectsColumnNotFound() throws IOException {
        File inputJsonFile = writeJson(
            "testJsonToColumnTypesRejectsColumnNotFound",
            "{\"column1\":{\"type\":\"numeric\"},\"column2\":{\"type\":\"boolean\"}}"
        );
        String[] columnNames = {"column1", "column2", "column3"};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.jsonToColumnTypes(inputJsonFile, columnNames));
    }

    @Test
    public void testJsonToColumnTypesRejectsExtraColumns() throws IOException {
        File inputJsonFile = writeJson(
            "testJsonToColumnTypesRejectsExtraColumns",
            "{\"column1\":{\"type\":\"numeric\"},\"extra\":{\"type\":\"numeric\"}}"
        );
        String[] columnNames = {"column1"};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.jsonToColumnTypes(inputJsonFile, columnNames));
    }

    @Test
    public void testJsonToColumnTypesRejectsUnsupportedType() throws IOException {
        File inputJsonFile = writeJson(
            "testJsonToColumnTypesRejectsUnsupportedType",
            "{\"column1\":{\"type\":\"numeric\"},\"column2\":{\"type\":\"unsupported_type\"}}"
        );
        String[] columnNames = {"column1", "column2"};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.jsonToColumnTypes(inputJsonFile, columnNames));
    }

    @Test
    public void testJsonToColumnTypesRejectsLegacyFlatFormat() throws IOException {
        File inputJsonFile = writeJson(
            "testJsonToColumnTypesRejectsLegacyFlatFormat",
            "{\"column1\":\"float\",\"column2\":\"string\"}"
        );
        String[] columnNames = {"column1", "column2"};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.jsonToColumnTypes(inputJsonFile, columnNames));
    }

    @Test
    public void testJsonToColumnTypesRejectsOrdinalWithoutOrder() throws IOException {
        File inputJsonFile = writeJson(
            "testJsonToColumnTypesRejectsOrdinalWithoutOrder",
            "{\"column1\":{\"type\":\"categorical_ordinal\"}}"
        );
        String[] columnNames = {"column1"};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.jsonToColumnTypes(inputJsonFile, columnNames));
    }

    @Test
    public void testDataToNumericMatrixConvertsAllSupportedTypes() {
        String[][] data = {
            {"1", "2.5", "3.5", "Yes", "red", "medium"},
            {"2", "4.5", "5.5", "No", "blue", "low"},
            {"3", "6.5", "7.5", "yes", "red", "high"}
        };
        ColumnType[] types = {
            ColumnType.numeric(),
            ColumnType.numeric(),
            ColumnType.numeric(),
            ColumnType.bool(),
            ColumnType.categoricalNominal(),
            ColumnType.categoricalOrdinal(Arrays.asList("low", "medium", "high"))
        };

        double[][] numericData = StaticUtils.dataToNumericMatrix(data, types, 3);

        assertArrayEquals(new double[] {1.0, 2.5, 3.5, 1.0, 0.0, 1.0}, numericData[0]);
        assertArrayEquals(new double[] {2.0, 4.5, 5.5, 0.0, 1.0, 0.0}, numericData[1]);
        assertArrayEquals(new double[] {3.0, 6.5, 7.5, 1.0, 0.0, 2.0}, numericData[2]);
    }

    @Test
    public void testDataToNumericMatrixConvertsCommonBooleanValues() {
        String[][] data = {
            {"True"},
            {"False"},
            {"1"},
            {"0"},
            {"yes"},
            {"no"}
        };
        ColumnType[] types = {ColumnType.bool()};

        double[][] numericData = StaticUtils.dataToNumericMatrix(data, types, 2);

        assertArrayEquals(new double[] {1.0}, numericData[0]);
        assertArrayEquals(new double[] {0.0}, numericData[1]);
        assertArrayEquals(new double[] {1.0}, numericData[2]);
        assertArrayEquals(new double[] {0.0}, numericData[3]);
        assertArrayEquals(new double[] {1.0}, numericData[4]);
        assertArrayEquals(new double[] {0.0}, numericData[5]);
    }

    @Test
    public void testDataToNumericMatrixPreservesMissingValuesForEveryColumnKind() {
        String[][] data = {
            {null, null, null, null},
            {"1.5", "yes", "red", "high"}
        };
        ColumnType[] types = {
            ColumnType.numeric(),
            ColumnType.bool(),
            ColumnType.categoricalNominal(),
            ColumnType.categoricalOrdinal(Arrays.asList("low", "high"))
        };

        double[][] numericData = StaticUtils.dataToNumericMatrix(data, types, 2);

        assertTrue(Double.isNaN(numericData[0][0]));
        assertTrue(Double.isNaN(numericData[0][1]));
        assertTrue(Double.isNaN(numericData[0][2]));
        assertTrue(Double.isNaN(numericData[0][3]));
        assertArrayEquals(new double[] {1.5, 1.0, 0.0, 1.0}, numericData[1]);
    }

    @Test
    public void testDataToNumericMatrixRejectsInvalidThreadCount() {
        String[][] data = {{"1"}};
        ColumnType[] types = {ColumnType.numeric()};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 0));
    }

    @Test
    public void testDataToNumericMatrixRejectsMismatchedTypes() {
        String[][] data = {{"1", "2"}};
        ColumnType[] types = {ColumnType.numeric()};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 1));
    }

    @Test
    public void testDataToNumericMatrixRejectsRowsWithDifferentColumnCounts() {
        String[][] data = {
            {"1", "2"},
            {"3"}
        };
        ColumnType[] types = {ColumnType.numeric(), ColumnType.numeric()};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 2));
    }

    @Test
    public void testDataToNumericMatrixRejectsNullMatrixAndRowsWithControlledError() {
        String[][] firstRowNull = {null, {"1"}};
        String[][] laterRowNull = {{"1"}, null};
        ColumnType[] types = {ColumnType.numeric()};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(null, types, 1));
        assertThrows(
            IllegalArgumentException.class,
            () -> StaticUtils.dataToNumericMatrix(firstRowNull, types, 1)
        );
        assertThrows(
            IllegalArgumentException.class,
            () -> StaticUtils.dataToNumericMatrix(laterRowNull, types, 1)
        );
    }

    @Test
    public void testDataToNumericMatrixPropagatesNumericConversionErrors() {
        String[][] data = {
            {"1"},
            {"not-a-number"}
        };
        ColumnType[] types = {ColumnType.numeric()};

        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> StaticUtils.dataToNumericMatrix(data, types, 2)
        );

        assertEquals("Invalid numeric value at row 1, column 0: not-a-number", exception.getMessage());
    }

    @Test
    public void testDataToNumericMatrixRejectsInvalidBooleanValues() {
        String[][] data = {
            {"true"},
            {"not-a-boolean"}
        };
        ColumnType[] types = {ColumnType.bool()};

        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> StaticUtils.dataToNumericMatrix(data, types, 2)
        );

        assertEquals("Invalid boolean value at row 1, column 0: not-a-boolean", exception.getMessage());
    }

    @Test
    public void testDataToNumericMatrixRejectsNullTypes() {
        String[][] data = {{"1"}};
        ColumnType[] types = {null};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 1));
    }

    @Test
    public void testDataToNumericMatrixRejectsUnknownOrdinalValues() {
        String[][] data = {{"unknown"}};
        ColumnType[] types = {ColumnType.categoricalOrdinal(Arrays.asList("low", "high"))};

        assertThrows(IllegalArgumentException.class, () -> StaticUtils.dataToNumericMatrix(data, types, 1));
    }

    private File writeJson(String prefix, String content) throws IOException {
        File inputJsonFile = File.createTempFile(prefix, ".json");
        inputJsonFile.deleteOnExit();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(inputJsonFile))) {
            bw.write(content);
        }
        return inputJsonFile;
    }
}
