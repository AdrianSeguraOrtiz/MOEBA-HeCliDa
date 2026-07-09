package moeba;

import org.junit.jupiter.api.Test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import static moeba.StaticUtils.csvToStringMatrix;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
