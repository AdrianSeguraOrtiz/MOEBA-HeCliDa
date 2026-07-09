package moeba;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

public final class ColumnType {
    private final ColumnKind kind;
    private final List<String> ordinalValues;

    private ColumnType(ColumnKind kind, List<String> ordinalValues) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.ordinalValues = ordinalValues == null
            ? Collections.emptyList()
            : Collections.unmodifiableList(new ArrayList<>(ordinalValues));
    }

    public static ColumnType numeric() {
        return new ColumnType(ColumnKind.NUMERIC, null);
    }

    public static ColumnType bool() {
        return new ColumnType(ColumnKind.BOOLEAN, null);
    }

    public static ColumnType categoricalNominal() {
        return new ColumnType(ColumnKind.CATEGORICAL_NOMINAL, null);
    }

    public static ColumnType categoricalOrdinal(List<String> ordinalValues) {
        if (ordinalValues == null || ordinalValues.isEmpty()) {
            throw new IllegalArgumentException("Ordinal columns require a non-empty order.");
        }
        Set<String> uniqueValues = new HashSet<>(ordinalValues);
        if (uniqueValues.size() != ordinalValues.size()) {
            throw new IllegalArgumentException("Ordinal columns cannot declare duplicate values.");
        }
        return new ColumnType(ColumnKind.CATEGORICAL_ORDINAL, ordinalValues);
    }

    public ColumnKind getKind() {
        return kind;
    }

    public List<String> getOrdinalValues() {
        return ordinalValues;
    }

    public boolean isNumeric() {
        return kind == ColumnKind.NUMERIC;
    }
}
