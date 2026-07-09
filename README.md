<img src="./docs/logo.svg" width="40%" align="right" style="margin: 1em">

**MOEBA-BIO (Multi-Objective Evolutionary Biclustering Algorithm for BIOmedical applications)** is a software project whose java implementation is aimed at making possible a better use of the information of the biological domain of biclustering problems in order to maximise learning during the execution of the algorithm and to achieve its expansion to questions not dealt with by other proposals. The information provided by the data's belonging to a specific biological domain (e.g., gene co-expression, clinical patient analysis, medical image segmentation, etc.) has the potential to enrich the traditional establishment of purely mathematical objectives based on data cohesion within biclusters and their size. However, the representation of individuals traditionally chosen to solve the biclustering problem hinders the effective injection of knowledge by not allowing a direct equivalence between the algorithm's solution and the real problem's solution. **MOEBA-BIO** aims to establish a solid framework to overcome these limitations by implementing several completely novel aspects that allow the inclusion of domain knowledge not only in the algorithm's learning but also in the self-configuration of parameters and self-determination of the number of biclusters.

## Usage

### Runner (main execution)
`Runner.java` is the main entry point. Build the runnable jar and execute it with the required inputs.

```bash
mvn -DskipTests package
java -cp target/moeba-bio-1.0.0-jar-with-dependencies.jar moeba.Runner \
  --input-dataset path/to/data.csv \
  --input-column-types path/to/column-types.json \
  --representation GENERIC \
  --output-folder output/
```

Required inputs:
* `--input-dataset`: CSV dataset to bicluster.
* `--input-column-types`: JSON object keyed by CSV column name with the semantic type of each column.

Column type JSON schema:

```json
{
  "age": {"type": "numeric"},
  "responded": {"type": "boolean"},
  "diagnosis": {"type": "categorical_nominal"},
  "severity": {"type": "categorical_ordinal", "order": ["low", "medium", "high"]}
}
```

Supported `type` values are `numeric`, `boolean`, `categorical_nominal`, and `categorical_ordinal`. Ordinal categorical columns must declare `order`; nominal categorical values are encoded only as transient numeric identifiers during matrix conversion.

Common optional flags:
* `--str-fitness-functions`: Objectives separated by `;` (e.g., `BiclusterSizeNormComp;MeanSquaredResidueNorm`).
* `--str-algorithm`: Algorithm name (e.g., `NSGAII-AsyncParallel`).
* `--population-size`, `--max-evaluations`, `--num-threads`.
* `--have-internal-cache`, `--have-external-cache`.
* `--output-folder`: Folder to store `VAR.csv`, `FUN.csv`, `VAR-translated.csv`, and observer CSVs.

Notes:
* Use `--specific-num-biclusters` only with `--representation SPECIFIC`.
* Use `--generic-initial-min-num-bics` and `--generic-initial-max-num-bics` only with `--representation GENERIC`.

### ValidationRunner (validation vs gold standard)
`ValidationRunner.java` evaluates inferred biclusters against a gold standard. The usual input is the `VAR-translated.csv` produced by `Runner`.

```bash
java -cp target/moeba-bio-1.0.0-jar-with-dependencies.jar moeba.validation.ValidationRunner \
  --inferred-translated output/VAR-translated.csv \
  --gold-standard-translated path/to/gold-standard.csv \
  --representation GENERIC \
  --validation-metrics ScorePrelicRelevance;ScorePrelicRecovery;ScoreDice \
  --output-file output/scores.csv
```

Common optional flags:
* `--save-process`: Stores the validation process and compresses it to `process.zip` next to the output file.
* `--num-threads`: Number of threads for metric computation.

## Maven Central

Use MOEBA-BIO as a dependency from Maven Central:

```xml
<dependency>
  <groupId>io.github.adrianseguraortiz</groupId>
  <artifactId>moeba-bio</artifactId>
  <version>1.0.0</version>
</dependency>
```

Gradle (Kotlin DSL):

```kotlin
dependencies {
    implementation("io.github.adrianseguraortiz:moeba-bio:1.0.0")
}
```

## Representation

The usual representation applied to the biclustering problem is one where each individual represents a bicluster by specifying the rows and columns that comprise it. This approach highlights two main drawbacks:

1. **Limitation of Learning**: The fitness of an individual can only be measured by the individual qualities of the bicluster it represents, without considering other fundamental aspects such as the global distribution of all biclusters, differentiation between them, or the overall coverage of the original data matrix. This limitation is even greater when aiming to satisfy real-world aspects of the biomedical data context.

2. **Distortion of Reality**: An individual does not represent a possible real solution to the problem, but rather a part of it. Traditionally, the algorithm's solution front has been taken as the problem's solution, which brings about two new issues: a fairly subjective post-processing of the algorithm's solution is required, and within the final solution, there are biclusters specialized in certain objectives that belong to the front at the expense of not satisfying the others. This can result in large biclusters without cohesion among their data and vice versa.

**MOEBA-BIO** provides a new representation called **generic representation** that aims to overcome all these drawbacks of the traditional representation, referred to as **individual representation**. In this novel representation (limited to prohibiting overlap between rows or columns, which is usual), each individual represents a real solution to the problem where the number of biclusters it contains is variable and thus part of the algorithm's learning. This representation, in addition to allowing the inclusion of objectives that pursue global aspects of the real solution, also allows for self-determination and convergence of the optimal number of biclusters. This not only facilitates the incorporation of specific biomedical domain knowledge but also expands it to a fundamental aspect such as the optimal number of biclusters.

![Alt text](./docs/representation.svg)

## Self-Configuration

Often, regardless of the nature of the data, the same fitness functions have been used in the literature to solve the biclustering problem. Some of these famous functions include: maximizing the size of the biclusters, minimizing the variance within the biclusters, minimizing the Mean Squared Residue, maximizing the variance between the rows, maximizing correlation formulas, minimizing the Virtual Error, etc. After introducing the new designed representation, other new possible functions have been unlocked, capable of performing a more global evaluation that considers all candidate biclusters of the solution and addresses their distribution, coverage, distance, etc. Additionally, this global position achieved thanks to the implemented representation facilitates the design of new objectives that may be associated with the specific biomedical context of the data. Therefore, it is of little use to establish this framework full of possibilities without offering a way to answer the logical question: How do I know which subset of all available objectives explains more accurately the biclustering of data from a specific domain?

To answer this question reliably, a self-configurator based on two nested phases has been designed:

1. **Supervised Phase**: In this phase, different combinations of objectives and values of their subparameters are tested to determine which configuration minimizes a specific validation metric concerning the gold standard of the benchmark representing the biomedical domain of the data. However, since different configurations at this level imply different search spaces with varying numbers of dimensions, it would be reckless to set the same values for the rest of the parameters in all cases. Therefore, given a configuration at this level, before proceeding to its evaluation, it is necessary to extract the best values for the rest of the parameters from the next phase.

2. **Unsupervised Phase**: Once the problem objectives and their corresponding subparameters are set, this phase optimizes the values of the remaining parameters, including some as crucial as the algorithm to be used (NSGA-II, MOEAD, MOSA, SPEA2, IBEA, etc.) with their corresponding subparameters, operators, the probabilities of each phase, population size, etc. This optimization, already performed on comparable fronts, can be based on common unsupervised metrics such as Hypervolume, Epsilon, etc.

Examples of files where the different options to consider during the self-configuration are specified can be found in the [parameterization](./parameterization/) folder. As can be seen, one file is necessary to specify the options for the parameters optimized during the supervised phase, and another to set the options for the parameters of the unsupervised phase. If either file sets a single value for all parameters, that phase will be ignored during the self-configuration.

![Alt text](./docs/autoconfiguration.svg)

## Implementation

The code for this project is neatly segmented into components to facilitate the inclusion of new objectives, operators, and even representations. The project extends the well-known jMetal framework, providing a reliable foundation for its construction. Notable components include the following:

* **Fitness Functions**: The [fitness functions](./src/main/java/moeba/fitnessfunction/impl/) can implement three different interfaces: [IndividualBiclusterFitnessFunction](./src/main/java/moeba/fitnessfunction/IndividualBiclusterFitnessFunction.java) for functions that assign a value to each bicluster without access to the rest (as in traditional cases), [GenericBiclusterFitnessFunction](./src/main/java/moeba/fitnessfunction/GenericBiclusterFitnessFunction.java) for objectives that assign a value to each bicluster but whose calculation also depends on the other biclusters (only possible with the newly designed representation), and [GlobalFitnessFunction](./src/main/java/moeba/fitnessfunction/GlobalFitnessFunction.java) for fitness functions that assign a global value to the set of biclusters given by an individual of the new representation. In any case, to include a new objective, it is only necessary to implement the function specified by the chosen interface. Additionally, all fitness functions have access to the data type of each column in case functions for heterogeneous data are considered in the future.

* **Representations**: Everything related to a specific representation is provided by its [wrapper](./src/main/java/moeba/representationwrapper/impl/), which defines fundamental aspects such as the number of variables of each type, the limits of these variables, the available crossover and mutation operators, the interpretation of its encoding for the extraction of biclusters to the common intermediate state, etc.

* **Operators**: The [operators](./src/main/java/moeba/operator/) are responsible for crossing and mutating individuals. Each representation requires its own operators, and all have corresponding interfaces to facilitate the inclusion of new implementations.

* **Caches**: The [caches](./src/main/java/moeba/utils/storage/impl/) prevent the algorithm from wasting computational resources by repeatedly evaluating the same biclusters. Since the generic representation is redundant, multiple individuals may translate into the same common intermediate state, or several individuals may contain a bicluster already present in another candidate's set. To make the cache lighter, a hybrid version based on the [Ehcache](https://www.ehcache.org/) library has been implemented, which also allows for a [common cache for multiple machines](https://www.ehcache.org/documentation/3.5/clustered-cache.html) that could be very useful if the dynamic representation proliferates.

* **Observers**: The [observers](./src/main/java/moeba/utils/observer/impl/) are very useful components for recording the algorithm's evolution in different aspects. There are observers to view the fitness values of each generation (minimum, maximum, or average), display in the console the number of evaluations performed in real-time, observe the distribution of bicluster quantities in each generation, show the number of calls made to the internal or external cache, and even specific observers for parameter self-configuration. These observers execute their code after the evaluation of each individual and only require implementing a register function and a writeToFile function.

* **Validation Metrics**: Although not part of the main execution of this proposal, to feed other options in the supervised phase of the self-configurator, numerous [validation metrics](./src/main/java/moeba/validation/metric/impl/) have been implemented to quantify the accuracy with which the biclusters have been inferred.
