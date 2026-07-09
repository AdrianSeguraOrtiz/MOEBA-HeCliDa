import pandas as pd
import json
import copy
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from biclustlib.algorithms import *
from biclustlib.algorithms.wrappers import *
from biclustlib.benchmark import GeneExpressionBenchmark, Algorithm
from biclustlib.benchmark.data import load_tavazoie, load_prelic, load_jaskowiak
from biclustlib.models import Biclustering, Bicluster

INPUT_FOLDER = "biclustlib_benchmark_data"
OUTPUT_FOLDER = "exp6-functional-enrichment"

def get_moeba_column_type(dtype):
    if pd.api.types.is_bool_dtype(dtype):
        return {"type": "boolean"}
    if pd.api.types.is_numeric_dtype(dtype):
        return {"type": "numeric"}
    return {"type": "categorical_nominal"}

def store_benchmark_data(benchmarks: list[pd.DataFrame], names: list[str]):
    for i, data in enumerate(benchmarks):
        # 1. Save normalized data to CSV
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(data))
        normalized_data.to_csv(f"{INPUT_FOLDER}/{names[i]}-data.csv", index=False)
        
        # 2. Save the MOEBA-BIO semantic column types as JSON
        column_data_types = {col: get_moeba_column_type(dtype) for col, dtype in normalized_data.dtypes.items()}
        with open(f"{INPUT_FOLDER}/{names[i]}-types.json", "w") as file:
            json.dump(column_data_types, file, indent=4, ensure_ascii=False)


def discretize_data(raw_data: pd.DataFrame, n_bins: int = 2) -> pd.DataFrame:
    return pd.DataFrame(KBinsDiscretizer(n_bins, encode='ordinal', strategy='kmeans').fit_transform(raw_data),
                        index=raw_data.index).astype(int if n_bins > 2 else bool)
    
def read_moeba_bio_solutions(moeba_file_solution: str) -> list[Biclustering]:
    # Read MOEBA-BIO biclusters
    with open(moeba_file_solution, "r") as f:
        lines = f.readlines()
        solutions = []
        for line in lines:
            line = line.replace("\n", "")
            line = line.replace(": (rows: [", " ")
            line = line.replace("] cols: [", ",")
            line = line.replace("])", "")
            line = line.replace(", ", ";")
            line = line.split(";")
            biclusters = []
            for bic in line:
                bic = bic.split(",")
                rows = bic[0].split(" ")[1:]
                cols = bic[1].split(" ")
                cols = [] if len(cols) == 1 and cols[0] == "" else cols
                if len(rows) > 2:
                    biclusters.append(Bicluster(list(map(int, rows)), list(map(int, cols))))
            
            solutions.append(Biclustering(biclusters))
    return solutions
    
def functional_enrichment(data: pd.DataFrame, name: str, moeba_bio_solutions: list) -> GeneExpressionBenchmark:
    
    n_biclusters = max(1, int(0.1 * (data.shape[0] + data.shape[1]) / 2))
    discretion_level = 10

    data_dis = discretize_data(data, discretion_level)
    data_bin = discretize_data(data)

    setup = [
        Algorithm('CCA', ChengChurchAlgorithm(n_biclusters), data),
        Algorithm('Bimax', RBinaryInclusionMaximalBiclusteringAlgorithm(n_biclusters), data_bin),
        Algorithm('ISA', IterativeSignatureAlgorithm2(), data),
        Algorithm('OPSM', OrderPreservingSubMatrix(), data),
        Algorithm('Plaid', RPlaid(n_biclusters), data),
        Algorithm('BiBit', BitPatternBiclusteringAlgorithm(), data_bin),
        Algorithm('xMotifs', RConservedGeneExpressionMotifs(n_biclusters), data_dis),
        Algorithm('LAS', LargeAverageSubmatrices(n_biclusters), data),
        Algorithm('Spectral', Spectral(n_clusters=data.shape[1] // 2), data + abs(data.min().min()) + 1),
    ]

    benchmark = GeneExpressionBenchmark(algorithms=setup, raw_data=data).run()

    # Add one algotihm for each solution and change biclusters of MOEBA-BIO
    for i in range(len(moeba_bio_solutions)):
        benchmark.algorithms.append(Algorithm(f"MOEBA-BIO {i}", ChengChurchAlgorithm(1), data))
        data = copy.copy(benchmark["CCA"])
        data["biclustering"] = moeba_bio_solutions[i]
        data["n_found"] = len(moeba_bio_solutions[i].biclusters)
        benchmark.__setitem__(f"MOEBA-BIO {i}", data)
        
    for alg in benchmark.algorithms:
        biclusters = benchmark[alg.name]["biclustering"].biclusters
        if len(biclusters) > 300:
            benchmark[alg.name]["biclustering"] = Biclustering(biclusters[:300])
            benchmark[alg.name]["n_found"] = 300
        if len(biclusters) == 0:
            benchmark[alg.name]["biclustering"] = Biclustering([Bicluster([0], [0])])
            benchmark[alg.name]["n_found"] = 1
        print(f"{alg.name}: {benchmark[alg.name]['n_found']} biclusters")

    benchmark.perform_goea()
    benchmark.generate_goea_report(report_dir=f"{OUTPUT_FOLDER}/{name}/report")


benchmarks = load_jaskowiak() + [load_tavazoie(), load_prelic()]
names = [f"Jaskowiak-{i}" for i in range(1, len(benchmarks)-1)] + ["Tavazoie", "Prelic"]
#store_benchmark_data(benchmarks, names)

for i, data in enumerate(benchmarks):
    moeba_bio_solutions = read_moeba_bio_solutions(f"{OUTPUT_FOLDER}/{names[i]}/VAR-translated.csv")
    functional_enrichment(data, names[i], moeba_bio_solutions)
