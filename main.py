from default_settings import EMPTY
from example_puzzles import P0, P1, P2, P71, P72

from benchmark import compare, run_benchmark, plot_benchmark




if __name__=="__main__":
    # list(EMPTY.solve(disp=True, mode=2))
    compare(P2, modes = [1,2], nr_tests = 10)

    # list(P2.solve(disp=True, mode=0))
    # 1. Run computation and save data
    # data_file_path = run_benchmark(EMPTY, modes=[0, 1, 2], nr_tests=10, T=20)

    # # data_file_path = os.path.join("benchmark_results", "T=10_modes=[0-1-2]_tests=10", "raw_data.pkl")
    # # 2. Plot from data file later without recalculating
    # plot_benchmark(data_file_path, show_trendline=True)
