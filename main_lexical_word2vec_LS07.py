import torch
import argparse
import numpy as np
import random
import time
from tqdm import tqdm

from metrics.evaluation import evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    # --------------- train dataset
    parser.add_argument("-tf", "--train_file", type=str, help="path of the train file dataset",
                        default='dataset/LS07/trial/lst_trial.preprocessed')
    parser.add_argument("-gf", "--train_golden_file", type=str, help="path of the golden file dataset",
                        default='dataset/LS07/trial/lst_trial.gold')

    # --------------- test dataset
    parser.add_argument("-tt", "--test_file", type=str, help="path of the test file dataset",
                        default='dataset/LS07/test/lst_test.preprocessed')
    parser.add_argument("-tgf", "--test_golden_file", type=str, help="path of the golden file dataset",
                        default='dataset/LS07/test/lst_test.gold')



    # --------------- output results
    parser.add_argument("-outp", "--output_results", type=str, help="path of the output file with results",
                        default='baseline_word2vec/results/first_results_file')
    parser.add_argument("-eval", "--results_file", type=str, help="path of the output file with gap metric",
                        default='baseline_word2vec/results/first_results_final_metrics')

    # -------------------parameters
    
    # ----------gap flags
    parser.add_argument("-g", "--gap", type=bool, help="whether we use the gap ranking (candidate ranking)",
                        default=True)
    parser.add_argument("-gfc", "--golden_file_cadidates", type=str, help="path of the golden file dataset for gap",
                        default='dataset/LS07/lst.gold.candidates')

    parser.add_argument("-seed", "--seed", type=int, help="the seed that we are using for the experiment",
                        default=6809)
    args = parser.parse_args()
    seed = args.seed

    """
    reader of features/labels and candidates if gap
    """
    

    evaluation_metric = evaluation()


    "======================================================================================================"
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    "======================================================================================================"
    output_best_rank_file = args.output_results + ".ranked"
    output_rank_list = open(output_gap_rank_file, 'r').read().split('\n')
    output_rank_list = output_rank_list[:-1]
    output_rank_file = open(args.output_results + "lemmatized.ranked",'w')
    for i in range(len(output_rank_list)):
        
    
    if args.gap:
        evaluation_metric.gap_calculation(args.test_golden_file,
                                          args.output_results + ".ranked",
                                          args.results_file + "_gap.txt")
    evaluation_metric.calculation_perl(args.test_golden_file,
                                       args.output_results + ".generated.best",
                                       args.output_results + ".generated.oot",
                                       args.results_file + ".best",
                                       args.results_file + ".oot")
    """
    evaluation_metric.calculation_p1(args.test_golden_file,
                                     args.output_results + "_p1.txt",
                                     args.results_file + args.fna + "_" + str(seed) + "_p1.txt")
    """