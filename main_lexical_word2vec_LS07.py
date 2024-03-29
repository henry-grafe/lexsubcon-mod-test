import torch
import argparse
import numpy as np
import random
import time
from tqdm import tqdm

from metrics.evaluation import evaluation

def make_candidates_dict(filename):
    f = open(filename, 'r', encoding="latin5")
    lines = f.read().split('\n')[:-1]
    candidates_dict = {}
    for i in range(len(lines)):
        line = lines[i]
        line = line.split("::")
        target_word = line[0]
        candidates = line[1].split(";")
        candidates_dict[target_word] = candidates
    return candidates_dict

def make_fixed_gap_output(original_file_name, fixed_file_name, candidates_dict):
    in_f = open(original_file_name, 'r', encoding='latin5')
    out_f = open(fixed_file_name, 'w', encoding='latin5')
    in_data = in_f.read().split("\n")[:-1]
    #print(in_data)
    for i in range(len(in_data)):
        line = in_data[i].split("\t")
        assert line[0] == "RANKED"
        target_word = " ".join(line[1].split(' ')[:-1])
        target_id = line[1].split(' ')[-1]
        line_candidates = line[2:]
        #print(line_candidates)
        if len(line_candidates)==1 and line_candidates[0]=='':
            #print(target_word, target_id)
            #print(line_candidates)
            fixed_line = fix_empy_line(target_word=target_word, target_id=target_id, candidates_dict=candidates_dict)
            out_f.write(fixed_line+'\n')
        else:
            out_f.write(in_data[i]+'\n')
    out_f.close()

def fix_empy_line(target_word, target_id, candidates_dict):
    fixed_line = "RANKED\t"+target_word+" "+target_id
    candidates_list = candidates_dict[target_word]
    scores = np.random.random(len(candidates_list))
    scores = np.flip(np.sort(scores))
    for i in range(len(candidates_list)):
        fixed_line = fixed_line + "\t" + candidates_list[i] + " " + str(float(scores[i]))
    return fixed_line

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
                        default=6942)
    args = parser.parse_args()
    seed = args.seed
    seed = 42
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
    print("loading gap candidate file...")
    candidates_dict = make_candidates_dict("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/dataset/LS14/coinco.gold.candidates")
    print("fixing the ranked candidates score file... (adding candidates in random order for lines where the target word was not in the word2vecf vocabulary)")
    oo = make_fixed_gap_output("baseline_word2vec/results/results_coinco_test_mul.ranked", 
                               "baseline_word2vec/results/results_coinco_test_mul_fixed.ranked", 
                               candidates_dict=candidates_dict)
    
    print("computing metrics...")
    if args.gap:
        evaluation_metric.gap_calculation("dataset/LS14/test_refactored/coinco_test_multitokens.gold",
                                          "baseline_word2vec/results/results_coinco_test_mul_fixed.ranked",
                                          "dataset/results/coinco_results_multitoken_word2vecf_mult" + "_gap.txt")
    evaluation_metric.calculation_perl("dataset/LS14/test_refactored/coinco_test_multitokens.gold",
                                       "baseline_word2vec/results/results_coinco_test_mul.generated.best",
                                       "baseline_word2vec/results/results_coinco_test_mul.generated.oot",
                                       "dataset/results/coinco_results_multitoken_word2vecf_mult" + ".best",
                                       "dataset/results/coinco_results_multitoken_word2vecf_mult" + ".oot")
    """
    evaluation_metric.calculation_p1("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/dataset/LS14/test_refactored/coinco_test_multitokens.gold",
                                     args.output_results + "_p1.txt",
                                     args.results_file + args.fna + "_" + str(seed) + "_p1.txt")
    """