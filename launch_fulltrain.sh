conda activate test-env-newgpu-python39
python main_lexical_coinco.py -seed 6809 -pro True -val True -glossc True -similn True -outp "dataset/results/results_gap_multitoken_full_coinco_" -eval "dataset/results/results_final_gap_multitoken_full_coinco_" -tt "dataset/LS14/test_refactored/coinco_test_multitokens.preprocessed" -tgf "dataset/LS14/test_refactored/coinco_test_multitokens.gold"
