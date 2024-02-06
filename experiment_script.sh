 #!/bin/bash         

/home/user/anaconda3/envs/test-env-newgpu-python39/bin/python "/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/main_lexical_multitoken.py" -glns "AVERAGE-GLOSS" -val True -outp "dataset/results/coinco_results_multitoken_average_gloss_pure_val_" -eval "dataset/results/coinco_results_multitoken_average_gloss_pure_val_final_"

/home/user/anaconda3/envs/test-env-newgpu-python39/bin/python "/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/main_lexical_multitoken.py" -glns "AVERAGE-GLOSS" -smiln True -outp "dataset/results/coinco_results_multitoken_average_gloss_pure_similn_" -eval "dataset/results/coinco_results_multitoken_average_gloss_pure_similn_final_"

/home/user/anaconda3/envs/test-env-newgpu-python39/bin/python "/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/main_lexical_multitoken.py" -glns "AVERAGE-GLOSS" -glossc True -outp "dataset/results/coinco_results_multitoken_average_gloss_pure_glossc_" -eval "dataset/results/coinco_results_multitoken_average_gloss_pure_glossc_final_"

/home/user/anaconda3/envs/test-env-newgpu-python39/bin/python "/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/main_lexical_multitoken.py" -glns "AVERAGE-GLOSS" -pro True -val True -similn True -glossc True -outp "dataset/results/coinco_results_multitoken_average_gloss_all_" -eval "dataset/results/coinco_results_multitoken_average_gloss_all_final_"
