#!/bin/bash
#SBATCH --job-name=lsmlpsA    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=875532@stud.unive.it     # Where to send mail	
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # 2
#SBATCH --ntasks-per-node=1         # 2
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=1000:00:00               # Time limit hrs:min:sec
#SBATCH --output=empty_log.log   # Standard output and error log
#SBATCH --gpus=quadro_rtx_5000:1

conda activate algtg

#lsmlps_gtg lstmreg_gtg lstmbc_gtg

#python3 app/main.py -gpus 1 -m lsmlps_gtg -ds cifar10 -tr 1 -am corr -am_ts mean -e_s mean -gtg_iter 10 #-am_s uncertanity #--temp_unlab_pool
#python3 app/main.py -gpus 1 -m lsmlps_gtg llmlp_gtg lstmreg_gtg lstmbc_gtg -ds cifar10 -tr 1 -am corr -am_ts mean -e_s mean -gtg_iter 10 #-am_s uncertanity #--temp_unlab_pool

#python3 app/main.py -gpus 1 -m ll_gtg -ds cifar10 -tr 1 -am corr -am_s uncertanity -am_ts mean -e_s mean #--temp_unlab_pool
#python3 app/main.py -gpus 1 -m ll_gtg -ds cifar10 -tr 1 -am corr -am_s uncertanity -am_ts none -e_s mean -am_md UU #--affnity_matrix_self_loop #--temp_unlab_pool
#python3 app/main.py -gpus 1 -m coreset badge bald cdal tavaal alphamix tidal -ds cifar10 -tr 5 --temp_unlab_pool

#python3 app/main.py -gpus 1 -m random entropy ll -ds cifar10 -tr 5 -cs_s kmeans

#python3 app/main.py -gpus 1 -m ll_gtg -ds cifar10 -tr 1 -am corr -am_s uncertanity -am_ts mean -e_s mean
#python3 app/main.py -gpus 1 -m ll_gtg -ds cifar10 -tr 1 -am corr -am_s uncertanity -am_ts mean -e_s mean --affnity_matrix_self_loop

#python3 app/main.py -gpus 1 -m random entropy coreset ll tidal cdal bald badge alphamix tavaal -ds tinyimagenet -tr 3


#python3 app/main.py -gpus 1 -m lsmlps_gtg -ds cifar10 -tr 5 -am corr -am_ts mean -e_s mean -gtg_iter 10 #-am_s uncertanity #--temp_unlab_pool
python3 app/main.py -gpus 1 -m lsmlpsA_gtg -ds cifar10 -tr 5 -am corr -am_ts mean -e_s mean -gtg_iter 10 #-am_s uncertanity #--temp_unlab_pool
