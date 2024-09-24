#ade100-50
python train_inc_CLIP_CISS.py --num-gpus 8 --config-file configs/ade20k/semantic-segmentation/CLIP_CISS-Base-ADE20K-SemanticSegmentation.yaml OUTPUT_DIR results CONT.BASE_CLS 100 CONT.INC_CLS 50 CONT.MODE overlap SEED 42 CONT.TASK 0 SOLVER.BASE_LR 0.00001 NAME ade_exp SOLVER.MAX_ITER 30000 SOLVER.IMS_PER_BATCH 16 CONT.DIST.GPD 1.0 CONT.ORI_CLIP True TEST.EVAL_PERIOD 2000 

python train_inc_CLIP_CISS.py --num-gpus 8 --config-file configs/ade20k/semantic-segmentation/CLIP_CISS-Base-ADE20K-SemanticSegmentation.yaml OUTPUT_DIR results CONT.BASE_CLS 100 CONT.INC_CLS 50 CONT.MODE overlap SEED 42 CONT.TASK 1 SOLVER.BASE_LR 0.000005 NAME ade_exp SOLVER.MAX_ITER 20000 SOLVER.IMS_PER_BATCH 16 CONT.DIST.GPD 1.0 CONT.DIST.Q_KD 0.15 TEST.EVAL_PERIOD 2000

#ade50-50
python train_inc_CLIP_CISS.py --num-gpus 8 --config-file configs/ade20k/semantic-segmentation/CLIP_CISS-Base-ADE20K-SemanticSegmentation.yaml OUTPUT_DIR results CONT.BASE_CLS 50 CONT.INC_CLS 50 CONT.MODE overlap SEED 42 CONT.TASK 0 SOLVER.BASE_LR 0.00001 NAME ade_exp SOLVER.MAX_ITER 20000 SOLVER.IMS_PER_BATCH 16 CONT.DIST.GPD 1.0 CONT.ORI_CLIP True TEST.EVAL_PERIOD 2000 

for t in 1 2; do
python train_inc_CLIP_CISS.py --num-gpus 8 --config-file configs/ade20k/semantic-segmentation/CLIP_CISS-Base-ADE20K-SemanticSegmentation.yaml OUTPUT_DIR results CONT.BASE_CLS 50 CONT.INC_CLS 50 CONT.MODE overlap SEED 42 CONT.TASK ${t} SOLVER.BASE_LR 0.000005 NAME ade_exp SOLVER.MAX_ITER 20000 SOLVER.IMS_PER_BATCH 16 CONT.DIST.GPD 1.0 CONT.DIST.Q_KD 0.15 TEST.EVAL_PERIOD 2000
done


#ade100-10
python train_inc_CLIP_CISS.py --num-gpus 8 --config-file configs/ade20k/semantic-segmentation/CLIP_CISS-Base-ADE20K-SemanticSegmentation.yaml OUTPUT_DIR /root/results CONT.BASE_CLS 100 CONT.INC_CLS 10 CONT.MODE overlap SEED 42 CONT.TASK 0 SOLVER.BASE_LR 0.00001 NAME ade_exp SOLVER.MAX_ITER 30000 SOLVER.IMS_PER_BATCH 16 CONT.DIST.GPD 1.0 CONT.ORI_CLIP True TEST.EVAL_PERIOD 2000 

for  t in 1 2 3 4 5; do
python train_inc_CLIP_CISS.py --num-gpus 8 --config-file configs/ade20k/semantic-segmentation/CLIP_CISS-Base-ADE20K-SemanticSegmentation.yaml OUTPUT_DIR /root/results CONT.BASE_CLS 10 CONT.INC_CLS 10 CONT.MODE overlap SEED 42 CONT.TASK  ${t}  SOLVER.BASE_LR 0.000008 NAME ade_exp SOLVER.MAX_ITER 5000 SOLVER.IMS_PER_BATCH 16 CONT.DIST.GPD 1.0 CONT.DIST.Q_KD 0.15 TEST.EVAL_PERIOD 500
done

ade100-5
python train_inc_CLIP_CISS.py --num-gpus 8 --config-file configs/ade20k/semantic-segmentation/CLIP_CISS-Base-ADE20K-SemanticSegmentation.yaml OUTPUT_DIR results CONT.BASE_CLS 100 CONT.INC_CLS 5 CONT.MODE overlap SEED 42 CONT.TASK 0 SOLVER.BASE_LR 0.00001 NAME ade_exp SOLVER.MAX_ITER 30000 SOLVER.IMS_PER_BATCH 16 CONT.DIST.GPD 1.0 CONT.ORI_CLIP True TEST.EVAL_PERIOD 2000

for t in 1 2 3 4 5 6 7 8 9 10; do
python train_inc_CLIP_CISS.py --num-gpus 8 --config-file configs/ade20k/semantic-segmentation/CLIP_CISS-Base-ADE20K-SemanticSegmentation.yaml OUTPUT_DIR results CONT.BASE_CLS 100 CONT.INC_CLS 5 CONT.MODE overlap SEED 42 CONT.TASK ${t} SOLVER.BASE_LR 0.00008 NAME ade_exp SOLVER.MAX_ITER 4000 SOLVER.IMS_PER_BATCH 16 CONT.DIST.GPD 1.0 CONT.DIST.Q_KD 0.15 TEST.EVAL_PERIOD 200
done
