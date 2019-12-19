TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
CUDA_VISIBLE_DEVICES=1 \
python train_model.py --name test \
    --corpus /PATH/TO/FILE/msvd_corpus_glove.pkl
    --ecores /PATH/TO/FILE/msvd_eco_res_avg_norm.npy \
    --tag    /PATH/TO/FILE/msvd_semantic_tag_eco_res_avg.npy \
    --ref    /PATH/TO/FILE/video_feats/msvd_ref3.pkl \
    --test   ./saves/lr2_msvd_eco_res_avg_norm_semantic_tag_eco_res_avg_ss16-best.ckpt \
    > test.log
    
