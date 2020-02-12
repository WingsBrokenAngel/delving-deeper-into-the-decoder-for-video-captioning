TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
CUDA_VISIBLE_DEVICES=1 \
python train_model.py --name test \
    --corpus /PATH/TO/FILE/msvd_corpus_glove.pkl \
    --ecores /PATH/TO/FILE/msvd_eco_norm.npy \
    --tag    /PATH/TO/FILE/msvd_semantic_tag_res_avg.npy \
    --ref    /PATH/TO/FILE/msvd_ref3.pkl \
    --test   ./saves/lr2_msvd_eco_norm_semantic_tag_res_avg_ss16_gamma_8_v2-best.ckpt \
    > test.log
    
