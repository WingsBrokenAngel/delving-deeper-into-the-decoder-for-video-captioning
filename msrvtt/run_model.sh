TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
CUDA_VISIBLE_DEVICES=8 \
python train_model.py --name test \
    --corpus /data2/chenhaoran/video_feats/msrvtt_corpus_glove.pkl \
    --ecores /data2/chenhaoran/video_feats/msrvtt_eco_res_avg_norm.npy \
    --tag    /data2/chenhaoran/video_feats/msrvtt_semantic_tag_eco_res_avg.npy \
    --ref    /data2/chenhaoran/video_feats/msrvtt_ref.pkl \
    --test   ./saves/msrvtt_eco_res_avg_norm_semantic_tag_eco_res_avg_gamma_4_v2-best.ckpt \
    > test.log
