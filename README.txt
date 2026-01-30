# SGT-DDI

## Environment
- Python: 3.8
- PyTorch: 1.12.0+cu116
- torchvision: 0.13.0+cu116
- torchaudio: 0.12.0+cu116
- PyTorch Geometric: 2.0.4
- torch-scatter: 2.0.9
- torch-sparse: 0.6.14
- torch-cluster: 1.6.0+pt112cu116
- torch-spline-conv: 1.2.2
- RDKit: 2024.3.5
- NetworkX: 3.1
- graph-tool: 2.71
- fairseq: 0.12.2
- NumPy: 1.24.4
- SciPy: 1.10.1
- scikit-learn: 1.1.1
- pandas: 2.0.3



## Training Command
python train.py --user-dir ./SGTDDI --save-dir ckpts/ddi-1224-24 --ddp-backend=legacy_ddp --dataset-name mydata --dataset-source pyg --data-dir dataset --task graph_prediction_ddi --id-type cycle_graph+path_graph+star_graph+k_neighborhood --ks [8,4,6,2] --sampling-redundancy 6 --criterion binary_logloss --arch sgt_ddi --deepnorm --num-classes 1 --num-workers 16 --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 --lr-scheduler polynomial_decay --power 1 --warmup-updates 640000 --total-num-update 2560000 --lr 2e-4 --end-learning-rate 1e-6 --batch-size 8 --fp16 --data-buffer-size 20 --encoder-layers 24 --encoder-embed-dim 80 --encoder-ffn-embed-dim 80 --encoder-attention-heads 8 --max-epoch 1000 --keep-best-checkpoints 2 --keep-last-epochs 3 --fuse-method concat --do_project --valid-subset valid,test --output-folder /root/SGT-DDI/result/
