# source tools/probe_helpers.sh  (set LQ, ANN, and optionally HQ; set ANNREF to the image the Label.txt boxes are drawn on)
# ev EXPORT NAME SEED [inference args]: restore + OCR with both recognizers, size buckets in pred pixels.
ev(){ local E=$1 N=$2 S=$3; shift 3; torchrun --standalone --nproc_per_node=8 inference_vosr_ablation.py --export $E \
  --input $LQ --output res/$N/s$S --seed $S "$@" && for M in PP-OCRv5_server_rec PP-OCRv4_server_rec; do \
  python evaluate.py --pred res/$N/s$S --lq $LQ --ann $ANN --ann-ref ${ANNREF:-lq} ${HQ:+--gt $HQ} \
  --size-buckets 16 24 32 48 --ocr-model $M --output eval/$N/s$S/$M; done; }
# mk SRC DST 'PYTHON': copy of an export whose model.json token_compression (bound to t, whole config c) is edited;
# coarse_depth is dropped so it is recomputed for the new layout. Weights are symlinked; SRC is never modified.
mk(){ mkdir -p $2 && ln -sf $(realpath $1/model.safetensors) $2/model.safetensors && cp $1/pipeline.json $2/ && \
  python -c 'import json,sys; c=json.load(open(sys.argv[1])); t=c["token_compression"]; exec(sys.argv[3]); c.pop("coarse_depth",None); json.dump(c,open(sys.argv[2],"w"),indent=2)' \
  $1/model.json $2/model.json "$3"; }
# ko SRC DST: detail-stream knock-out (detail_stream.exit.* and every detail_stream.blocks.*.write.* set to zero).
ko(){ mkdir -p $2 && cp $1/model.json $1/pipeline.json $2/ && python -c 'import sys,torch; from safetensors.torch import load_file,save_file; s=load_file(sys.argv[1]); save_file({k:(torch.zeros_like(v) if k.startswith("detail_stream.exit.") or (k.startswith("detail_stream.blocks.") and ".write." in k) else v) for k,v in s.items()},sys.argv[2])' \
  $1/model.safetensors $2/model.safetensors; }
