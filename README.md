# Top-p VadCLIP Retrieval

Pipeline trong thư mục này chạy trực tiếp trên video thô ở `/workspace/test`:

1. Decode video va sample frame.
2. Encode tung frame bang CLIP `ViT-B/16`.
3. Dua chuoi frame feature vao VadCLIP checkpoint `/workspace/model_ucf.pth` de lay temporal anomaly scores.
4. Softmax anomaly scores thanh anomaly distribution.
5. Chon tap frame nho nhat co tong xac suat vuot `top-p`.
6. Average embedding cua cac frame duoc chon de tao video embedding.
7. Encode text query bang CLIP text encoder va rank video theo cosine similarity.

## File chinh

- `run_top_p_retrieval.py`

## Luu y

- Lan chay dau script se tu tai CLIP weights `ViT-B/16` vao `/workspace/ablation_study/clip_cache`.
- `model_ucf.pth` khong chua full CLIP backbone, nen viec tai them CLIP weights la can thiet.
- Script dang dung nhanh nhanh cho ablation: anomaly score lay tu branch classifier cua VadCLIP, sau do top-p frame selection va retrieval bang CLIP image/text encoder.
- Python packages toi thieu can them neu moi truong chua co: `ftfy regex Pillow tqdm scipy scikit-learn`.
- Script dung `ffmpeg` de tach frame. May hien tai da co `/usr/bin/ffmpeg`.

## Lenh chay mau

```bash
cd /workspace/ablation_study
python run_top_p_retrieval.py \
  --video-root /workspace/test \
  --checkpoint /workspace/model_ucf.pth \
  --query "a person shoplifting in a store" \
  --top-p 0.7
```

## Output

- Bang xep hang in ra terminal.
- File CSV mac dinh: `/workspace/ablation_study/output/retrieval_results.csv`
