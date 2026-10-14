#############on-plant_pepper_phenotyping###########

### STEP 1 — 라벨링

conda activate easydam_py310
python pepper_sam_server.py --img_dir /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_dataset/train/images/ --sam_model vit_h

conda activate easydam_py310
python pepper_sam_server.py --img_dir /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_dataset/val/images/ --sam_model vit_h


# → http://localhost:5000
* Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.30.13:5000


**단축키:**
| 키 | 클래스 | 설명 |
|---|---|---|
| `L` | leaf | 잎 |
| `B` | bar | 수직 막대 (계통 구분) |
| `G` | green_pepper | 녹색 고추 |
| `R` | red_pepper | 빨강 고추 |
| 우클릭 | 네거티브 | 제외 영역 |

### STEP 2 — 변환 + 학습

python 02_convert_pepper.py --dataset_root /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_dataset
python 03_train_pepper.py --dataset_root /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_dataset --model yolov8m


# 1. 기존 JSON 점 보간 (8~29점 → 72점)
python interpolate_polygons.py \
    --dataset_root /dongoh_RnD/linux/gvfs/CAL/cu/cucumber_dataset





### STEP 3 — 추론 (계통별 집계)
python 04_infer_pepper.py \
    --model /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_phenotype_pipeline/runs/segment/runs/segment/pepper/weights/best.pt \
    --img_dir /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_dataset/test/images \
    --plants_per_variety 6 \
    --out_dir results_pepper/

---

## 출력 결과

| 파일 | 내용 |
|---|---|
| `results_by_variety.csv` | 계통별 잎수, 녹색/빨강 고추수, 잎면적 |
| `results_by_image.csv` | 이미지별 상세 카운트 |
| `annotated/*.jpg` | 마스킹된 결과 이미지 |
| `summary_chart.png` | 계통별 비교 차트 3종 |

**bar(막대)가 감지되면 자동으로 새 계통으로 전환**되고, 계통당 6개체로 나눈 평균값이 CSV에 저장.


grep -n "add_argument\|checkpoint\|\.pth" pepper_sam_server.py

python pepper_sam_server.py --img_dir /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_dataset/train/images/ --sam_model vit_h



# 1. 기존 JSON 점 보간 (8~29점 → 72점)
python interpolate_polygons.py \
    --dataset_root /dongoh_RnD/linux/gvfs/CAL/cu/cucumber_dataset

# 3. 변환 + 재학습
python 02_convert_cucumber.py \
    --dataset_root /dongoh_RnD/linux/gvfs/CAL/cu/cucumber_dataset

python 03_train_cucumber.py \
    --dataset_root /dongoh_RnD/linux/gvfs/CAL/cu/cucumber_dataset \
    --model yolov8m --epochs 100 --imgsz 1280





100 epochs completed in 0.084 hours.
Optimizer stripped from /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_phenotype_pipeline/runs/segment/runs/segment/pepper/weights/last.pt, 54.9MB
Optimizer stripped from /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_phenotype_pipeline/runs/segment/runs/segment/pepper/weights/best.pt, 54.9MB

Validating /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_phenotype_pipeline/runs/segment/runs/segment/pepper/weights/best.pt...
Ultralytics 8.4.23 🚀 Python-3.10.14 torch-2.5.1+cu121 CUDA:0 (NVIDIA L40S, 45468MiB)
YOLOv8m-seg summary (fused): 106 layers, 27,224,700 parameters, 0 gradients, 104.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 1/2 1.4it                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 7.2it/s 0.3s
                   all         20        442      0.857      0.735      0.842      0.674      0.877      0.741      0.854      0.664
                   bar         20         20      0.921       0.95      0.952       0.74      0.928       0.95      0.952      0.799
          green_pepper         18        302      0.789       0.53      0.682      0.541      0.823       0.54       0.71      0.532
            red_pepper         19        120      0.861      0.725      0.893      0.742      0.881      0.733      0.898       0.66
Speed: 0.2ms preprocess, 5.7ms inference, 0.0ms loss, 1.4ms postprocess per image
Results saved to /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_phenotype_pipeline/runs/segment/runs/segment/pepper

완료! 최고 모델: runs/segment/pepper/weights/best.pt
다음: python 04_infer_pepper.py --model runs/segment/pepper/weights/best.pt --img_dir ...
(easydam_py310) [krjo@dongph_genome pepper_phenotype_pipeline]$ /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_phenotype_pipeline/runs/segment/runs/segment/pepper/weights/
bash: /dongoh_RnD/linux/gvfs/CAL/pe_sam/pepper_phenotype_pipeline/runs/segment/runs/segment/pepper/weights/: Is a directory
(easydam_py310) [krjo@dongph_genome pepper_phenotype_pipeline]$ 














====================================== 
==========================================
conda create -n anylabeling python=3.10 -y
conda activate anylabeling
pip install anylabeling

FILE=$(python -c "import anylabeling, os; print(os.path.join(os.path.dirname(anylabeling.__file__), 'views/labeling/label_widget.py'))")
echo "$FILE"

cp "$FILE" "$FILE.bak"

sed -i 's/LABEL_COLORMAP = imgviz.label_colormap()/LABEL_COLORMAP = imgviz.label_colormap().copy()/' "$FILE"

grep -n "LABEL_COLORMAP = imgviz" "$FILE"



pip list 2>/dev/null | grep -i -E "opencv|pyqt|qt"

echo $LD_LIBRARY_PATH

#env 활성화 시 자동으로 올바른 경로를 앞에 붙이기
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat << 'EOF' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat << 'EOF' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
unset _OLD_LD_LIBRARY_PATH
EOF

#적용:
conda deactivate
conda activate anylabeling
echo $LD_LIBRARY_PATH

anylabeling











