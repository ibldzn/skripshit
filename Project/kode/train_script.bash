for i in {1..20}; do
  yolo classify train \
    model=yolov8n-cls.pt \
    data=tomato-leaf-disease \
    epochs=50 \
    imgsz=224 \
    batch=16 \
    device=0 \
    workers=2 \
    cache=False \
    seed=$i \
    deterministic=True \
    pretrained=True \
    project=runs/yolov8n_seed_experiment \
    name=seed-$i \
    exist_ok=True

  yolo classify train \
    model=yolo11n-cls.pt \
    data=tomato-leaf-disease \
    epochs=50 \
    imgsz=224 \
    batch=16 \
    device=0 \
    workers=2 \
    cache=False \
    seed=$i \
    deterministic=True \
    pretrained=True \
    project=runs/yolo11n_seed_experiment \
    name=seed-$i \
    exist_ok=True
done
