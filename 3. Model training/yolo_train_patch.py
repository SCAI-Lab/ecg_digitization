from ultralytics import YOLO

model = YOLO("yolo11x-seg.pt")

results = model.train(data="/dataset_patch.yaml",
                      project="/runs",
                      name="yolo11_patch",
                      epochs=100,
                      imgsz=1700,
                      batch=8,
                      device=[0,1,2,3]
                     )