from ultralytics import YOLO

model = YOLO("yolo11x.pt")

results = model.train(data="/dataset_pulse.yaml",
                      project="/runs",
                      name="yolo11_pulse",
                      epochs=100,
                      imgsz=1500,
                      batch=8,
                      device=[0,1,2,3]
                     )