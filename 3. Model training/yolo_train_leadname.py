from ultralytics import YOLO

model = YOLO("yolo11x.pt")

results = model.train(data="/dataset_leadname.yaml",
                      project="/runs",
                      name="yolo11_leadname",
                      epochs=100,
                      imgsz=1500,
                      batch=8,
                      device=[0,1,2,3]
                     )