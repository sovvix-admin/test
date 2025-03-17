from ultralytics import YOLO
def main():


    model = YOLO("yolov8n.pt")

    config_file_path = "dataset.yaml"

    project = "valid"
    experiment ="My-Model"

    batch_size = 32

    result = model.train(data=config_file_path,
                         epochs = 1500,
                         name = experiment,
                         batch = batch_size,
                         device = "cuda",
                         patience = 300, 
                         imgsz = 350,
                         verbose = True,
                         val = True)
    
if __name__ == "__main__":
    main()
