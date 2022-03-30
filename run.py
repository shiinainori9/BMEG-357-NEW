from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="dataset")
trainer.setTrainConfig(object_names_array=["sicklecell", "redbloodcell"], batch_size=2, num_experiments=20,
                       train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()