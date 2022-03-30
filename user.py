from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("dataset/models/")
detector.setJsonPath("dataset/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="3958.jpg", output_image_path="ima-detected.jpg", minimum_percentage_probability=90)
