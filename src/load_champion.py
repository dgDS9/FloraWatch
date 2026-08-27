import mlflow


MODEL_NAME = "FloraWatchClassifier"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

model_uri = f"models:/{MODEL_NAME}@champion"

print(f"Loading model: {model_uri}")

model = mlflow.tensorflow.load_model(model_uri)

print("Champion loaded successfully ✅")
print("Model type:", type(model))
print("Input shape:", model.input_shape)