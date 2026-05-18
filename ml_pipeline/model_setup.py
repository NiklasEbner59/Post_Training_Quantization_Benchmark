import tensorflow as tf
import numpy as np

# loading the pretrained MobileNetV2 model from Keras applications
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# initialize TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Converting without additional optimizations (float32)
tflite_model_f32 = converter.convert()

# save the float32 TFLite model
with open('mobilenet_v2_f32.tflite', 'wb') as f:
    f.write(tflite_model_f32)

# function for representative dataset generation for quantization
def representative_data_gen():
    for _ in range(100):
        # normally real images, but for the benchmark, we use random data
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_data_gen

# Force Integer Arithmetic (important for CPU advantage)
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

# Convert
tflite_model_int8 = converter_int8.convert()

# save the int8 TFLite model
with open('mobilenet_v2_int8.tflite', 'wb') as f:
    f.write(tflite_model_int8)

