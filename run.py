import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
import matplotlib.pyplot as plt
import os

# Load and concatenate datasets
sound_maps = np.concatenate([
    np.load("/work/yz886/J-wave/data/sound_maps.npy"),
    np.load("/work/yz886/J-wave/data/sound_maps_1.npy"),
    np.load("/work/yz886/J-wave/data/sound_maps_2.npy"),
    np.load("/work/yz886/J-wave/data/sound_maps_3.npy")
], axis=0)

srcs = np.concatenate([
    np.load("/work/yz886/J-wave/data/srcs.npy"),
    np.load("/work/yz886/J-wave/data/srcs_1.npy"),
    np.load("/work/yz886/J-wave/data/srcs_2.npy"),
    np.load("/work/yz886/J-wave/data/srcs_3.npy")
], axis=0)

fields = np.concatenate([
    np.load("/work/yz886/J-wave/data/fields.npy"),
    np.load("/work/yz886/J-wave/data/fields_1.npy"),
    np.load("/work/yz886/J-wave/data/fields_2.npy"),
    np.load("/work/yz886/J-wave/data/fields_3.npy")
], axis=0)

domains = np.concatenate([
    np.load("/work/yz886/J-wave/data/domains.npy"),
    np.load("/work/yz886/J-wave/data/domains_1.npy"),
    np.load("/work/yz886/J-wave/data/domains_2.npy"),
    np.load("/work/yz886/J-wave/data/domains_3.npy")
], axis=0)

meta_data = np.concatenate([
    np.load("/work/yz886/J-wave/data/meta_data.npy"),
    np.load("/work/yz886/J-wave/data/meta_data_1.npy"),
    np.load("/work/yz886/J-wave/data/meta_data_2.npy"),
    np.load("/work/yz886/J-wave/data/meta_data_3.npy")
], axis=0)

# Print shapes of loaded datasets
print("Loaded sound maps:", sound_maps.shape)
print("Loaded srcs:", srcs.shape)
print("Loaded fields:", fields.shape)
print("Loaded domains:", domains.shape)
print("Loaded meta data:", meta_data.shape)


def normalize_data_per_sample(sound_maps, srcs, fields, domains):
    """Extract real and imaginary parts, normalize, and prepare input-output data."""

    # 1. Extract real and imaginary parts from srcs and fields
    srcs_real = np.real(srcs)
    srcs_imag = np.imag(srcs)

    fields_real = np.real(fields)
    fields_imag = np.imag(fields)

    # 2. Normalize sound maps (Z-score normalization per sample)
    sound_maps_normalized = np.array([
        sample / np.max(np.abs(sample)) for sample in sound_maps
    ])

    # 3. Normalize real part of fields (Z-score normalization)
    srcs_real_normalized = np.array([
        sample / np.max(np.abs(sample)) for sample in srcs_real
    ])

    fields_real_normalized = np.array([
        sample / np.max(np.abs(sample)) for sample in fields_real
    ])

    # 4. Normalize imaginary part of fields (Z-score normalization)
    srcs_imag_normalized = np.array([
        sample / np.max(np.abs(sample)) for sample in srcs_imag
    ])

    fields_imag_normalized = np.array([
        sample / np.max(np.abs(sample)) for sample in fields_imag
    ])

    # 5. Concatenate sound maps with real and imaginary parts of srcs (input)
    sound_maps_expanded = np.expand_dims(sound_maps_normalized, axis=-1)
    srcs_real_expanded = np.expand_dims(srcs_real_normalized, axis=-1)

    # Concatenate along the last axis
    inputs = np.concatenate([sound_maps_expanded, srcs_real_expanded, domains], axis=-1)


    # 6. Concatenate real and imaginary parts of fields (output)
    outputs = np.stack([fields_real_normalized], axis=-1)

    return inputs, outputs


# Apply normalization and prepare input-output data
inputs, outputs = normalize_data_per_sample(sound_maps, srcs, fields, domains)

# Verify the shapes of inputs and outputs
print(f"Inputs shape: {inputs.shape}")  # Should be (1000, 512, 512, 2)
print(f"Outputs shape: {outputs.shape}")  # Should be (1000, 512, 512, 1)

# Split the data into training and test sets with shuffling enabled (default)
(train_input, test_input,
 train_output, test_output) = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42, shuffle=True
)

# Display the shapes of the resulting splits
print(f"Training input: {train_input.shape}")
print(f"Test input: {test_input.shape}")
print(f"Training output: {train_output.shape}")
print(f"Test output: {test_output.shape}")

# Data generator function
def data_generator(inputs, outputs):
    """
    Yield inputs (sound map, srcs real + imaginary) and outputs (fields real + imaginary).
    """
    for input_sample, output_sample in zip(inputs, outputs):
        yield input_sample, output_sample

# Define the output signature
output_signature = (
    tf.TensorSpec(shape=(256, 256, 4), dtype=tf.float32),  # Input: sound map + srcs real/imag
    tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)   # Output: fields real + imag
)

# Create TensorFlow datasets for training and testing
train_data = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=output_signature,
    args=(train_input, train_output)
)

test_data = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=output_signature,
    args=(test_input, test_output)
)

# Verify the dataset shapes
for sample_input, sample_output in train_data.take(1):
    print(f"Input sample shape: {sample_input.shape}")   # Expected: (256, 256, 4)
    print(f"Output sample shape: {sample_output.shape}") # Expected: (256, 256, 1)

# Prepare datasets
train_data = train_data.shuffle(len(train_input)).batch(64).repeat().prefetch(tf.data.AUTOTUNE)
test_data = test_data.shuffle(len(test_input)).batch(16).repeat().prefetch(tf.data.AUTOTUNE)


for batch in train_data.take(1):  # Take a single batch
    input, output = batch
    print(f"Input sample shape: {input.shape}, dtype: {input.dtype}")
    print(f"Output sample shape: {output.shape}, dtype: {output.dtype}")


## Building the model Architecture - DO NOT CHANGE
def stereoDepthNet():
        with tf.name_scope('layer'):
            n_filters = 16

            left_in = Input(shape = (256,256,4),name = 'left_in')

            l_conv1 = Conv2D(n_filters * 1,(3,3),activation = 'relu',padding = 'same',name = 'left_conv1')(left_in)
            l_conv1 = BatchNormalization()(l_conv1)
            l_conv2 = Conv2D(n_filters * 1,(3,3),activation = 'relu',padding = 'same')(l_conv1)
            l_conv2 = BatchNormalization()(l_conv2)
            l_pool1 = MaxPooling2D(pool_size = (2,2),padding = 'same')(l_conv2)

            l_conv3 = Conv2D(n_filters * 2,(3,3),activation = 'relu',padding = 'same')(l_pool1)
            l_conv3 = BatchNormalization()(l_conv3)
            l_conv4 = Conv2D(n_filters * 2,(3,3),activation = 'relu',padding = 'same')(l_conv3)
            l_conv4 = BatchNormalization()(l_conv4)
            l_pool2 = MaxPooling2D(pool_size = (2,2),padding = 'same')(l_conv4)

            l_conv5 = Conv2D(n_filters * 4,(3,3),activation = 'relu',padding = 'same')(l_pool2)
            l_conv5 = BatchNormalization()(l_conv5)
            l_conv6 = Conv2D(n_filters * 4,(3,3),activation = 'relu',padding = 'same')(l_conv5)
            l_conv6 = BatchNormalization()(l_conv6)
            l_pool3 = MaxPooling2D(pool_size = (2,2),padding = 'same')(l_conv6)

            l_conv7 = Conv2D(n_filters * 8,(3,3),activation = 'relu',padding = 'same')(l_pool3)
            l_conv7 = BatchNormalization()(l_conv7)
            l_conv8 = Conv2D(n_filters * 8,(3,3),activation = 'relu',padding = 'same')(l_conv7)
            l_conv8 = BatchNormalization()(l_conv8)
            l_pool4 = MaxPooling2D(pool_size = (2,2),padding = 'same')(l_conv8)

            l_conv9 = Conv2D(n_filters * 16,(3,3),activation = 'relu',padding = 'same')(l_pool4)
            l_conv9 = BatchNormalization()(l_conv9)
            l_conv10 = Conv2D(n_filters * 16,(3,3),activation = 'relu',padding = 'same')(l_conv9)
            l_conv10 = BatchNormalization()(l_conv10)
            l_pool5 = MaxPooling2D(pool_size = (2,2),padding = 'same')(l_conv10)

            l_conv11 = Conv2D(n_filters * 32,(3,3),activation = 'relu',padding = 'same')(l_pool5)
            l_conv11 = BatchNormalization()(l_conv11)
            l_conv12 = Conv2D(n_filters * 32,(3,3),activation = 'relu',padding = 'same')(l_conv11)
            l_conv12 = BatchNormalization()(l_conv12)
            l_pool6 = MaxPooling2D(pool_size = (2,2),padding = 'same')(l_conv12)

            l_conv13 = Conv2D(n_filters * 64,(3,3),activation = 'relu',padding = 'same')(l_pool6)
            l_conv13 = BatchNormalization()(l_conv13)
            l_conv14 = Conv2D(n_filters * 64,(3,3),activation = 'relu',padding = 'same')(l_conv13)
            l_conv14 = BatchNormalization()(l_conv14)


            flattened = Flatten()(l_conv14)
            fc1 = Dense(4*4*n_filters * 64, activation='relu')(flattened)
            fc2 = Dense(4*4*n_filters * 64, activation='relu')(fc1)
            reshaped = Reshape((4, 4, n_filters * 64))(fc2)  # Reshape to fit the input for the upsampling path


            l_out1 = Conv2D(n_filters * 64,(3,3),activation = 'relu',padding = 'same')(reshaped)
            l_out1 = BatchNormalization()(l_out1)
            l_out1 = Conv2D(n_filters * 64,(3,3),activation = 'relu',padding = 'same')(l_out1)
            l_out1 = BatchNormalization()(l_out1)

            l_up1 = concatenate([UpSampling2D(size=(2, 2))(l_out1), l_conv12], axis=3)

            l_out2 = Conv2D(n_filters * 32,(3,3),activation = 'relu',padding = 'same')(l_up1)
            l_out2 = BatchNormalization()(l_out2)
            l_out2 = Conv2D(n_filters * 32,(3,3),activation = 'relu',padding = 'same')(l_out2)
            l_out2 = BatchNormalization()(l_out2)

            l_up2 = concatenate([UpSampling2D(size=(2, 2))(l_out2), l_conv10], axis=3)

            l_out3 = Conv2D(n_filters * 16,(3,3),activation = 'relu',padding = 'same')(l_up2)
            l_out3 = BatchNormalization()(l_out3)
            l_out3 = Conv2D(n_filters * 16,(3,3),activation = 'relu',padding = 'same')(l_out3)
            l_out3 = BatchNormalization()(l_out3)

            l_up3 = concatenate([UpSampling2D(size=(2, 2))(l_out3), l_conv8], axis=3)

            l_out4 = Conv2D(n_filters * 8,(3,3),activation = 'relu',padding = 'same')(l_up3)
            l_out4 = BatchNormalization()(l_out4)
            l_out4 = Conv2D(n_filters * 8,(3,3),activation = 'relu',padding = 'same')(l_out4)
            l_out4 = BatchNormalization()(l_out4)

            l_up4 = concatenate([UpSampling2D(size=(2, 2))(l_out4), l_conv6], axis=3)

            l_out5 = Conv2D(n_filters * 4,(3,3),activation = 'relu',padding = 'same')(l_up4)
            l_out5 = BatchNormalization()(l_out5)
            l_out5 = Conv2D(n_filters * 4,(3,3),activation = 'relu',padding = 'same')(l_out5)
            l_out5 = BatchNormalization()(l_out5)

            l_up5 = concatenate([UpSampling2D(size=(2, 2))(l_out5), l_conv4], axis=3)

            l_out6 = Conv2D(n_filters * 2,(3,3),activation = 'relu',padding = 'same')(l_up5)
            l_out6 = BatchNormalization()(l_out6)
            l_out6 = Conv2D(n_filters * 2,(3,3),activation = 'relu',padding = 'same')(l_out6)
            l_out6 = BatchNormalization()(l_out6)

            l_up6 = concatenate([UpSampling2D(size=(2, 2))(l_out6), l_conv2], axis=3)

            l_out7 = Conv2D(n_filters * 1,(3,3),activation = 'relu',padding = 'same')(l_up6)
            l_out7 = BatchNormalization()(l_out7)
            l_out7 = Conv2D(n_filters * 1,(3,3),activation = 'relu',padding = 'same')(l_out7)
            l_out7 = BatchNormalization()(l_out7)

            l_out = Conv2D(1,(1,1),activation = 'tanh',padding = 'same',name = 'left_output')(l_out7)


            model = Model(inputs = [left_in],outputs = [l_out])

            return model

# Instantiate and compile the model
model = stereoDepthNet()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'left_output': MeanSquaredError()},
    metrics={'left_output': [RootMeanSquaredError(), MeanAbsoluteError()]}
)

# Train the model
history = model.fit(
    train_data, epochs=100, steps_per_epoch=125,
    validation_data=test_data, validation_steps=125
)



# Create the directory to save images (if not already created)
save_dir = "/work/yz886/J-wave/L2/"
os.makedirs(save_dir, exist_ok=True)


# Plotting function for metrics
def plot_and_save_metrics(history, metric_name, title, filename):
    """Plot the training and validation metrics and save the plot."""
    plt.figure(figsize=(10, 6))

    # Plot for 'left_output' (Real part)
    plt.plot(history.history[f'{metric_name}'], label='Real Part - Training')
    plt.plot(history.history[f'val_{metric_name}'], label='Real Part - Validation')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.replace('_', ' ').capitalize())
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

# Plot and save RMSE
plot_and_save_metrics(
    history, 'root_mean_squared_error',
    'Root Mean Square Error (RMSE)', 'rmse_plot.png'
)

# Plot and save MAE
plot_and_save_metrics(
    history, 'mean_absolute_error',
    'Mean Absolute Error (MAE)', 'mae_plot.png'
)

# Randomly select 5 indices from the train and 5 from the test dataset
train_indices = np.random.choice(train_input.shape[0], 5, replace=False)
test_indices = np.random.choice(test_input.shape[0], 5, replace=False)

# Function to predict and visualize results
def predict_and_visualize(inputs, outputs, index, dataset_type):
    """Predict using the model and visualize the original and predicted fields."""

    # Expand dimensions of the input to add batch size of 1 (1, 512, 512, 3)
    input_sample = np.expand_dims(inputs[index], axis=0)

    # Predict the output
    pred_output = model.predict(input_sample)

    # Extract the original data for visualization
    sound_map = inputs[index, :, :, 0]  # Sound map
    src_real = inputs[index, :, :, 1]   # Real part of src
    field_real = outputs[index, :, :, 0]  # Real part of field
    pred_field_real = pred_output[0, :, :, 0]  # Predicted real part of field

    # Plot the input and output data
    fig, axes = plt.subplots(1, 4, figsize=(18, 12))

    # 1. Sound Map Plot
    im0 = axes[0].imshow(sound_map, cmap='viridis')
    axes[0].set_title('Sound Map')
    plt.colorbar(im0, ax=axes[0])

    # 2. Real Part of Src Plot
    im1 = axes[1].imshow(src_real, cmap='seismic')
    axes[1].set_title('Real Part of Src')
    plt.colorbar(im1, ax=axes[1])

    # 3. Real Part of Field Plot
    im2 = axes[2].imshow(field_real, cmap='seismic')
    axes[2].set_title('Real Part of Field')
    plt.colorbar(im2, ax=axes[2])

    # 4. Predicted Real Part of Field Plot
    im3 = axes[3].imshow(pred_field_real, cmap='seismic')
    axes[3].set_title('Predicted Real Part of Field')
    plt.colorbar(im3, ax=axes[3])

    # Adjust layout and show the plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{dataset_type}_prediction_{index}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Free memory after saving
    print(f"Saved visualization for {dataset_type} index {index} at {save_path}")

# Visualize predictions for 5 random train and 5 random test samples
for idx in train_indices:
    predict_and_visualize(train_input, train_output, idx, "train")

for idx in test_indices:
    predict_and_visualize(test_input, test_output, idx, "test")
