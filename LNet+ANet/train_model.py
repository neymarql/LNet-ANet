# 作者：钱隆
# 时间：2022/9/1 23:10


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from sklearn.svm import SVC
import tensorflow_probability as tfp

# 加载并预处理数据集
train_dir = '/data1/CelebA/train'
val_dir = '/data1/CelebA/val'
test_dir = '/data1/CelebA/test'

# 数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 128
val_batchsize = 128
test_batchsize = 128

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(227, 227),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(227, 227),
    batch_size=val_batchsize,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(227, 227),
    batch_size=test_batchsize,
    class_mode='categorical')

# vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# 定义LNeto模型

lneto = keras.Sequential([
    keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    keras.layers.Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')
])
lneto.summary()
# 定义LNets模型

lnets = keras.Sequential([
    keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    keras.layers.Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')
])
#resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
lnets.summary()
# 定义ANet模型
anet = keras.Sequential([
    # resnet,
    keras.layers.Conv2D(20, (4, 4), activation='relu', input_shape=(227, 227, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(40, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(60, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(80, (2, 2), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(40, activation='sigmoid')
])

anet.summary()


# 将LNeto和LNets组合在一起
model_input = keras.layers.Input(shape=(227, 227, 3))
lneto_output = lneto(model_input)
attention_map1 = tf.reduce_mean(lneto_output, axis=-1)

# 对注意力图进行阈值化并提取前10%的区域
threshold = tfp.stats.percentile(attention_map1, 90.0)
mask = tf.greater(attention_map1, threshold)
indices = tf.where(mask)
xmin = tf.reduce_min(indices[:, 0])
ymin = tf.reduce_min(indices[:, 1])
xmax = tf.reduce_max(indices[:, 0])
ymax = tf.reduce_max(indices[:, 1])
offset_height = tf.cast(xmin, tf.int32)
offset_width = tf.cast( ymin, tf.int32)
target_height = tf.cast(xmax-xmin, tf.int32)
target_width = tf.cast(ymax-ymin, tf.int32)
# 裁剪输入图像的相应区域并传递到LNets模型中
cropped_input = tf.image.crop_to_bounding_box(model_input, offset_height, offset_width, target_height, target_width)
cropped_input = tf.keras.layers.experimental.preprocessing.Resizing(227, 227)(cropped_input)
lnets_output = lnets(cropped_input)
attention_map2 = tf.reduce_mean(lneto_output, axis=-1)

# 对注意力图进行阈值化并提取前10%的区域
threshold = tfp.stats.percentile(attention_map2, 90.0)
mask = tf.greater(attention_map2, threshold)
indices = tf.where(mask)
xmin = tf.reduce_min(indices[:, 0])
ymin = tf.reduce_min(indices[:, 1])
xmax = tf.reduce_max(indices[:, 0])
ymax = tf.reduce_max(indices[:, 1])
offset_height = tf.cast(xmin, tf.int32)
offset_width = tf.cast( ymin, tf.int32)
target_height = tf.cast(xmax-xmin, tf.int32)
target_width = tf.cast(ymax-ymin, tf.int32)
# 裁剪输入图像的相应区域并传递到LNets模型中
cropped_input = tf.image.crop_to_bounding_box(model_input, offset_height, offset_width, target_height, target_width)
cropped_input = tf.keras.layers.experimental.preprocessing.Resizing(227, 227)(cropped_input)
anet_output = anet(cropped_input)
# 构建整个模型3
model = keras.Model(inputs=model_input, outputs=anet_output)
model.summary()
"""for layer in vgg16.layers:
    layer.trainable = False
for layer in .layers:
    layer.trainable = False"""
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples/train_generator.batch_size,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples/validation_generator.batch_size)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples/test_generator.batch_size)
print('Test accuracy:', test_acc)

