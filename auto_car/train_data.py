import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 데이터 경로 설정
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# 이미지 데이터 생성기 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # 클래스 모드를 categorical로 설정
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # 클래스 모드를 categorical로 설정
)

# 사전 학습된 MobileNetV2 모델 로드 및 미세 조정
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # 클래스 수를 4로 변경

model = Model(inputs=base_model.input, outputs=predictions)

# 일부 층을 학습 가능하게 설정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# 모델 저장
model.save('traffic_light_and_lane_model.h5')
