import numpy as np
import imgaug.augmenters as iaa

# 加载训练集和测试集原始数据
train_data = np.load('./cifar10/train_data.npy')
train_label = np.load('./cifar10/train_label.npy')
test_data = np.load('./cifar10/test_data.npy')
test_label = np.load('./cifar10/test_label.npy')

# 从5w训练集中随机抽取4w张样本
index = np.random.choice(50000, 40000, replace=False)
train_data = train_data[index]
train_label = train_label[index]

# 利用imgaug进行图像增广
augment_a = iaa.OneOf([
    iaa.Rain(speed=0.3),
    iaa.imgcorruptlike.Snow(severity=2),
    iaa.imgcorruptlike.Frost(severity=2),
    iaa.imgcorruptlike.Saturate(severity=4)
])

augment_b = iaa.OneOf([
    iaa.arithmetic.Dropout(p=0.075),
    iaa.CoarseDropout(0.075, size_percent=1., per_channel=1.),
    iaa.Snowflakes(flake_size=0.3, speed=0.01)
])

augment_c = iaa.OneOf([
    iaa.arithmetic.AdditiveGaussianNoise(scale=15.0, per_channel=True),
    iaa.ElasticTransformation(alpha=0.625, sigma=0.25),
    iaa.KMeansColorQuantization(n_colors=12)
])

augment_d = iaa.OneOf([
    iaa.blur.GaussianBlur(sigma=1.0),
    iaa.imgcorruptlike.MotionBlur(severity=1),
    iaa.JpegCompression(compression=70),
    iaa.imgcorruptlike.Pixelate(severity=1)
])

data_aug1 = augment_a(images=train_data[: 10000])
data_aug2 = augment_b(images=train_data[10000: 20000])
data_aug3 = augment_c(images=train_data[20000: 30000])
data_aug4 = augment_d(images=train_data[30000:])

# 对抗扰动
delta_linf = np.load('delta_linf.npy')
delta_l2 = np.load('delta_l2.npy')
data_aug5 = np.clip(test_data.astype(np.int32) + delta_linf + delta_l2, 0, 255).astype(np.uint8)

# 最终数据集
final_data = np.vstack([data_aug1, data_aug2, data_aug3, data_aug4, data_aug5])
final_label = np.vstack([train_label, test_label])
np.save('final_data.npy', final_data)
np.save('final_label.npy', final_label)
