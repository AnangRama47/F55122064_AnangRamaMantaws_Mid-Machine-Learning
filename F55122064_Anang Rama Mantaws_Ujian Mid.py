import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Mendefinisikan fungsi untuk mengambil gambar dan label dari folder
def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = os.listdir(folder)
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(folder, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))  # Ubah ukuran gambar sesuai kebutuhan
                images.append(img)
                labels.append(idx)  # Label sesuai dengan indeks kelas
    return np.array(images), np.array(labels)

# Load dataset dari folder
folder_path = 'Shoe vs Sandal vs Boots'
images, labels = load_images_from_folder(folder_path)

# Split data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Konversi gambar menjadi vektor
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melatih model SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluasi model SVM
svm_score = svm_model.score(X_test, y_test)
print("SVM Accuracy:", svm_score)

# Melatih model K-NN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluasi model K-NN
knn_score = knn_model.score(X_test, y_test)
print("K-NN Accuracy:", knn_score)
