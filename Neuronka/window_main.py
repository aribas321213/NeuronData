import cv2
import os
import glob
import shutil
import numpy as np
import time
import json
import config1

num = ['']
v = []

subjects = json.load(open(config1.path+'Person.txt', 'r'))

# добавим свою функцию для изменения размера изображения с сохранением соотношения сторон
def my_resize(image, new_width=400):
    ratio = float(new_width) / image.shape[1] # отношение новой ширины 200px к исходной.
    new_shape = (new_width, int(image.shape[0] * ratio))
    resized = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
    return resized

# функция для рисования прямоугольника по координатам левого верхнего угла, ширины и высоты
# cv2.rectangle(img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# функция для написания имени (лейбла) по координатам левого верхнего угла
# cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)`
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def draw_confidence(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# функция для определения лиц
def detect_face(img):
    img_copy = np.copy(img)
# конвертируем изображение в ЧБ, т.к. opencv face detector принимает ЧБ изображения
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

# можно использовать LBP, т.к. он работает быстрее
    #
    #face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

# можно использовать haarcascade для более точного определения
    face_cascade = cv2.CascadeClassifier(config1.path+'haarcascade_frontalface_default.xml')

# Теперь найдем лица на фото с помощью функции detectMultiScale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

# если лиц не обнаружено, вернем исходное изображение
    if len(faces) == 0:
        print("Error")
        return None, None

# с условием, что на изображении находится одно лицо (!) получим его координаты
    (x, y, w, h) = faces[0]

# возвратим только часть с лицом из кадра
    return gray[y:y + w, x:x + h], faces[0]

    #shutil.rmtree('E:\\Neuronka\\Training\\s'+str(label))


def prepare_training_data_new(data_folder_path, faces, labels):
    dirs = os.listdir(data_folder_path)
    subject_dir_path = dirs[-1]

    subject_images_names = os.listdir(subject_dir_path)
    label= len(subjects)-1
# —----ШАГ-3--------
# считаем каждое изображение
# распознаем лицо и добавим его в список faces
    for image_name in subject_images_names:
# игнорируем системные файлы типа .DS_Store
        if image_name.startswith("."):
            continue

# запишем в переменную путь до изображения
# например, image_path = 014_train_data/s1/1.jpg
        image_path = subject_dir_path + "/" + image_name

# читаем изображение
        image = cv2.imread(image_path)

# найдем лицо на фото
        face, rect = detect_face(image)

# —----ШАГ-4--------
# в рамках данного туториала мы будем игнорировать лица, которые не были найдены
        if face is not None:

# добавим лицо в список
            faces.append(face)
# добавим лейбл в список
            labels.append(label)

    return faces, labels


def prepare_training_data(data_folder_path):
# —----ШАГ-1--------
# получим список директорий (папок), содержащихся в папке data_folder_path
    dirs = os.listdir(data_folder_path)
# список лиц
    faces = []
# список лейблов (меток)
    labels = []

# считаем изображения в кадой директории по очереди
    for dir_name in dirs:
# директории каждого человека начинаются с буквы 's', поэтому будем рассматриввть только их и игнорировать
# остальные папки, если они есть
        if not dir_name.startswith("s"):
            continue

# —----ШАГ-2--------
# получим лейбл из имени папки: просто удалим букву 's' и переведем строку в целое число (integer)
        label = int(dir_name.replace("s", ""))
# запишем в переменную путь до директории, содержащей изображения
# например, subject_dir_path = "014_train_data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

# получаем список из имен изображений в папке
        subject_images_names = os.listdir(subject_dir_path)

# —----ШАГ-3--------
# считаем каждое изображение
# распознаем лицо и добавим его в список faces
        for image_name in subject_images_names:
# игнорируем системные файлы типа .DS_Store
            if image_name.startswith("."):
                continue

# запишем в переменную путь до изображения
# например, image_path = 014_train_data/s1/1.jpg
            image_path = subject_dir_path + "/" + image_name

# читаем изображение
            image = cv2.imread(image_path)

# найдем лицо на фото
            face, rect = detect_face(image)

# —----ШАГ-4--------
# в рамках данного туториала мы будем игнорировать лица, которые не были найдены
            if face is not None:
# добавим лицо в список
                faces.append(face)
# добавим лейбл в список
                labels.append(label)

    return faces, labels

def func1(faces, labels):
    [faces, labels] = prepare_training_data_new(config1.path+'Training', faces, labels)
    face_recognizer.train(faces, np.array(labels))

# эта функция идентифицирует человека со входящей фотографии и рисует прямоугольник с именем
def predict(input_img, subject):
    img = input_img.copy()
# определим лицо на фото
    face, rect = detect_face(img)
    pers=json.load(open(config1.path+'Person.txt', 'r'))
    if len(subject)!=len(pers):
        func1(faces, labels)
        subject = pers.copy
    subjects = subject
# если не нашли лицо в кадре вернем исходный кадр
    if face is None:
        return input_img
# идентифицируем лицо
    label, confidence = face_recognizer.predict(face)
# получим имя по лейблу
    label_text = subjects[label]
    num[0] = label
# рисуем прямоугольник
    draw_rectangle(img, rect)
# пишем имя
    t=0
    v=json.load(open(config1.path+'delet.txt', 'r'))


    threshold = 40
    if confidence < threshold:
        for i in range(len(v)):
            if label==v[i]:
                t=1
        if t==0:
            draw_text(img, label_text, rect[0], rect[1] - 5)
            conf_str = str(round(label, 3))
            draw_confidence(img, conf_str, rect[0], rect[1] - 30)
            butto = cv2.waitKey(1) & 0xFF
            if butto == ord("s"):
                print("p")
            #string = json.load(open('E:\\Neuronka\\output.txt', 'r'))
            #if string[-1] != label_text:
            # string.append(label_text)
                #json.dump(string, open('E:\\Neuronka\\output.txt', 'w', encoding='UTF-8'))
            #
        if t==1:

            draw_text(img, 'Unknown', rect[0], rect[1] - 5)
            return img
        t=0

    else:
        draw_text(img, 'Unknown', rect[0], rect[1] - 5)
        return img
# threshold = 300
# if confidence < threshold:
    #draw_text(img, label_text, rect[0], rect[1] - 5)
    #conf_str = str(round(confidence, 3))
    #draw_confidence(img, conf_str, rect[0], rect[1] - 30)
# else:
# draw_text(img, 'Unknown', rect[0], rect[1] - 5)
    return img



print("Preparing data...")
[faces, labels] = prepare_training_data(config1.path+'Training')
print("Data prepared")

# выведем в длины списков лиц и лейблов
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# создаем LBPH face recognizer

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# тренируем модель
face_recognizer.train(faces, np.array(labels))


