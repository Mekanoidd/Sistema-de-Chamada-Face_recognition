from copyreg import pickle
from venv import create
from flask import Flask, render_template, Response, url_for, redirect, request
import cv2
import face_recognition
import csv
import os
import pickle
import numpy as np
import glob
import os
from PIL import Image
from datetime import datetime
import timeit


app=Flask(__name__)
VideoCapture = cv2.VideoCapture(0)

inicio = timeit.default_timer()
now = datetime.now()
#Informa o caminho da imagem de cada aluno, e extração das informações de cada uma
Aluno1_image = face_recognition.load_image_file("photos/Aluno1.jpg")
Aluno1_encoding = face_recognition.face_encodings(Aluno1_image)[0]

Aluno2_image = face_recognition.load_image_file("photos/Aluno2.jpg")
Aluno2_encoding = face_recognition.face_encodings(Aluno2_image)[0]

Aluno3_image = face_recognition.load_image_file("photos/Aluno3.jpg")
Aluno3_encoding = face_recognition.face_encodings(Aluno3_image)[0]

Aluno4_image = face_recognition.load_image_file("photos/Aluno4.jpg")
Aluno4_encoding = face_recognition.face_encodings(Aluno4_image)[0]

Aluno5_image = face_recognition.load_image_file("photos/Aluno5.jpg")
Aluno5_encoding = face_recognition.face_encodings(Aluno5_image)[0]

Aluno6_image = face_recognition.load_image_file("photos/Aluno6.jpg")
Aluno6_encoding = face_recognition.face_encodings(Aluno6_image)[0]


fim = timeit.default_timer()
tempo = fim - inicio
tempos = str(tempo)
pickle.dump(tempos, open("tempotreino.txt", "wb"))
#Lista com os valores das imagens dos alunos
known_face_encoding = [
    Aluno1_encoding,
    Aluno2_encoding,
    Aluno3_encoding,
    Aluno4_encoding,
    Aluno5_encoding,
    Aluno6_encoding
]
#Lista com o nome dos Alunos
known_face_names = [
    "Aluno1",
    "Aluno2",
    "Aluno3",
    "Aluno4",
    "Aluno5",
    "Aluno6"
]

alunos = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


font = cv2.FONT_HERSHEY_SIMPLEX


#Função que marca a presença, que recebe o nome da pessoa reconhecida pelo reconhecedor
def MarcarPresença(name):
#Recebe a data atual como dia-mês-ano
    inicio = timeit.default_timer()
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
#Cria a lista de chamada nomeada com a data atual (dia-mês-ano)
    open(current_date+'.csv','a+')
    with open(current_date+'.csv','r+') as f:
        datalist = f.readlines()
        namelist = []
#Percorre as linhas da tabela recebendo os alunos presetes
        for line in datalist:
            entry = line.split(',')
            namelist.append(entry[0])
#Caso o aluno reconhecido não esteja na lista de chamada, e não seja um desconhecido. Ele é marcado como presente
        if name not in namelist and name != 'desconhecido':
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            fim = timeit.default_timer()
            tempo = fim - inicio
            tempos = str(tempo)
            pickle.dump(tempos, open("temporeconhecimento.txt", "wb"))

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')



#Função que realiza a captura de video e o reconhecimento facial
def gen_frames():
    while True:
#carregamento de câmera
        _,frame = VideoCapture.read()
        small_frame = cv2.resize(frame,(0,0),fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]
#carregamento dos rostos capturados pela camera e busca pelo rosto cadastrado mais parecido
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                name=""
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
#realização da chamada                
                face_names.append(name)
                if name in known_face_names:
                    if name in alunos:
                        alunos.remove(name)
                        MarcarPresença(name)
#exibição da imagem dos rostos detectados
        for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')   
       

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chamada')
def chamada():
    return render_template('chamada.html')

if __name__=='__main__':
    app.run(debug=True)