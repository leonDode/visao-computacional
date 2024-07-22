# foi necessario a regredir o numpy pra uma versao anterior a 2.0 , pois estava danto conflito com o easyocr
# pip install numpy<2

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import easyocr


import util

# Configuracao do modelo
model_cfg = os.path.join('.', 'yolov3.cfg')
model_pessos = os.path.join('.', 'yolov3.weights')
model_classes = os.path.join('.', 'coco.names')

input_fotos = './fotos'

# Carrega os nomes das classes
with open(model_classes, 'r') as f:
    class_names = [j.strip() for j in f.readlines() if j.strip()]

# Carrega  o modelo
net = cv2.dnn.readNetFromDarknet(model_cfg, model_pessos)

# inicia o leitor OCR
reader = easyocr.Reader(['en'])



for img_name in os.listdir(input_fotos):
    arquivo_img = os.path.join(input_fotos, img_name)
    
    # Carrega a imagem
    img = cv2.imread(arquivo_img)
    H, W, _ = img.shape
    
    # Converte a imagem
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)
    
    net.setInput(blob)
    detections = util.get_outputs(net)
    
    bboxes = []
    class_ids = []
    scores = []
    
    for detection in detections:
        bbox = detection[:4]
        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
        
        bbox_confidence = detection[4]
        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])
        
        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)
    
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)
    


    for bbox_, bbox in enumerate(bboxes):

        
        xc, yc, w, h = bbox
        


        
        # Recortar a placa da imagem original
        plate = img[int(yc - (h / 2)): int(yc + (h / 2)), int(xc - (w / 2)): int(xc + (w / 2)), :].copy()

        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            10)
        
        gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # binariza a imagem da placa,deixando ela em preto e branco 
        _, thresh_plate = cv2.threshold(gray_plate, 64, 255, cv2.THRESH_BINARY_INV)

        output = reader.readtext(thresh_plate)

        print("Saída do EasyOCR:", output)
        for out in output:
            text_bbox, text, text_score = out
            print(f"Texto: {text}, Score: {text_score}")

            if len(text_bbox) == 4:
                (startX, startY), _, (endX, endY), _ = text_bbox

                startX = int(startX + xc - (w / 2))
                startY = int(startY + yc - (h / 2))
                endX = int(endX + xc - (w / 2))
                endY = int(endY + yc - (h / 2))

               
                img = cv2.putText(img, text, (int(startX), int(startY)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 255), 3)

        
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # placa recortada
       # plt.figure()
       # plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))

        # placa binarizada (preto/branco)
       # plt.figure()
       # plt.imshow(cv2.cvtColor(thresh_plate, cv2.COLOR_BGR2RGB))
        
        plt.show()
