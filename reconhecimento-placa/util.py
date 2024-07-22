import numpy as np
import cv2


def NMS(boxes, class_ids, confidences, overlapThresh = 0.5):

    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    confidences = np.asarray(confidences)

   
    if len(boxes) == 0:
        return [], [], []

    # Calcula as coordenadas dos cantos 
    x1 = boxes[:, 0] - (boxes[:, 2] / 2)  
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)  
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)  
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)  

    # Calcula as areas das caixas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
      
        temp_indices = indices[indices != i]

        # Calcula as coordenadas de intersecao
        xx1 = np.maximum(box[0] - (box[2] / 2), boxes[temp_indices, 0] - (boxes[temp_indices, 2] / 2))
        yy1 = np.maximum(box[1] - (box[3] / 2), boxes[temp_indices, 1] - (boxes[temp_indices, 3] / 2))
        xx2 = np.minimum(box[0] + (box[2] / 2), boxes[temp_indices, 0] + (boxes[temp_indices, 2] / 2))
        yy2 = np.minimum(box[1] + (box[3] / 2), boxes[temp_indices, 1] + (boxes[temp_indices, 3] / 2))

        # Calcular largura e altura da interseção
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Calcula a area de sobreposicao
        overlap = (w * h) / areas[temp_indices]
      
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]

  
    return boxes[indices], class_ids[indices], confidences[indices]


def get_outputs(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    outs = net.forward(output_layers)

    # Filtra as deteccoes com baixa confianca
    outs = [c for out in outs for c in out if c[4] > 0.1]

    return outs


def draw(bbox, img):

    xc, yc, w, h = bbox
    img = cv2.rectangle(img,
                        (xc - int(w / 2), yc - int(h / 2)),
                        (xc + int(w / 2), yc + int(h / 2)),
                        (0, 255, 0), 20)

    return img
