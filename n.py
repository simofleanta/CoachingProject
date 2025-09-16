import torch
from ultralytics import YOLO
import networkx as nx
import matplotlib.pyplot as plt
import cv2  # pentru citirea imaginilor

# 1. Încarci un model YOLOv8 pre-antrenat
model = YOLO("yolov8l.pt")  # varianta mică

# 2. Încarci poza cu OpenCV
img_path = r"C:\Sust\PY\SUSTAI-artificial-intelligence-research-sphinx_carbon_emissions_detection\citations\img_path.jpg"
image = cv2.imread(img_path)

# 3. Rulezi inferența pe imagine
results = model(image)

# 4. Extragi obiectele detectate
labels = []
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        labels.append(label)

# 5. Creezi un graph cu obiectele ca noduri
G = nx.Graph()

for obj in labels:
    G.add_node(obj)

# 6. Adaugi muchii între obiecte care apar împreună
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        G.add_edge(labels[i], labels[j])

# 7. Vizualizezi graful
plt.figure(figsize=(8,6))
nx.draw(G, with_labels=True, node_color="lightblue", font_size=10, node_size=1500)
plt.show()
