import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
from PIL import Image
import numpy as np

# Custom waste categories
waste_classes = [
    "a pile of non-recyclable trash like used tissue paper, dirty food wrappers, and greasy paper plates",
    "clean white or colored office paper sheets, magazines, or books ready for recycling",
    "clear or green glass bottles or jars, like wine bottles, jam jars, or broken glass pieces",
    "fresh organic waste including vegetable peels, fruit skins, banana peels, and salad scraps",
    "crumpled plastic water bottles, plastic containers, shopping bags, and cling film",
    "discarded electronic items such as smartphones, earphones, USB cables, and broken remote controls",
    "shiny aluminum soda cans, beer cans, and tin food containers",
    "flattened brown cardboard boxes, shipping cartons, or cereal boxes",
    "used AA batteries, coin cells, or lithium-ion batteries from old electronics",
    "fast food containers made of styrofoam, greasy pizza boxes, and plastic cutlery",
    "a pile of used textiles or clothing waste like old shirts, socks, and torn fabric",
    "a cluster of biodegradable food waste including bread crusts, egg shells, and tea bags",
    "broken ceramic mugs, shattered plates, and porcelain tiles",
    "medical waste like disposable masks, gloves, used bandages, and medicine blister packs",
    "construction debris like bricks, concrete chunks, wood splinters, and nails",
    "garden waste such as leaves, twigs, branches, and grass clippings",
    "hazardous chemical containers like pesticide bottles or bleach cans with warning signs",
    "a tangled bunch of wires, circuit boards, and motherboard scraps from broken electronics",
    "old CDs, DVDs, floppy disks, and scratched discs with labels",
    "ink cartridges, toner boxes, and printer ribbons from office electronic waste"
]


# Load CLIP model + processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Predict
    inputs = processor(text=waste_classes, images=pil_img, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1).cpu().detach().numpy()[0]
    label = waste_classes[np.argmax(probs)]
    conf = probs[np.argmax(probs)]

    # Display on frame
    cv2.putText(frame, f"{label} ({conf:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Waste Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
