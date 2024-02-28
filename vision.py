import textwrap
import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import time
import numpy as np

model = AutoModel.from_pretrained('MiniCPM-V', trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('MiniCPM-V', trust_remote_code=True)
# model.eval()
model = model.to(device='mps', dtype=torch.float16)  # for Apple Silicon

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cv2.namedWindow('Live Camera Feed', cv2.WINDOW_NORMAL)


def draw_multiline_text(image, text, position, font, font_scale, font_color, line_spacing):
    x, y = position
    line_height = int(font_scale * 40)
    lines = textwrap.wrap(text, width=40)
    for line in lines:
        cv2.putText(image, line, (x, y), font, font_scale, font_color, 1, lineType=cv2.LINE_AA)
        y += line_height + line_spacing


response_text, time_cost = '', ''
try:
    while True:

        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        draw_multiline_text(frame, response_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 5)
        draw_multiline_text(frame, time_cost, (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 5)

        cv2.imshow('Live Camera Feed', frame)
        time_to_wait = max(0.5 - (time.time() - start_time), 0)
        if cv2.waitKey(max(int(time_to_wait * 1000), 1)) & 0xFF == ord('q'):
            break

        model_start_time = time.time()
        # question = f'Tell me what is this person doing? And this is his previous action: {response_text}. Now predict the next action.'
        question = f'Tell me what is this person doing?'
        msgs = [{'role': 'user', 'content': question}]

        res, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7
        )

        model_end_time = time.time()

        response_text = f'{res}'
        time_cost = f'{model_end_time - model_start_time:.2f}s'

finally:
    cap.release()
    cv2.destroyAllWindows()
