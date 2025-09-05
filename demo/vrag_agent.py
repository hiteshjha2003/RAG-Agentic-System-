import base64
import json
import re
import math
from io import BytesIO
from openai import OpenAI
from PIL import Image, ImageDraw
import os

class VRAG:
    def __init__(self,
                 base_url='https://api.sambanova.ai/v1',
                 generator=True,
                 api_key='fxxxxxxxx',):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.max_steps = 10
        self.generator = generator
        self.image_raw = []
        self.image_input = []

    # ---------- unchanged helper ----------
    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError("Unsupported image type")

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        return image, f"data:image/jpeg;base64,{encoded}"

    # ---------- main entry point ----------
    def run(self, messages, *, use_reasoning=True):
        self.image_raw.clear()
        self.image_input.clear()
        max_steps = self.max_steps

        # Decode and store initial images from first user message content
        if messages and isinstance(messages[0].get("content"), list):
            for part in messages[0]["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_data = part["image_url"].get("url", "")
                    try:
                        image_bytes = base64.b64decode(image_data.split(",")[-1])
                        img = Image.open(BytesIO(image_bytes))
                        self.image_raw.append(img)
                        self.image_input.append(img)
                    except Exception:
                        pass

        while max_steps > 0:
            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct",
                messages=messages,
                stream=False
            )
            response_content = response.choices[0].message.content

            if self.generator:
                yield 'think', response_content, None

            # If reasoning is OFF, return immediately after first call
            if not use_reasoning:
                yield 'answer', response_content.strip(), response_content
                return

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_content}]
            })

            # Check for final answer tag
            match_answer = re.search(r'<answer>((.|\n)*?)</answer>', response_content)
            if match_answer:
                answer_text = match_answer.group(1).strip()
                yield 'answer', answer_text, match_answer.group(0)
                break

            # Handle bbox tag only if reasoning is enabled
            match_bbox = re.search(r'<bbox>((.|\n)*?)</bbox>', response_content)
            if match_bbox:
                yield from self._handle_bbox(match_bbox.group(1).strip(), messages)
                max_steps -= 1
                continue

            max_steps -= 1
        else:
            # max steps reached
            yield 'answer', 'Max reasoning steps reached.', ''

    # ---------- unchanged bbox handler ----------
    def _handle_bbox(self, content, messages):
        try:
            bbox = json.loads(content)
            input_img = self.image_input[-1]
            raw_img = self.image_raw[-1]
            input_w, input_h = input_img.size
            raw_w, raw_h = raw_img.size

            x0 = bbox[0] * raw_w / input_w
            y0 = bbox[1] * raw_h / input_h
            x1 = bbox[2] * raw_w / input_w
            y1 = bbox[3] * raw_h / input_h
            pad = 56

            crop_box = [max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, raw_w), min(y1 + pad, raw_h)]
            cropped = raw_img.crop(crop_box)

            image_input, base64_img = self.process_image(cropped)
            self.image_raw.append(cropped)
            self.image_input.append(image_input)

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the cropped region of the image."},
                    {"type": "image_url", "image_url": {"url": base64_img}}
                ]
            })

            if self.generator:
                draw_img = self.image_input[-2].copy()
                draw = ImageDraw.Draw(draw_img)
                draw.rectangle(bbox, outline=(255, 0, 0), width=5)
                yield 'crop_image', image_input, draw_img
        except Exception:
            pass