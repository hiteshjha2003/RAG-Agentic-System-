import streamlit as st
import base64
import json
import math
import re
from io import BytesIO
from PIL import Image, ImageDraw
from time import sleep
from openai import OpenAI

# === Configuration ===
API_KEY = 'fxxxxxxxx'
BASE_URL = 'https://api.sambanova.ai/v1'
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct"

# === Helper Functions ===
def encode_image(image: Image.Image) -> str:
    if image.mode == "RGBA":
        # Remove alpha channel by converting to RGB (white background)
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


# === VRAG Agent Class ===
class VRAG:
    def __init__(self, max_steps=10):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.max_steps = max_steps
        self.image_raw = []
        self.image_input = []

    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")

        area = image.width * image.height
        if area > self.max_pixels or area < self.min_pixels:
            target = self.max_pixels if area > self.max_pixels else self.min_pixels
            factor = math.sqrt(target / area)
            image = image.resize((int(image.width * factor), int(image.height * factor)))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image, encode_image(image)

    def run(self, messages):
        self.image_raw.clear()
        self.image_input.clear()
        steps_remaining = self.max_steps

        # Load initial image if exists in messages
        if messages and isinstance(messages[0].get("content"), list):
            for part in messages[0]["content"]:
                if part.get("type") == "image_url":
                    try:
                        img_data = base64.b64decode(part["image_url"]["url"].split(",")[-1])
                        img = Image.open(BytesIO(img_data))
                        self.image_raw.append(img)
                        self.image_input.append(img)
                    except Exception as e:
                        yield 'answer', f"⚠️ Error decoding image: {e}", ''

        while steps_remaining > 0:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False
            )
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

            # Yield think messages for intermediate thoughts
            if match := re.search(r'<think>(.*?)</think>', content, re.DOTALL):
                yield 'think', match.group(1).strip(), match.group(0)

            # Yield final answer and stop
            if match := re.search(r'<answer>((.|\n)*?)</answer>', content):
                yield 'answer', match.group(1).strip(), match.group(0)
                return

            # Handle bounding boxes for cropping
            if match := re.search(r'<bbox>((.|\n)*?)</bbox>', content):
                yield from self._handle_bbox(match.group(1).strip(), messages)

            steps_remaining -= 1

        yield 'answer', 'Max reasoning steps reached.', ''

    def _handle_bbox(self, bbox_str, messages):
        try:
            bbox = json.loads(bbox_str)
            input_img = self.image_input[-1]
            raw_img = self.image_raw[-1]
            iw, ih, rw, rh = *input_img.size, *raw_img.size

            # Convert bbox coords from input_img scale to raw_img scale
            x0, y0, x1, y1 = [bbox[i] * rw / iw if i % 2 == 0 else bbox[i] * rh / ih for i in range(4)]
            pad = 56
            crop = [max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, rw), min(y1 + pad, rh)]
            cropped = raw_img.crop(crop)

            img_processed, encoded = self.process_image(cropped)
            self.image_raw.append(cropped)
            self.image_input.append(img_processed)

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the cropped region of the image."},
                    {"type": "image_url", "image_url": {"url": encoded}}
                ]
            })

            # Draw bbox on previous input image for visualization
            bbox_scaled = [
                bbox[0], bbox[1], bbox[2], bbox[3]
            ]
            img_with_bbox = self.image_input[-2].copy()
            draw = ImageDraw.Draw(img_with_bbox)
            draw.rectangle(bbox_scaled, outline=(255, 0, 0), width=5)

            yield 'crop_image', img_processed, img_with_bbox

        except Exception as e:
            yield 'answer', f"❌ Error handling bbox: {e}", ''

# === Streamlit App ===
def main():
    st.set_page_config(page_title="VRAG: Discovering More in Depth", page_icon="🔍", layout="wide")
    agent = VRAG()

    st.markdown("""
    <style>
    .info-box { font-size: 18px; background: #E3F2FD; padding: 10px; border-left: 5px solid #3498DB; margin: 10px 0; }
    .caption { font-weight: bold; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 VRAG: Discovering More in Depth")

    with st.sidebar:
        agent.max_steps = st.number_input("Max Reasoning Steps:", 3, 10, 10)

    col1, col2 = st.columns(2)

    with col1:
        question = st.text_input("Your Question:")
        submit = st.button("Submit Question")
        reasoning_box = st.container()
        answer_box = st.container()

    with col2:
        uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        image_prompt = st.text_area("Describe what to analyze:")
        analyze_btn = st.button("Analyze Image")
        image_display = st.container()
        result_display = st.container()

    def run_agent(messages, display_container, image_container=None):
        try:
            generator = agent.run(messages)
            think_box = display_container.empty()
            while True:
                action, content, raw = next(generator)
                if action == 'think':
                    think_box.markdown(f"💭 {content}")
                elif action == 'crop_image' and image_container is not None:
                    with image_container:
                        col_a, col_b = st.columns(2)
                        col_a.image(raw, caption="With Bounding Box", use_container_width=True)
                        col_b.image(content, caption="Cropped Region", use_container_width=True)
                elif action == 'answer':
                    display_container.success(f"✅ {content}")
                    break
        except StopIteration:
            pass
        except Exception as e:
            display_container.error(f"❌ Error: {str(e)}")

    # Submit question + optional image
    if submit and question:
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        if uploaded_img:
            img = Image.open(uploaded_img)
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": encode_image(img)}})
        run_agent(messages, reasoning_box, image_display)

    # Analyze image with prompt
    if analyze_btn and uploaded_img and image_prompt:
        img = Image.open(uploaded_img)
        prompt = f"Analyze the following image based on this description: {image_prompt}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": encode_image(img)}}
            ]
        }]
        image_display.image(img, caption="Uploaded Image", use_container_width=True)
        run_agent(messages, result_display, image_display)


if __name__ == "__main__":
    main()
