import streamlit as st
from vrag_agent import VRAG
from PIL import Image
from time import sleep
from io import BytesIO
import base64
import os

# ---------- page config ----------
st.set_page_config(
    page_title="VRAG: Discovering More in Depth",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- helpers ----------
def typewriter_effect(container, text, delay=0.02):
    final_text = ''
    for char in text:
        final_text += char
        display_text = final_text.replace("<", "&lt;").replace(">", "&gt;")
        container.markdown(f'<div class="info-box">{display_text}</div>', unsafe_allow_html=True)
        sleep(delay)
    container.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def encode_image(image: Image.Image) -> str:
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

# ---------- main ----------
def main():
    st.title("🔍 VRAG: Discovering More in Depth")

    st.markdown("""
    <style>
        .sidebar-header { font-size: 18px; color: #1ABC9C; margin-bottom: 15px; }
        .header-text { font-size: 20px; color: #34495E; margin-bottom: 10px; }
        .success-message {
            background-color: #D5F5E3; color: #1E8449; padding: 10px;
            border-radius: 5px; margin-top: 10px;
        }
        .info-box {
            font-size: 18px; background-color: #E3F2FD;
            padding: 15px; border-left: 6px solid #2E86C1;
            margin-bottom: 10px; color: #1E2A38;
        }
        .image-container img {
            display: block; margin-left: auto; margin-right: auto;
            max-height: 400px;
        }
        .caption {
            font-size: 16px; color: black; text-align: left;
            font-weight: bold; margin-left: 10px;
            margin-top: 10px; margin-bottom: -5px;
        }
    </style>
    """, unsafe_allow_html=True)

    agent = VRAG()

    # ---------- sidebar ----------
    with st.sidebar:
        st.markdown('<p class="sidebar-header">⚙️ Configuration Options</p>', unsafe_allow_html=True)
        MAX_ROUNDS = st.number_input('Number of Max Reasoning Iterations:', min_value=3, max_value=10, value=10)
        use_reasoning = st.checkbox(
            "🔍 Enable visual reasoning (cropping & re-asking)",
            value=False,
            help="Turn OFF for a single-shot, plain answer without extra visual reasoning."
        )
        st.markdown('<p class="sidebar-header">📚 Example Questions</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    # ---------- left column ----------
    with col_left:
        st.markdown('<p class="header-text">📝 Question and Reasoning</p>', unsafe_allow_html=True)
        question = st.text_input(
            "Question Input:",
            placeholder="Type your question here...",
            key="question_input",
            help="Enter your question for the VRAG agent to process",
            max_chars=500,
            label_visibility="collapsed"
        )
        submit_button = st.button("Submit Question", key="submit_question")
        reasoning_container = st.container()
        answer_container = st.container()

    # ---------- right column ----------
    with col_right:
        st.markdown('<p class="header-text">🖼️ Image Analysis</p>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"], key="image_uploader")
        image_description = st.text_area(
            "Image Description:",
            placeholder="Provide a description or specific query about the image...",
            key="image_description",
            height=100
        )
        analyze_image_button = st.button("Analyze Image", key="analyze_image")
        image_container = st.container()
        analysis_container = st.container()

    # ---------- question + optional image ----------
    if submit_button and question:
        with reasoning_container:
            st.markdown('<p class="success-message">Question submitted! Processing…</p>', unsafe_allow_html=True)
            agent.max_steps = MAX_ROUNDS

            # build message
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_url = encode_image(image)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }]
            else:
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                }]

            try:
                generator = agent.run(messages, use_reasoning=use_reasoning)
                think_placeholder = reasoning_container.empty()
                for action, content, raw_content in generator:
                    if action == 'think' and use_reasoning:
                        think_placeholder.markdown(f"💭 Thinking: {content}")
                    elif action == 'bbox' and use_reasoning:
                        typewriter_effect(reasoning_container, f"📷 <strong>Region of Interest:</strong> {content}")
                    elif action == 'crop_image' and use_reasoning:
                        with image_container:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown('<p class="caption">🤔 Image with Bounding Box</p>', unsafe_allow_html=True)
                                st.image(raw_content, use_container_width=True)
                            with col2:
                                st.markdown('<p class="caption">✂️ Cropped Region</p>', unsafe_allow_html=True)
                                st.image(content, use_container_width=True)
                    elif action == 'answer':
                        answer_container.success(f"✅ Answer: {content}")
                        break
            except Exception as e:
                answer_container.error(f"❌ Runtime error: {str(e)}")

    # ---------- standalone image analysis ----------
    if analyze_image_button and uploaded_image and image_description:
        with analysis_container:
            st.markdown('<p class="success-message">Analyzing image based on description…</p>', unsafe_allow_html=True)
            image = Image.open(uploaded_image)
            image_url = encode_image(image)
            image_analysis_prompt = f"Analyze the following image based on this description: {image_description}"

            try:
                generator = agent.run(
                    [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": image_analysis_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }],
                    use_reasoning=use_reasoning
                )

                think_placeholder = analysis_container.empty()
                with image_container:
                    st.markdown('<p class="caption">📷 Uploaded Image</p>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)

                while True:
                    action, content, raw_content = next(generator)
                    if action == 'think' and use_reasoning:
                        think_placeholder.markdown(f"💭 Analyzing: {content}")
                        sleep(0.5)
                    elif action == 'bbox' and use_reasoning:
                        typewriter_effect(analysis_container, f"📷 <strong>Region of Interest:</strong> {content}")
                    elif action == 'crop_image' and use_reasoning:
                        with image_container:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown('<p class="caption">🤔 Image with Bounding Box</p>', unsafe_allow_html=True)
                                st.image(raw_content, use_container_width=True)
                            with col2:
                                st.markdown('<p class="caption">✂️ Cropped Region</p>', unsafe_allow_html=True)
                                st.image(content, use_container_width=True)
                    elif action == 'answer':
                        analysis_container.success(f"✅ Image Analysis Result: {content}")
                        break

            except StopIteration as e:
                try:
                    action, content, _ = e.value
                    if action == 'answer':
                        analysis_container.success(f"✅ Image Analysis Result: {content}")
                except Exception:
                    analysis_container.error("❌ An error occurred while finalizing the image analysis.")
            except Exception as e:
                analysis_container.error(f"❌ Runtime error: {str(e)}")

if __name__ == "__main__":
    main()