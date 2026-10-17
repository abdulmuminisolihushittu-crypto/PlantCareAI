

import streamlit as st
from transformers import pipeline
from PIL import Image, UnidentifiedImageError

# -----------------------------
# 1. PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 PlantCare AI")
st.subheader("AI-Based Plant Disease Detection System")

st.write(
    "Upload a picture of a plant leaf to let an AI model "
    "predict its most likely condition."
)

# -----------------------------
# 2. LOAD THE AI MODEL
# -----------------------------
MODEL_NAME = "rodynaemad/plant-disease-dinov2-small"


@st.cache_resource(show_spinner="Loading the AI model...")
def load_model():
    return pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=-1
    )


# -----------------------------
# 3. IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a plant leaf picture",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        if image.width * image.height > 20_000_000:
            st.error("This image is too large. Choose a smaller picture.")
            st.stop()

        st.image(
            image,
            caption="Your uploaded plant picture",
            use_container_width=True
        )

        if st.button("🔍 Analyse Plant", type="primary"):
            if image.width < 50 or image.height < 50:
                st.error("Please upload a clearer, larger picture.")
                st.stop()

            with st.spinner("The AI is examining your picture..."):
                classifier = load_model()
                results = classifier(image, top_k=3)

            st.success("Analysis complete!")

            # -----------------------------
            # 4. DISPLAY PREDICTIONS
            # -----------------------------
            st.header("Prediction Result")

            best = results[0]
            label = best["label"].replace("___", " — ").replace("_", " ")
            score = best["score"] * 100

            st.subheader(label)
            st.metric("Model score", f"{score:.2f}%")

            st.caption(
                "The model score is not a guarantee that the prediction "
                "is correct, especially for real-world farm photographs."
            )

            st.subheader("Other possible predictions")

            for result in results[1:]:
                readable_label = (
                    result["label"]
                    .replace("___", " — ")
                    .replace("_", " ")
                )

                st.write(
                    f"**{readable_label}** — "
                    f"{result['score'] * 100:.2f}%"
                )

            # -----------------------------
            # 5. SAFETY WARNING
            # -----------------------------
            st.warning(
                "This application is an educational screening tool. "
                "It may make mistakes and does not confirm a plant disease. "
                "The model always selects from its supported classes, "
                "so it may give a misleading result for an unsupported "
                "plant or an unclear picture. Consult an agricultural "
                "expert before treating your crops."
            )

    except (UnidentifiedImageError, OSError):
        st.error("This file is not a valid picture. Please try another image.")

    except Exception:
        st.error(
            "The analysis failed. Check your internet connection and "
            "try again. If the problem continues, check the terminal "
            "for the error message."
        )

# -----------------------------
# 6. ABOUT THE PROJECT
# -----------------------------
st.divider()

st.header("About This Project")

st.write(
    "PlantCare AI demonstrates how artificial intelligence and "
    "image classification can support agriculture. It uses a "
    "pretrained machine-learning model to classify plant leaf images."
)

st.write("**Created by:** Abdulsalam")

st.write(
    "**Limitation:** The model was evaluated on PlantVillage images. "
    "Its results may be less reliable on photographs taken in real farms."
)
