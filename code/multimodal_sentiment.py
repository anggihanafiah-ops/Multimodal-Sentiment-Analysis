!pip install -q transformers accelerate bitsandbytes opencv-python pillow
pip install bitsandbytes accelerate qwen-vl-utils

## Connect google drive
from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive/Multimodal/"
VIDEO_PATH = BASE_PATH + "dataset_video/"
LABEL_PATH = BASE_PATH + "labels/"

## Huggingface Login
from huggingface_hub import login
login("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")

## Load Model
import torch
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)

# =======================
# MODEL NAME
# =======================
ONEVISION_MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# =======================
# 4BIT CONFIG (HEMAT VRAM)
# =======================
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# =======================
# MAIN FUNCTION LOADER
# =======================
def load_model(model_type="onevision"):
    if model_type == "onevision":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            ONEVISION_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(ONEVISION_MODEL)
        print("Loaded: LLaVA-OneVision (4bit)")

    elif model_type == "qwen":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(QWEN_MODEL)
        print("Loaded: Qwen2.5-VL (4bit)")

    else:
        raise ValueError("model_type harus 'onevision' atau 'qwen'")

    return model, processor

## Predict text only
def predict_text_dataset(model, processor, df, model_name="model"):

    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    results = []

    print("\n================ TEXT ONLY MODE ================\n")

    for i, row in df.iterrows():
        transcript = row["transcript"]
        emotion = row["emotion"]
        label = row["label"]

        try:
            # =========================
            # 🔥 BASE PROMPT + EMOTION
            # =========================
            base_prompt = f"""
You are an AI that classifies sentiment from short dialogue.

Consider:
- the meaning of the dialogue
- the implied emotion
- the given emotion label

Dialogue:
"{transcript}"

Emotion:
{emotion}

Answer:
"""

            # =========================
            # HYPOTHESIS
            # =========================
            prompt_pos = base_prompt + "Positive"
            prompt_neg = base_prompt + "Negative"

            inputs_pos = processor(text=prompt_pos, return_tensors="pt").to(model.device)
            inputs_neg = processor(text=prompt_neg, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_pos = model(**inputs_pos).logits
                out_neg = model(**inputs_neg).logits

            score_pos = out_pos.mean().item()
            score_neg = out_neg.mean().item()

            pred = "Positive" if score_pos > score_neg else "Negative"

        except Exception as e:
            print("ERROR:", e)
            pred = None

        print(f"[{i+1}/{len(df)}] {row['video_id']} -> {pred} | emotion: {emotion}")

        results.append({
            "video_id": row["video_id"],
            "transcript": transcript,
            "emotion": emotion,
            "label": label,
            "predicted": pred
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.dropna()

    if len(results_df) == 0:
        print("Semua prediksi gagal")
        return results_df

    # =========================
    # SAVE
    # =========================
    results_df.to_csv(f"hasil_text_{model_name}.csv", index=False)

    # =========================
    # EVALUASI
    # =========================
    labels = ["Negative", "Positive"]

    accuracy = accuracy_score(results_df["label"], results_df["predicted"])
    print(f"\nAkurasi: {accuracy*100:.2f}%")

    print("\nClassification Report:\n")
    print(classification_report(
        results_df["label"],
        results_df["predicted"],
        labels=labels,
        target_names=labels,
        zero_division=0
    ))

    cm = confusion_matrix(
        results_df["label"],
        results_df["predicted"],
        labels=labels
    )

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Text Only")
    plt.show()

    return results_df

---

## Video Only
def predict_video(model, processor, df):

    import pandas as pd
    import numpy as np
    import torch
    import av
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    def read_video_pyav(container, indices):
        frames = []
        container.seek(0)

        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frame)

        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def preprocess_video(row):
        eps = row["video_id"].split("_")[1]
        vid = row["video_id"].split("_")[0]
        video_path = VIDEO_PATH + f"{eps}/{vid}.mp4"

        container = av.open(video_path)
        total_frames = container.streams.video[0].frames

        indices = np.linspace(0, total_frames-1, 30).astype(int)

        video_tensor = read_video_pyav(container, indices)
        container.close()

        return video_tensor

    df = df.copy()
    print("\n=== EXTRACT VIDEO ===\n")
    df["video_tensor"] = df.apply(preprocess_video, axis=1)

    results = []
    print("\n=== VIDEO ONLY ===\n")

    for i, row in df.iterrows():

        label = row["label"]
        video_tensor = row["video_tensor"]

        try:
            conversation = [
                {
                    "role": "system",
                    "content": (
                        "You analyze sentiment based on visual cues from a video.\n"
                        "Focus on expressions, interaction, and scene atmosphere.\n\n"
                        "Do not assume negative unless there is clear conflict.\n"
                        "If the scene looks normal or unclear, choose Positive.\n\n"
                        "Answer only: Positive or Negative."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": "Determine the sentiment of this scene."},
                    ],
                },
            ]

            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            inputs = processor(
                text=prompt,
                videos=video_tensor,
                return_tensors="pt"
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )

            result = processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()

            if "negative" in result:
                pred = "Negative"
            elif "positive" in result:
                pred = "Positive"
            else:
                pred = "Positive"

        except Exception as e:
            print("ERROR:", row["video_id"], e)
            pred = "Positive"

        print(f"[{i+1}/{len(df)}] {row['video_id']} -> {pred}")

        results.append({
            "video_id": row["video_id"],
            "label": label,
            "predicted": pred
        })

    # =========================
    # DATAFRAME & CLEANING
    # =========================
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=["label", "predicted"])

    results_df["label"] = results_df["label"].astype(str)
    results_df["predicted"] = results_df["predicted"].astype(str)

    # =========================
    # DEBUG INFO
    # =========================
    print("\nPreview:")
    print(results_df.head())

    print("\nDistribusi Label:")
    print(results_df["label"].value_counts())

    print("\nDistribusi Prediksi:")
    print(results_df["predicted"].value_counts())

    # =========================
    # EVALUASI
    # =========================
    labels = ["Negative", "Positive"]

    acc = accuracy_score(results_df["label"], results_df["predicted"])
    print(f"\nAkurasi: {acc*100:.2f}%")

    print("\nClassification Report:\n")
    print(classification_report(
        results_df["label"],
        results_df["predicted"],
        labels=labels,
        target_names=labels,
        zero_division=0
    ))

    cm = confusion_matrix(
        results_df["label"],
        results_df["predicted"],
        labels=labels
    )

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Video Only")
    plt.show()

    return results_df

---
## Multimodal
def predict_multimodal(model, processor, df):

    import pandas as pd
    import numpy as np
    import torch
    import av
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # =========================
    # READ VIDEO
    # =========================
    def read_video_pyav(container, indices):
        frames = []
        container.seek(0)

        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frame)

        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    # =========================
    # PREPROCESS VIDEO
    # =========================
    def preprocess_video(row):
        try:
            eps = row["video_id"].split("_")[1]
            vid = row["video_id"].split("_")[0]

            video_path = VIDEO_PATH + f"{eps}/{vid}.mp4"

            container = av.open(video_path)
            total_frames = container.streams.video[0].frames

            indices = np.linspace(0, total_frames-1, 30).astype(int)

            video_tensor = read_video_pyav(container, indices)
            container.close()

            return video_tensor

        except Exception as e:
            print("VIDEO ERROR:", row["video_id"], e)
            return None

    # =========================
    # EXTRACT VIDEO
    # =========================
    print("\n=== EXTRACT VIDEO ===\n")
    df = df.copy()
    df["video_tensor"] = df.apply(preprocess_video, axis=1)

    # 🔥 buang video yang gagal
    df = df.dropna(subset=["video_tensor"])

    print("Jumlah video valid:", len(df))

    results = []

    print("\n=== MULTIMODAL (FIXED) ===\n")

    for i, row in df.iterrows():

        transcript = row["transcript"]
        emotion = row["emotion"]
        label = row["label"]
        video_tensor = row["video_tensor"]

        # 🔥 default biar ga pernah None
        pred = "Positive"

        try:
            conversation = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that determines sentiment based on video, dialogue, and emotion.\n"
                        "Consider facial expressions, interaction, and dialogue meaning.\n\n"
                        "Sentiment:\n"
                        "- Positive → calm, friendly, or pleasant situation\n"
                        "- Negative → conflict, tension, or negative emotion\n\n"
                        "Use emotion as additional context.\n"
                        "Answer only: Positive or Negative."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'Dialogue: "{transcript}"\nEmotion: {emotion}'
                        },
                        {"type": "video"},
                    ],
                },
            ]

            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            inputs = processor(
                text=prompt,
                videos=video_tensor,
                return_tensors="pt"
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=5)

            result = processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()

            # =========================
            # PARSING AMAN
            # =========================
            if "negative" in result:
                pred = "Negative"
            elif "positive" in result:
                pred = "Positive"
            else:
                # fallback pakai emotion
                if emotion in ["anger", "sadness"]:
                    pred = "Negative"
                else:
                    pred = "Positive"

        except Exception as e:
            print("ERROR:", row["video_id"], e)
            pred = "Positive"

        # 🔥 DEBUG kalau aneh
        if pred not in ["Positive", "Negative"]:
            print("ANOMALI:", row["video_id"], "->", pred)
            pred = "Positive"

        print(f"[{i+1}/{len(df)}] {row['video_id']} -> {pred} | emotion: {emotion}")

        results.append({
            "video_id": row["video_id"],
            "transcript": transcript,
            "emotion": emotion,
            "label": label,
            "predicted": pred
        })

    # =========================
    # DATAFRAME
    # =========================
    results_df = pd.DataFrame(results)

    # 🔥 FIX ERROR UTAMA
    results_df = results_df.dropna(subset=["label", "predicted"])

    results_df["label"] = results_df["label"].astype(str)
    results_df["predicted"] = results_df["predicted"].astype(str)

    print("\nPreview:")
    print(results_df.head())

    print("\nDistribusi Label:")
    print(results_df["label"].value_counts())

    print("\nDistribusi Prediksi:")
    print(results_df["predicted"].value_counts())

    # =========================
    # EVALUASI
    # =========================
    labels = ["Negative", "Positive"]

    acc = accuracy_score(results_df["label"], results_df["predicted"])
    print(f"\nAkurasi: {acc*100:.2f}%")

    print("\nClassification Report:\n")
    print(classification_report(
        results_df["label"],
        results_df["predicted"],
        labels=labels,
        target_names=labels,
        zero_division=0
    ))

    cm = confusion_matrix(
        results_df["label"],
        results_df["predicted"],
        labels=labels
    )

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Multimodal")
    plt.show()

    return results_df

---

## Read Data

def Read_data(path, addVideoPath=True):

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    print("\n===========================================================\n")
    print("MEMBACA DATA:", path)
    print("\n===========================================================\n")

    # =========================
    # LOAD DATA
    # =========================
    data = pd.read_csv(path)

    print("\nDATA AWAL")
    print("\n===========================================================\n")
    print(data.info())
    print(data.head())

    # =========================
    # VALIDASI KOLOM
    # =========================
    if "label" not in data.columns:
        raise ValueError("Kolom harus bernama 'label'")

    if "emotion" not in data.columns:
        print("\n[WARNING] Kolom 'emotion' tidak ditemukan")

    # =========================
    # HANDLE NULL
    # =========================
    print("\nJumlah data kosong:\n", data.isnull().sum())
    data = data.dropna()

    # =========================
    # NORMALISASI LABEL
    # =========================
    data['label'] = (
        data['label']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    mapping = {
        'negative': 'Negative',
        'positive': 'Positive',
        '0': 'Negative',
        '1': 'Positive'
    }

    data['label'] = data['label'].map(mapping)

    # =========================
    # 🔥 NORMALISASI EMOTION
    # =========================
    if "emotion" in data.columns:

        data['emotion'] = (
            data['emotion']
            .astype(str)
            .str.lower()
            .str.strip()
        )

        emo_map = {
            'joy': 'joy',
            'happy': 'joy',

            'sad': 'sadness',
            'sadness': 'sadness',

            'angry': 'anger',
            'anger': 'anger',

            'neutral': 'neutral'
        }

        data['emotion'] = data['emotion'].map(emo_map)

        print("\nDistribusi Emotion:")
        print("\n===========================================================\n")
        print(data['emotion'].value_counts())

    # =========================
    # TAMBAH PATH VIDEO
    # =========================
    if addVideoPath:
        filename = os.path.basename(path)
        eps = filename.split("_")[0]

        data['path'] = data['video_id'].apply(
            lambda x: f"{eps}/{x}.mp4"
        )

    dataclean = data

    # =========================
    # INFO HASIL
    # =========================
    print("\n===========================================================\n")
    print("DATA SETELAH CLEANING")
    print("\n===========================================================\n")
    print(dataclean.info())

    label_counts = dataclean['label'].value_counts()

    print("\nDistribusi Label:")
    print("\n===========================================================\n")
    print(label_counts)

    # =========================
    # VISUALISASI LABEL
    # =========================
    plt.figure(figsize=(5, 4))
    sns.countplot(x='label', data=dataclean)
    plt.title('Distribusi Label')
    plt.show()

    # =========================
    # VISUALISASI EMOTION
    # =========================
    if "emotion" in dataclean.columns:
        plt.figure(figsize=(5, 4))
        sns.countplot(x='emotion', data=dataclean)
        plt.title('Distribusi Emotion')
        plt.show()

    print("\n===========================================================\n")
    print("DATA FINAL")
    print("\n===========================================================\n")
    print(dataclean.head())

    return dataclean

---

## Load model Llava

model_llava, processor_llava = load_model("onevision")

## Load model Qwen

model_qwen, processor_qwen = load_model("qwen")


## Eps1

df = Read_data(LABEL_PATH + "eps1_transcripts.csv")


### text only
#### Qwen
predict_text_dataset(model_qwen, processor_qwen, df.head(50))

#### Llava
predict_text_dataset(model_llava, processor_llava, df.head(50))

### Video Only
#### Qwen
predict_video(model_qwen, processor_qwen, df.head(50))

#### Llava
predict_video(model_llava, processor_llava, df.head(50))

#### Qwen
predict_video(model_qwen, processor_qwen, df.head(50))

### Multimodal
#### Llava
predict_multimodal(model_llava, processor_llava, df.head(50))

#### Qwen
predict_multimodal(model_qwen, processor_qwen, df.head(50))



## Eps2

df2 = Read_data(LABEL_PATH + "eps2_transcripts.csv")


### text only
#### Qwen
predict_text_dataset(model_qwen, processor_qwen, df2.head(50))

#### Llava
predict_text_dataset(model_llava, processor_llava, df2.head(50))

### Video Only
#### Qwen
predict_video(model_qwen, processor_qwen, df2.head(50))

#### Llava
predict_video(model_llava, processor_llava, df2.head(50))

#### Qwen
predict_video(model_qwen, processor_qwen, df2.head(50))

### Multimodal
#### Llava
predict_multimodal(model_llava, processor_llava, df2.head(50))

#### Qwen
predict_multimodal(model_qwen, processor_qwen, df2.head(50))



## Eps3

df3 = Read_data(LABEL_PATH + "eps3_transcripts.csv")


### text only
#### Qwen
predict_text_dataset(model_qwen, processor_qwen, df3.head(50))

#### Llava
predict_text_dataset(model_llava, processor_llava, df3.head(50))

### Video Only
#### Qwen
predict_video(model_qwen, processor_qwen, df3.head(50))

#### Llava
predict_video(model_llava, processor_llava, df3.head(50))

#### Qwen
predict_video(model_qwen, processor_qwen, df3.head(50))

### Multimodal
#### Llava
predict_multimodal(model_llava, processor_llava, df3.head(50))

#### Qwen
predict_multimodal(model_qwen, processor_qwen, df3.head(50))


## Eps4

df4 = Read_data(LABEL_PATH + "eps4_transcripts.csv")


### text only
#### Qwen
predict_text_dataset(model_qwen, processor_qwen, df4.head(50))

#### Llava
predict_text_dataset(model_llava, processor_llava, df4.head(50))

### Video Only
#### Qwen
predict_video(model_qwen, processor_qwen, df4.head(50))

#### Llava
predict_video(model_llava, processor_llava, df4.head(50))

#### Qwen
predict_video(model_qwen, processor_qwen, df4.head(50))

### Multimodal
#### Llava
predict_multimodal(model_llava, processor_llava, df4.head(50))

#### Qwen
predict_multimodal(model_qwen, processor_qwen, df4.head(50))


## Eps5

df5 = Read_data(LABEL_PATH + "eps5_transcripts.csv")


### text only
#### Qwen
predict_text_dataset(model_qwen, processor_qwen, df5.head(50))

#### Llava
predict_text_dataset(model_llava, processor_llava, df5.head(50))

### Video Only
#### Qwen
predict_video(model_qwen, processor_qwen, df5.head(50))

#### Llava
predict_video(model_llava, processor_llava, df5.head(50))

#### Qwen
predict_video(model_qwen, processor_qwen, df5.head(50))

### Multimodal
#### Llava
predict_multimodal(model_llava, processor_llava, df5.head(50))

#### Qwen
predict_multimodal(model_qwen, processor_qwen, df5.head(50))


## Eps6

df6 = Read_data(LABEL_PATH + "eps6_transcripts.csv")


### text only
#### Qwen
predict_text_dataset(model_qwen, processor_qwen, df6.head(50))

#### Llava
predict_text_dataset(model_llava, processor_llava, df6.head(50))

### Video Only
#### Qwen
predict_video(model_qwen, processor_qwen, df6.head(50))

#### Llava
predict_video(model_llava, processor_llava, df6.head(50))

#### Qwen
predict_video(model_qwen, processor_qwen, df6.head(50))

### Multimodal
#### Llava
predict_multimodal(model_llava, processor_llava, df6.head(50))

#### Qwen
predict_multimodal(model_qwen, processor_qwen, df6.head(50))


## Eps7

df7 = Read_data(LABEL_PATH + "eps6_transcripts.csv")


### text only
#### Qwen
predict_text_dataset(model_qwen, processor_qwen, df7.head(50))

#### Llava
predict_text_dataset(model_llava, processor_llava, df7.head(50))

### Video Only
#### Qwen
predict_video(model_qwen, processor_qwen, df7.head(50))

#### Llava
predict_video(model_llava, processor_llava, df7.head(50))

#### Qwen
predict_video(model_qwen, processor_qwen, df7.head(50))

### Multimodal
#### Llava
predict_multimodal(model_llava, processor_llava, df7.head(50))

#### Qwen
predict_multimodal(model_qwen, processor_qwen, df7.head(50))








