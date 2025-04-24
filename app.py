import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os
from typing import List, Union


class BertPredictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise ValueError(f"Директория {model_path} не существует")
            
        required_files = ['model.safetensors', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json']
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                raise ValueError(f"Файл {file} не найден в {model_path}")

        self.config = AutoConfig.from_pretrained(model_path)
  
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,  
            trust_remote_code=False  
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=self.config,
            local_files_only=True,
            trust_remote_code=False
        )

        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, texts: Union[str, List[str]], max_length: int = 512) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        logits = outputs.logits 
        predictions = torch.softmax(logits, dim=-1)

        return predictions


def process_data(df, predictor):
    def ai_guru(row):
        if row:
            embeddings = predictor.predict(str(row))
            x = embeddings[-1][-1]
            return round(x.item())
    
    df['AI_pred'] = df['Заключение по исследованию'].apply(ai_guru)
    return df

# Интерфейс Streamlit
st.title("Обработка медицинских заключений с помощью AI")


uploaded_file = st.file_uploader("Загрузите файл Excel", type=['xlsx'])

if uploaded_file is not None:
    try:

        df = pd.read_excel(uploaded_file)
        st.success("Файл успешно загружен!")
        

        st.subheader("Предпросмотр данных")
        st.write(df.head())
        

        if st.button("Обработать данные"):
            with st.spinner("Идет обработка, пожалуйста подождите..."):
                try:
    
                    model_path = "model"
                    predictor = BertPredictor(model_path)
                    
  
                    processed_df = process_data(df, predictor)
                    
 
                    output_path = "processed_output.xlsx"
                    processed_df.to_excel(output_path, index=False)
                    
                    st.success("Обработка завершена!")
                    

                    st.subheader("Результат обработки")
                    st.write(processed_df.head())
                    

                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Скачать обработанный файл",
                            data=file,
                            file_name="processed_output.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                except Exception as e:
                    st.error(f"Произошла ошибка при обработке: {str(e)}")
    
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {str(e)}")
else:
    st.info("Пожалуйста, загрузите файл Excel для обработки")