# AI Сервис для анализа медицинских заключений для поиска в норме (по заключению  AI сервисов) патологий (по заключению врача)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

AI Сервис для анализа медицинских заключений для поиска в норме (по заключению  AI сервисов) патологий (по заключению врача) с использованием BERT-модели.

##  Возможности

- 📤 Загрузка данных в формате Excel
- 🔍 Автоматический анализ текстовых медицинских заключений
- ⌛ Real-time processing with progress indicator
- 📥 Индикатор выполнения процесса обработки
- 📊 Скачивание результата в формате Excel

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Framework**: PyTorch + Transformers
- **NLP Model**: Custom fine-tuned BERT
- **Data Processing**: Pandas

## Структура проекта

```bash
.
├── app.py                 # Streamlit-приложение
├── model/                 # Папка с BERT-модель
│   ├── model.safetensors 
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── config.json
├── requirements.txt       # Зависимости
└── README.md             #  Описание проекта




##  Установка и запуск

##Клонируйте репозиторий
```bash
pip install -r requirements.txt

##Установите зависимости
```bash
pip install -r requirements.txt

##Откройте в браузере
```bash
http://localhost:8501

##  Установка и запуск
MIT
