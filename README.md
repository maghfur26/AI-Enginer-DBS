# 🍳 Olah — Punya sisa bahan makanan? di Olah aja!

> **Sistem rekomendasi resep berbasis AI yang mengubah bahan sisa di dapurmu menjadi hidangan lezat.**
> Tidak ada lagi bahan terbuang sia-sia.

![Python](https://img.shields.io/badge/Python-3.11-4CAF50?style=flat-square&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Powered-FF9800?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Pipeline-8B5CF6?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-blue?style=flat-square)

---

## 📌 Tentang Proyek

### Masalah
Jutaan ton makanan terbuang setiap tahun karena orang tidak tahu cara mengolah bahan sisa yang ada di kulkas mereka. Ini bukan hanya pemborosan finansial, tapi juga berdampak besar pada lingkungan.

### Solusi
**Olah** adalah sistem AI yang memungkinkan pengguna untuk menginput bahan-bahan yang mereka miliki, lalu sistem akan merekomendasikan resep masakan yang relevan, lengkap dengan langkah-langkah memasak, waktu memasak, dan tips penyajian.

### Pendekatan AI
Menggunakan pipeline **RAG (Retrieval-Augmented Generation)** dikombinasikan dengan **Large Language Model (LLM)** untuk menghasilkan rekomendasi resep yang akurat, kreatif, dan kontekstual berdasarkan bahan yang tersedia.

---

## 🎯 Fitur Utama

- 🥦 **Input bahan fleksibel** — pengguna bisa memasukkan bahan apa saja yang tersedia
- 🔍 **Semantic search** — pencarian resep berdasarkan makna, bukan hanya keyword
- 🤖 **Generasi resep dengan LLM** — resep dihasilkan secara dinamis dan natural
- 🇮🇩 **Support masakan Indonesia & internasional**
- ⚡ **Respons cepat < 3 detik**
- ⭐ **Rating & feedback** dari pengguna untuk continuous improvement

---

## 🔄 Alur Sistem

```
Input Bahan  →  Preprocessing NLP  →  Vector Search  →  LLM Generation  →  Resep + Tips
    👤               🧹                    🔍                  🤖                 🍽️
```

1. **Input bahan** — pengguna mengetik bahan sisa yang dimiliki
2. **Preprocessing NLP** — normalisasi teks, entity extraction (nama bahan)
3. **Vector Search** — mencari resep relevan dari vector database (ChromaDB)
4. **LLM Generation** — LLM merangkai resep lengkap berdasarkan konteks bahan
5. **Output** — resep dengan bahan, langkah memasak, dan tips

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| **LLM** | OpenAI GPT-4o / Google Gemini |
| **RAG Framework** | LangChain |
| **Embeddings** | Sentence-BERT (`all-MiniLM-L6-v2`) |
| **Vector Database** | ChromaDB |
| **API Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Package Manager** | uv |
| **Experiment Tracking** | MLflow / Weights & Biases |
| **Containerization** | Docker |
| **Cloud Deployment** | Google Cloud Platform (GCP) |
| **Data Processing** | Pandas, NumPy |

---

## 📁 Struktur Folder

```
olah/
│
├── 📁 data/
│   ├── raw/                  # Dataset resep mentah (JSON/CSV)
│   ├── processed/            # Data setelah cleaning & preprocessing
│   └── embeddings/           # Vector embeddings yang sudah di-generate
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_embedding_experiment.ipynb
│   └── 04_rag_evaluation.ipynb
│
├── 📁 src/
│   ├── ingestion/            # Data loading & parsing pipeline
│   ├── preprocessing/        # Text cleaning & NLP processing
│   ├── embeddings/           # Vectorization & embedding logic
│   ├── retrieval/            # RAG retriever (ChromaDB query)
│   ├── generation/           # LLM prompt building & response parsing
│   └── utils/                # Helper functions
│
├── 📁 api/
│   ├── routes/               # FastAPI route handlers
│   ├── schemas/              # Pydantic models (request/response)
│   └── middleware/           # Auth, logging, rate limiting
│
├── 📁 prompts/
│   ├── system/               # System prompt templates
│   └── few_shot/             # Few-shot example prompts
│
├── 📁 tests/
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── evals/                # LLM evaluation & RAGAS metrics
│
├── 📁 infra/
│   ├── docker/               # Dockerfile & docker-compose
│   └── gcp/                  # Cloud Run / GKE config
│
├── 📁 docs/
│   ├── architecture.md       # System architecture diagram
│   └── experiments/          # Experiment logs & findings
│
├── .env.example              # Template environment variables
├── .gitignore
├── pyproject.toml            # Project config (dikelola uv)
├── uv.lock                   # Lock file uv
├── Makefile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Cara Menjalankan

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — install dengan `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker (opsional)
- API Key untuk LLM provider (OpenAI / Google)

### Instalasi

```bash
# 1. Clone repository
git clone https://github.com/username/olah.git
cd olah

# 2. Install dependencies & buat virtual environment otomatis
uv sync

# 3. Aktifkan virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 4. Setup environment variables
cp .env.example .env
# Edit .env dan isi API key kamu

# 5. Jalankan aplikasi
uv run uvicorn api.main:app --reload
```

> 💡 **Kenapa uv?** uv jauh lebih cepat dari pip (10-100x), otomatis mengelola virtual environment, dan menggunakan `pyproject.toml` sebagai satu sumber konfigurasi project.

### Menambah dependency baru

```bash
uv add langchain          # tambah package
uv add pytest --dev       # tambah dev dependency
uv remove langchain       # hapus package
```

### Menggunakan Docker

```bash
docker-compose up --build
```

Aplikasi akan berjalan di `http://localhost:8000`

---

## 📊 Target Metrik

| Metrik | Target |
|---|---|
| Relevance Score (RAGAS) | ≥ 85% |
| Response Time | < 3 detik |
| Jumlah resep terindeks | 50.000+ resep |
| User Rating | ≥ 4.0 / 5.0 |
| Faithfulness Score | ≥ 0.80 |

---

## 🧪 Evaluasi

Evaluasi sistem menggunakan framework **RAGAS** dengan metrik:

- **Faithfulness** — seberapa akurat resep yang dihasilkan vs konteks yang diambil
- **Answer Relevancy** — seberapa relevan resep dengan bahan yang diinput
- **Context Recall** — seberapa baik retriever menemukan resep yang tepat
- **Context Precision** — akurasi konteks yang di-retrieve

---

## 🗺️ Roadmap

- [x] Setup project structure & environment
- [x] Data collection & preprocessing pipeline
- [x] Embedding & vector store setup
- [ ] RAG pipeline implementation
- [ ] LLM integration & prompt engineering
- [ ] API development (FastAPI)
- [ ] Frontend (Streamlit)
- [ ] Evaluation & optimization
- [ ] Deployment ke GCP
- [ ] Dokumentasi final

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan **Capstone Project** dalam program AI Engineer.

---

## 👨‍💻 Author

**[Nama Kamu]**
AI Engineer — Capstone Project

> *"Turning leftover ingredients into delightful meals — one AI recommendation at a time."* 🍳

---

<div align="center">
  Made with ❤️ and a lot of 🍳 &nbsp;|&nbsp; Capstone Project &nbsp;|&nbsp; AI Engineer Track
</div>
