# ML-Studymate-MCQ-Generator-Evaluator-from-Educational-PDFs

ML-StudyMate is an AI-driven platform that automates the generation of Multiple Choice Questions (MCQs) and provides real-time evaluation, feedback, and personalized learning strategies. It empowers students by transforming static PDF content into interactive, adaptive learning experiences using machine learning and natural language processing.

---

## 🚀 Features

### 🧠 AI-Powered MCQ Generation
- Extracts key concepts from uploaded PDFs using **spaCy** and **TextRank**.
- Fine-tuned **T5 model** generates high-quality MCQs with contextually relevant answer choices.
- Distractors are generated using **Word2Vec/BERT**-based semantic similarity.

### 🔁 Preprocessing Pipeline (SciQ Dataset)
- Dataset preprocessing: tokenization, cleaning, POS tagging, answer highlighting.
- Splitting into training, validation, and test sets.
- Fine-tuning done on the **SciQ dataset** for question generation and difficulty classification.

### 🎯 Adaptive Testing
- Dynamically adjusts question difficulty based on user performance.
- Uses **KMeans clustering** and **Decision Trees** to classify user skill levels and adapt test paths accordingly.

### 📊 ML-Based Evaluation & Feedback
- Classifies correctness using **Logistic Regression / SVM** models.
- Tags questions with difficulty levels and confidence scores.
- Offers real-time feedback on answers with explanations.

### 🔍 Explainability & Justification
- Integrates **LIME/SHAP** for Explainable AI.
- Justifies answer correctness and difficulty predictions.
- Enhances trust and transparency in evaluation.

### 📚 Personalized Learning Engine
- Clusters user performance data using **unsupervised learning (KMeans/DBSCAN)**.
- Recommends:
  - Related study material
  - Weak topic-specific MCQs
  - Improvement strategies based on analytics

### ⚖️ Bias Detection
- Ensures fairness and diversity in generated MCQs.
- Applies topic diversity checks and gender/cultural bias detection using NLP metrics.

### 🕹️ Gamified Learning
- Includes:
  - Leaderboards
  - Timed quizzes
  - Challenge streaks
  - ML-driven progression paths

---

## 🧪 ML & NLP Workflow

```mermaid
graph TD
    A[PDF Upload] --> B[Text Extraction]
    B --> C[Preprocessing]
    C --> D[MCQ Generation (T5)]
    D --> E[Distractor Generation (BERT)]
    E --> F[Store in MongoDB]

    F --> G[User Takes Quiz]
    G --> H[ML-Based Evaluation (SVM/LogReg)]
    H --> I[User Score Analysis]
    I --> J[Adaptive Difficulty (Clustering)]
    J --> K[Personalized Recommendations]
```

---

## 🛠️ Tech Stack

| Component        | Tech Used                               |
|------------------|------------------------------------------|
| Frontend         | Tkinter GUI                              |
| Backend          | Flask / FastAPI                          |
| NLP & ML         | spaCy, T5, Word2Vec, BERT, SVM, KMeans   |
| Dataset          | SciQ (train/val/test CSVs)               |
| Database         | MongoDB                                  |
| Explainability   | LIME / SHAP                              |

---

## 📁 Project Structure

```
ML-StudyMate/
├── backend/
│   ├── app.py
│   ├── mcq_generator.py
│   ├── evaluator.py
│   └── recommender.py
├── data/
│   └── sciq/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   └── t5_qg_model/
├── frontend/
│   └── gui.py
├── utils/
│   └── preprocess.py
│   └── cluster.py
├── README.md
└── requirements.txt
```

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/ML-StudyMate.git
cd ML-StudyMate

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Model Training

1. Preprocess the SciQ dataset:
   ```bash
   python utils/preprocess.py
   ```

2. Fine-tune T5 for MCQ generation:
   ```bash
   python backend/train_t5.py
   ```

3. Train evaluation/classification models:
   ```bash
   python backend/train_classifier.py
   ```

---

## 📊 Run the App

```bash
# Start the backend server
python backend/app.py

# Run the GUI
python frontend/gui.py
```

---

## 📚 Dataset Used

- **SciQ Dataset**: A crowd-sourced dataset of science questions with correct answers and distractors.
  - [SciQ on AllenAI](https://allenai.org/data/sciq)

---

## ✅ To-Do & Enhancements

- [ ] Integrate quiz timer and challenge mode
- [ ] Add export to PDF for quiz results
- [ ] Create user login system for session tracking
- [ ] Extend recommendation engine with GPT fine-tuning

---

## 🧑‍💻 Contributors

- Saylee Kurale – ML Lead, Backend Developer  
- Kanika Gupta - Frontend 

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📬 Contact

Saylee Kurale  
kuralesaylee07@gmail.com 
MIT-WPU, Computer Science  
```
