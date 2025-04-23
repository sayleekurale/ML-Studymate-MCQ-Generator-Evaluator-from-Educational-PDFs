from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
import pandas as pd
import datetime
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os

app = Flask(__name__)
Bootstrap(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Ensure performance log exists
PERFORMANCE_LOG_FILE = "performance_log.csv"
if not os.path.exists(PERFORMANCE_LOG_FILE):
    with open(PERFORMANCE_LOG_FILE, "w") as f:
        f.write("timestamp,qnum,user,correct,is_correct\n")

def generate_mcqs(text, num_questions=5):
    if text is None:
        return []

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    num_questions = min(num_questions, len(sentences))
    selected_sentences = random.sample(sentences, num_questions)
    mcqs = []

    for sentence in selected_sentences:
        sent_doc = nlp(sentence)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
        if len(nouns) < 2:
            continue
        noun_counts = Counter(nouns)
        if noun_counts:
            subject = noun_counts.most_common(1)[0][0]
            question_stem = sentence.replace(subject, "______")
            answer_choices = [subject]
            distractors = list(set(nouns) - {subject})
            while len(distractors) < 3:
                distractors.append("[Distractor]")
            random.shuffle(distractors)
            for distractor in distractors[:3]:
                answer_choices.append(distractor)
            random.shuffle(answer_choices)
            correct_answer = chr(65 + answer_choices.index(subject))
            mcqs.append((question_stem, answer_choices, correct_answer))
    return mcqs

def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        text += page_text
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += file.read().decode('utf-8')
        else:
            text = request.form['text']

        num_questions = int(request.form['num_questions'])
        mcqs = generate_mcqs(text, num_questions=num_questions)
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
        return render_template('mcqs.html', mcqs=mcqs_with_index)
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    user_answers = []
    correct_answers = []
    performance_data = []
    timestamp = datetime.datetime.now().isoformat()

    for key in request.form:
        if key.startswith('answer'):
            q_num = key.replace('answer', '')
            user_answer = request.form[key]
            correct_answer = request.form.get(f'correct{q_num}')
            user_answers.append(user_answer)
            correct_answers.append(correct_answer)
            performance_data.append({
                "timestamp": timestamp,
                "qnum": int(q_num),
                "user": user_answer,
                "correct": correct_answer,
                "is_correct": user_answer == correct_answer
            })

    # Save to CSV
    df = pd.DataFrame(performance_data)
    df.to_csv(PERFORMANCE_LOG_FILE, mode='a', header=False, index=False)

    accuracy = accuracy_score(correct_answers, user_answers)
    report = classification_report(correct_answers, user_answers, output_dict=True, zero_division=0)


    return render_template('results.html',
                           accuracy=accuracy,
                           report=report,
                           user_answers=user_answers,
                           correct_answers=correct_answers)
                           
@app.route('/dashboard')
def dashboard():
    # Read the CSV or log file containing the results
    try:
        df = pd.read_csv(PERFORMANCE_LOG_FILE)
    except Exception as e:
        return f"Error loading data: {e}"

    # Check if necessary columns exist in the dataframe
    if 'is_correct' not in df.columns:
        return "Error: 'is_correct' column not found in the data."
    
    # Calculate accuracy (percent of correct answers)
    accuracy = df['is_correct'].mean() * 100
    
    # Extract current attempted questions and results
    attempted_mcqs = df[df['is_correct'].notna()]  # Filter out rows where 'is_correct' is NaN
    
    # Check if 'topic' column exists, if not, handle it gracefully
    if 'topic' in df.columns:
        weak_topics = attempted_mcqs[attempted_mcqs['is_correct'] == 0]['topic'].value_counts().head(3).index.tolist()
        strong_topics = attempted_mcqs[attempted_mcqs['is_correct'] == 1]['topic'].value_counts().head(3).index.tolist()
    else:
        weak_topics = []  # Handle missing 'topic' column
        strong_topics = []
    
    # Calculate total questions attempted
    total_attempted = len(attempted_mcqs)
    
    # Render dashboard with the calculated values
    return render_template(
        'dashboard.html',
        accuracy=accuracy,
        total_attempted=total_attempted,
        weak_topics=weak_topics,
        strong_topics=strong_topics
    )


if __name__ == '__main__':
    app.run(debug=True)
