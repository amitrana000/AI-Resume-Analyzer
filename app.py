from flask import Flask, request, jsonify, render_template
import PyPDF2
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)
client = InferenceClient(
       api_key=os.getenv("HF_TOKEN")
)
@app.route('/analyze', methods=['POST'])

def analyze():
    try:
        file = request.files.get('resume')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        job_description = request.form.get('job_description', '')
        filename = file.filename
        ext = filename.split('.')[-1].lower()
        resume_text = ""

        if ext == "pdf":
            pdf_read = PyPDF2.PdfReader(file)
            for i in pdf_read.pages:
                text_check = i.extract_text()
                if text_check:
                    resume_text += text_check + "\n"

        elif ext == "docx":
            docx_read = Document(file)
            for para in docx_read.paragraphs:
                if para.text.strip():
                    resume_text += para.text + "\n"

        else:
            return jsonify({"error": "Unsupported file type"}), 400

        resume_clean_text = resume_text.lower().translate(str.maketrans('', '', string.punctuation))
        token_text = nltk.word_tokenize(resume_clean_text)
        words = set(stopwords.words("english"))
        final_text = [w for w in token_text if w not in words]
        clean_text = " ".join(final_text)

        prompt = f"""
You are a professional resume analyzer and career advisor.

Analyze the following resume and provide:

1. Key skills mentioned in the resume
2. Experience level (Beginner / Intermediate / Advanced)
3. Recommended job roles for this candidate
4. Give ATS score out of 100 with reasons
5. Strengths of the resume
6. Weak or missing areas and suggestions to improve
7. Job Description:{job_description}

Resume Text:
{clean_text}

Provide the response in clear structured format.
"""

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": prompt}],
        )

        ai_text = response.choices[0].message.content
        return jsonify({"analysis": ai_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route('/')
def home_page():
      return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)



# 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