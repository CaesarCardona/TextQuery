from transformers import pipeline
from odf.opendocument import load
from transformers import pipeline

# Specify a model explicitly
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad"  # You can specify any other model too!
)

def preprocess_text(text):
    words = text.split()  # Simple whitespace-based tokenization
    return words

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering")

# Function to read .txt files
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Function to read .odt files
def read_odt_file(file_path):
    doc = load(file_path)
    text = ""
    for paragraph in doc.getElementsByType('text:p'):
        text += paragraph.firstChild.data + "\n"
    return text



# Function to answer questions based on the text
def answer_question(text, question):
    result = qa_pipeline(question=question, context=text)
    return result['answer']

# Main function
def ai_agent(file_path, file_type, question):
    if file_type == 'txt':
        text = read_txt_file(file_path)
    elif file_type == 'odt':
        text = read_odt_file(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    # Preprocess the text if needed
    processed_text = preprocess_text(text)
    
    # Answer the question based on the file content
    answer = answer_question(text, question)
    return answer

# Run this last, after all functions are defined
if __name__ == "__main__":
    file_path = "sample.txt"
    file_type = "txt"
    question = "What does actinin do?"

    answer = ai_agent(file_path, file_type, question)
    print("Answer:", answer)

# Save the answer to a file
def save_answer_to_file(answer, output_file="answer.txt"):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(answer)

# Usage
file_path = "sample.txt"
file_type = "txt"
question = "What does actinin do?"

answer = ai_agent(file_path, file_type, question)

# Save the answer to "answer.txt"
save_answer_to_file(answer, "answer.txt")
