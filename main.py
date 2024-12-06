from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import json

# ========== Component 1: Question Answering ========== #
print("Loading Question Answering model...")
qa_model = pipeline("question-answering")


def answer_question(question, context):
    """
    Answer questions based on a given context.
    """
    response = qa_model({"question": question, "context": context})
    return response.get("answer", "I couldn't find an answer.")


# ========== Component 2: Grammar Correction ========== #
print("Loading Grammar Correction model...")
grammar_model_name = "t5-small"  # Lightweight grammar correction model
grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
grammar_model = AutoModelForSeq2SeqLM.from_pretrained(grammar_model_name)


def fix_grammar(sentence):
    """
    Fix grammatical errors in a sentence.
    """
    input_text = f"fix grammar: {sentence}"
    inputs = grammar_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = grammar_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    corrected_sentence = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence


# ========== Component 3: Knowledge Learning ========== #
knowledge_base = {}  # In-memory knowledge base (use SQLite/DB for production)


def learn_fact(key, value):
    """
    Teach the AI a new fact.
    """
    knowledge_base[key] = value
    with open("knowledge_base.json", "w") as file:
        json.dump(knowledge_base, file)
    return f"I've learned: {key} -> {value}"


def retrieve_fact(key):
    """
    Retrieve a learned fact.
    """
    return knowledge_base.get(key, "I don't know that yet.")


# Load existing knowledge
try:
    with open("knowledge_base.json", "r") as file:
        knowledge_base = json.load(file)
except FileNotFoundError:
    pass  # No existing knowledge base


# ========== Main AI Interaction ========== #
def main():
    print("\nWelcome to Your AI!")
    print("I can answer questions, fix grammar, and learn new things from you.")
    print("Type 'exit' to quit.")

    while True:
        print("\nWhat would you like to do?")
        print("1: Ask a question")
        print("2: Fix grammar")
        print("3: Teach me something")
        print("4: Retrieve what I know")

        choice = input("Enter your choice: ").strip()
        if choice == "exit":
            print("Goodbye!")
            break

        if choice == "1":
            context = input("Provide some context: ").strip()
            question = input("What is your question? ").strip()
            print("Answer:", answer_question(question, context))

        elif choice == "2":
            sentence = input("Enter a sentence to fix: ").strip()
            print("Corrected Sentence:", fix_grammar(sentence))

        elif choice == "3":
            key = input("What should I learn? (Key): ").strip()
            value = input("What does it mean? (Value): ").strip()
            print(learn_fact(key, value))

        elif choice == "4":
            key = input("What do you want me to recall? ").strip()
            print("My answer:", retrieve_fact(key))

        else:
            print("Invalid choice. Please try again.")


# ========== Run the AI ========== #
if __name__ == "__main__":
    main()
