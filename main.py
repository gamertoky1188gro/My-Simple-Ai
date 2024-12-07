from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from difflib import get_close_matches
import json
import requests


def search_internet_with_custom_search(question):
    """
    Search the internet using Google's Custom Search JSON API.
    """
    api_key = "AIzaSyBlREWzWan1HTPXCJACm-2ofK0UT3ggt_Q"  # Replace with your actual API Key
    search_engine_id = "c114e1dee659c4825"  # Replace with your actual Search Engine ID (cx)
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": question,
        "num": 1,  # Fetch only the top result
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        if "items" in results:
            answer = results["items"][0].get("snippet", "No relevant result found.")
            return answer
        else:
            return "I couldn't find any relevant information on the internet."
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"


# ========== Component 1: Question Answering ========== #
print("Loading Question Answering model...")
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


def answer_question(question, context, knowledge_base):
    """
    Answer questions based on a given context, knowledge base, or the internet.
    """
    questions = [q.strip() for q in question.split("and")]
    answers = {}

    for q in questions:
        if q in knowledge_base:
            answers[q] = knowledge_base[q]
        elif context.strip():
            try:
                response = qa_model(question=q, context=context)
                answer = response.get("answer", "I couldn't find an answer.")
                if answer != "I couldn't find an answer.":
                    knowledge_base[q] = answer
                    save_knowledge(knowledge_base)
                answers[q] = answer
            except Exception as e:
                answers[q] = f"Error processing question: {str(e)}"
        else:
            internet_answer = search_internet_with_custom_search(q)
            answers[q] = internet_answer

    return answers


# ========== Component 2: Grammar Correction ========== #
print("Loading Grammar Correction model...")
grammar_model_name = "vennify/t5-base-grammar-correction"
grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
grammar_model = AutoModelForSeq2SeqLM.from_pretrained(grammar_model_name)


def fix_grammar(sentence):
    """
    Fix grammatical errors in a sentence.
    """
    try:
        input_text = f"fix grammatical errors: {sentence}"
        inputs = grammar_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = grammar_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        corrected_sentence = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print("Debug: Raw Output:", grammar_tokenizer.decode(outputs[0]))

        if corrected_sentence.lower() == input_text.lower():
            return "The grammar model returned unexpected results. Please verify the model."
        return corrected_sentence
    except Exception as e:
        return f"Error during grammar correction: {str(e)}"


# ========== Component 3: Knowledge Learning ========== #
def load_knowledge():
    """
    Load the knowledge base from a JSON file.
    """
    try:
        with open("knowledge_base.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def save_knowledge(knowledge_base):
    """
    Save the knowledge base to a JSON file.
    """
    with open("knowledge_base.json", "w") as file:
        json.dump(knowledge_base, file, indent=4)


def learn_fact(key, value, knowledge_base):
    """
    Teach the AI a new fact.
    """
    knowledge_base[key] = value
    save_knowledge(knowledge_base)
    return f"I've learned: {key} -> {value}"


def retrieve_fact(key, knowledge_base):
    """
    Retrieve a learned fact with support for approximate matching.
    """
    if key in knowledge_base:
        return knowledge_base[key]

    close_matches = get_close_matches(key, knowledge_base.keys(), n=1, cutoff=0.6)
    if close_matches:
        closest_key = close_matches[0]
        return f"Did you mean '{closest_key}'? {knowledge_base[closest_key]}"

    return "I don't know that yet."


# ========== Main AI Interaction ========== #
def main():
    print("\nWelcome to Your AI!")
    print("I can answer questions, fix grammar, and learn new things from you.")
    print("Type 'exit' to quit.")

    knowledge_base = load_knowledge()

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
            context = input("Provide some context (or leave blank to search online): ").strip()
            question = input("What is your question? ").strip()

            questions = [q.strip() for q in question.split("and")]
            answers_to_display = {}

            for q in questions:
                if q in knowledge_base:
                    answers_to_display[q] = knowledge_base[q]
                elif context.strip():
                    try:
                        response = qa_model(question=q, context=context)
                        answer = response.get("answer", "I couldn't find an answer.")

                        if answer != "I couldn't find an answer.":
                            knowledge_base[q] = {"answer": answer, "context": context}
                            save_knowledge(knowledge_base)

                        answers_to_display[q] = answer
                    except Exception as e:
                        answers_to_display[q] = f"Error processing question: {str(e)}"
                else:
                    internet_answer = search_internet_with_custom_search(q)

                    knowledge_base[q] = {"answer": internet_answer, "context": "Searched online"}
                    save_knowledge(knowledge_base)

                    answers_to_display[q] = internet_answer

            for q, a in answers_to_display.items():
                print(f"Q: {q}\nA: {a}\n")

        elif choice == "2":
            sentence = input("Enter a sentence to fix: ").strip()
            print("Corrected Sentence:", fix_grammar(sentence))

        elif choice == "3":
            key = input("What should I learn? (Key): ").strip()
            value = input("What does it mean? (Value): ").strip()
            print(learn_fact(key, value, knowledge_base))

        elif choice == "4":
            key = input("What do you want me to recall? ").strip()
            print("My answer:", retrieve_fact(key, knowledge_base))

        else:
            print("Invalid choice. Please try again.")


# ========== Run the AI ========== #
if __name__ == "__main__":
    main()
