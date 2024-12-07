from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import json


# ========== Component 1: Question Answering ========== #
print("Loading Question Answering model...")
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


def answer_question(question, context, knowledge_base):
    """
    Answer questions based on a given context and save the answer to the knowledge base.
    """
    response = qa_model({"question": question, "context": context})
    answer = response.get("answer", "I couldn't find an answer.")

    # Check if the answer can be saved as a fact
    if answer != "I couldn't find an answer.":
        knowledge_base[question] = answer
        save_knowledge(knowledge_base)  # Save the updated knowledge base

    return answer


# ========== Component 2: Grammar Correction ========== #
print("Loading Grammar Correction model...")
grammar_model_name = "t5-large"  # Lightweight grammar correction model
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
    Retrieve a learned fact.
    """
    return knowledge_base.get(key, "I don't know that yet.")


# ========== Main AI Interaction ========== #
def main():
    print("\nWelcome to Your AI!")
    print("I can answer questions, fix grammar, and learn new things from you.")
    print("Type 'exit' to quit.")

    knowledge_base = load_knowledge()  # Load existing knowledge base

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

            # Check if the answer is already in the knowledge base
            if question in knowledge_base:
                print("Answer:", knowledge_base[question])
            else:
                print("Answer:", answer_question(question, context, knowledge_base))

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
