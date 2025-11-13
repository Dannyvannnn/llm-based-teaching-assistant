import json
import os
import re
import csv
import time

import numpy as np
import pandas as pd

import studentsolutionformatter
from openai_client import client
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# is_scoring = True


def semantic_similarity(a, b):
    emb = client.embeddings.create(input=[a, b], model="text-embedding-3-small")
    v1, v2 = emb.data[0].embedding, emb.data[1].embedding
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def hybrid_accuracy(output: str, reference_answer: str):
    """
    Compute a hybrid accuracy score between model output and reference answer
    using:
      - semantic similarity (OpenAI embeddings)
      - token overlap (bag-of-words)
    Returns a float between 0 and 1.
    """

    # 🧼 Clean inputs
    if not output or not reference_answer:
        return 0.0
    out = output.strip().lower()
    ref = reference_answer.strip().lower()

    # Token overlap
    out_tokens = set(out.split())
    ref_tokens = set(ref.split())
    overlap = len(out_tokens & ref_tokens) / max(1, len(ref_tokens))

    # Semantic similarity via embeddings
    cosine_sim = semantic_similarity (out, ref)

    if overlap > 0.9 or cosine_sim > 0.9:
        return 1

    # Weighted combination
    hybrid_score = cosine_sim + overlap
    return min(1, hybrid_score)


def extract_user_answer(user_input, question_text_to_remove=None) -> str:
    # Ensure user_input is a string before attempting .strip()
    text = str(user_input).strip()

    if question_text_to_remove:
        # Remove repeated question if it's at the beginning of the user's answer
        # Use re.escape to handle special characters in the question text
        question_in_answer = re.escape(question_text_to_remove)
        # Remove only once if it's at the very beginning (case-insensitive)
        text = re.sub(rf'(?is)^{question_in_answer}\s*\??', '', text, 1).strip()

    text = re.sub(r'(?i)(filter:.*|status bar.*)', '', text).strip()     # remove UI text
    text = re.sub(r'(?i)(ans\s*[:\-]?\s*)', '', text).strip()            # remove "Ans:"
    return text


# --- LLM evaluator with adjustable model ---
def get_response_with_model(
        prompt, 
        *, 
        backend = "openai",
        model="gpt-4o", 
        temperature=None,
        top_p=None,
        max_tokens=None,
        load_in_4bit=False
        ):
    
    # Set up LLM using correct format
    # ----------------- OpenAI -----------------
    if backend == 'openai':
        # Build the parameters dictionary
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }

        # Only include parameters if they are not None
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        response = client.chat.completions.create(**params)
        return response.to_dict()
    
    # ----------------- Hugging Face / Local -----------------
    elif backend == "huggingface":
        # generator = pipeline("text-generation", model=model)
        tokenizer = AutoTokenizer.from_pretrained(model)

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            generator = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                dtype=torch.float16,
                quantization_config=bnb_config
            )
        else:
            generator = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                dtype=torch.float16
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)

        params = {}

        # Only include parameters if they are not None
        if temperature is not None:
            params["temperature"] = temperature
            params["do_sample"] = True
        if top_p is not None:
            params["top_p"] = top_p
            params["do_sample"] = True
        if max_tokens is not None:
            params["max_new_tokens"] = max_tokens
            params["do_sample"] = True

        # response = generator(prompt, **params)
        output_ids = generator.generate(
            **inputs,
            **params
        )

        # Decode into text
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Wrap in OpenAI-like response format
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    }
                }
            ]
        }
        return response


# --- LLM evaluator ---
def llm_eval(question, user_answer, reference_answer, model_settings):
    prompt = f"""
You are grading a question.

Question: {question}
Correct Answer: {reference_answer}
User Answer: {user_answer}

Evaluate the user's answer in JSON with:
- correctness: ["correct", "partially correct", "incorrect"]
- score: 0 to 1
- explanation: short reason
"""
    
    start_time = time.time() 
    response = get_response_with_model(prompt, **model_settings)
    end_time = time.time()
    
    # quantity scoring
    output = response["choices"][0]["message"]["content"].strip()
    duration = end_time - start_time

    # Count tokens in output for both backends
    if "usage" in response and "total_tokens" in response["usage"]:
        token_count = response["usage"]["total_tokens"]
    else:
        # Fallback for local / HF models
        token_count = len(output.split())

    # Accuracy Scoring
    # accuracy = int(reference_answer.lower() in output.lower())
    accuracy = hybrid_accuracy(reference_answer, output)

    # Add checks for empty response or choices
    if not response or "choices" not in response or not response["choices"]:
        print("Warning: LLM response or choices list is empty.")
        return {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": "LLM returned an empty response or no choices."
        }

    # Access the assistant message content
    llm_content = response["choices"][0].get("message", {}).get("content", "")
    if not llm_content:
        print("Warning: LLM message content is empty.")
        return {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": "LLM returned empty message content."
        }

    # Ensure llm_content is a string
    llm_content = str(llm_content)

    # Extract JSON from markdown code block if present
    json_match = re.search(r'```json\s*(.*?)\s*```', llm_content, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        json_string = llm_content.strip()  # fallback, assume pure JSON

    try:
        json_data = json.loads(json_string)

        # Append extra evaluation info
        json_data["accuracy"] = accuracy
        json_data["token_count"] = token_count
        json_data["duration"] = duration

        return json_data

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Raw LLM response (attempted parse): {json_string}")
        print(f"Original LLM content: {llm_content}")
        # Return default 'incorrect' evaluation to continue processing
        return {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": f"LLM returned malformed JSON. Original response: {llm_content[:100]}..."
        }


# --- Evaluate student submissions ---
def evaluate_student_submission(merged_data, model_settings):
    results = []

    for question_id, question_text, model_ans, student_ans_raw in merged_data:
        # question_id, question_text, model_ans, student_ans_raw are now directly available from merged_data

        # Pass the specific question_text to extract_user_answer for removal
        clean_answer = extract_user_answer(student_ans_raw, question_text)

        # Always use LLM evaluation
        if len(clean_answer) >= 1:
            result = llm_eval(question_text, clean_answer, model_ans, model_settings)
        else:
            result = llm_eval(question_text, student_ans_raw, model_ans, model_settings)

        results.append({
            "question_id": question_id,
            "question": question_text,
            "model_answer": model_ans,
            "raw_user_answer": student_ans_raw,
            "clean_user_answer": clean_answer,
            **result
        })
    return results


def Write_Student_Evaluation(student_evaluation_list: list, student_name: str, *, base_output_dir: str = "submissions", model = ""):
    # Construct the directory path for the student
    student_output_dir = os.path.join(base_output_dir, student_name)

    # Ensure the student's directory exists
    os.makedirs(student_output_dir, exist_ok=True)

    # Construct the full filepath for the evaluation CSV
    if model != "":
        csv_name = model + "_evaluation.csv"
    else:
        csv_name = "evaluation.csv"

    filepath = os.path.join(student_output_dir, csv_name)

    with open(filepath,"w", newline = "", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=student_evaluation_list[0].keys())
        writer.writeheader()
        writer.writerows(student_evaluation_list)


def batch_process_student_submissions(student_submissions_data : list[str], questions_answers : list[dict], llm_choice = "gpt-4o"):
    model_settings = get_llm_setting(llm_choice)
    metadata_eval_overall = []
    
    for student in student_submissions_data:
        merged = []
        student_answer_eval = []
        student_data = studentsolutionformatter.parse_student_submission(student['submission_path'])

        for q in questions_answers:
            q_num = str(q['question_id']) + '.'  # match student_submission keys
            question_text = q['question']
            model_ans = q['answer']
            student_ans = student_data['answers'].get(q_num, "")  # fallback empty string if missing
            merged.append((q_num, question_text, model_ans, student_ans)) # Include question_id

        # --- Run evaluation ---
        # Now passing only merged directly to the evaluation function
        results = evaluate_student_submission(merged, model_settings)

        for r in results:
            if (r['question_id'] == "None."):
                continue

            student_answer_results = {
                "question_id": r['question_id'],
                "answer": r['raw_user_answer'],
                "correctness": r['correctness'],
                "score": r['score'],
                "explanation": r['explanation']
            }

            llm_metadata = {
                'accuracy': r['accuracy'],
                'token_count': r['token_count'],
                'duration': r['duration']
            }

            student_answer_eval.append(student_answer_results)
            metadata_eval_overall.append(llm_metadata)

        Write_Student_Evaluation(student_answer_eval, student['student_name'], model = model_settings['model'])
    return model_settings['model'], metadata_eval_overall

def get_llm_setting (llm_choice: str):
    # --- LLM Settings to be tested --- 
    LLM_SETTINGS = {
        # -------------------- OpenAI models --------------------
        
        "gpt-4o": {
            "backend": "openai",
            "model": "gpt-4o",
            "temperature": 0,
            "top_p": 1,
        },
        "gpt-4o-mini": {
            "backend": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0,
            "top_p": 1,
        },
        "gpt-3.5-turbo": {
            "backend": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "top_p": 1,
        },
        
        # -------------------- Hugging Face / Local models --------------------
        # "llama-2-7b": {
        #     "backend": "huggingface",
        #     "model": "meta-llama/Llama-2-7b-chat-hf",
        #     "temperature": 0.1,
        #     "top_p": 1,
        # },
        # "mpt-7b-instruct": {
        #     "backend": "huggingface",
        #     "model": "mosaicml/mpt-7b",
        #     "temperature": 0.1,
        #     "top_p": 0.9,
        #     "max_tokens": 500,
        #     "load_in_4bit":True
        # },
        # "gpt4all-mini": {
        #     "backend": "huggingface",
        #     "model": "nomic-ai/gpt4all-mini",
        #     "temperature": 0.1,
        #     "top_p": 0.9,
        #     "max_tokens": 500,
        #     "load_in_4bit":True
        # }
    }
    if llm_choice == "All":
        return LLM_SETTINGS.keys()
    return LLM_SETTINGS[llm_choice]

def llm_eval_student_batch_process (student_submissions_data : list[str], questions_answers : list[dict]):
    llm = get_llm_setting("All")
    print(f"LLM list: {llm}")

    llm_results = []
    # processing LLM evaluation csv
    for settings in llm:
        # print (LLM_SETTINGS[settings])
        llm_results.append(batch_process_student_submissions(student_submissions_data, questions_answers, settings))

    # Store results
    summary = {}

    for model, results in llm_results:
        total_accuracy = sum(item['accuracy'] for item in results)
        total_tokens = sum(item['token_count'] for item in results)
        total_duration = sum(item['duration'] for item in results)
        count = len(results)
        
        summary[model] = {
            'total_accuracy': total_accuracy,
            'avg_accuracy': total_accuracy / count,
            'total_tokens': total_tokens,
            'total_duration': total_duration,
            'avg_duration': total_duration / count,
            'num_entries': count
        }

    # Convert to DataFrame for nice display
    df = pd.DataFrame(summary).T
    print(df)

    # Optional: Save to CSV
    df.to_csv("llm_evaluation_summary.csv", index=True)
    print(" Saved CSV: llm_evaluation_summary.csv")

