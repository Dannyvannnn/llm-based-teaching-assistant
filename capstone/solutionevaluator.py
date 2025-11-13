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
    """
    Compare strings a and b to see how semantically similary they are to each other
    return a float value showing the cosine similarity
    """
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

    # Clean inputs
    if not output or not reference_answer:
        return 0.0
    out = output.strip().lower()
    ref = reference_answer.strip().lower()

    # Get token overlap
    out_tokens = set(out.split())
    ref_tokens = set(ref.split())
    overlap = len(out_tokens & ref_tokens) / max(1, len(ref_tokens))

    # Semantic similarity via embeddings
    cosine_sim = semantic_similarity (out, ref)

    # Return true if overlap or cosine similarities are at least 90%
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
    
    # remove UI text
    text = re.sub(r'(?i)(filter:.*|status bar.*)', '', text).strip()

    # remove "Ans:"
    text = re.sub(r'(?i)(ans\s*[:\-]?\s*)', '', text).strip()
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
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                llm_int8_enable_fp32_cpu_offload=True
                )
            
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
    """
    Evaluate a student's answer using an LLM and return a structured JSON evaluation.
    Works for both OpenAI API and local HuggingFace models.
    """

    # --- Construct prompt ---
    prompt = f"""
You are grading a question. Return ONLY a JSON object.

Question: {question}
Correct Answer: {reference_answer}
User Answer: {user_answer}

JSON fields:
- "correctness": "correct", "partially correct", or "incorrect"
- "score": float between 0 and 1
- "explanation": short string

Return JSON like this:
{{"correctness": "correct", "score": 1.0, "explanation": "User gave correct answer."}}
"""

    # --- Call LLM ---
    start_time = time.time()
    response = get_response_with_model(prompt, **model_settings)
    end_time = time.time()
    duration = end_time - start_time

    # --- Extract output text ---
    if "choices" in response and response["choices"]:
        llm_content = response["choices"][0].get("message", {}).get("content", "")
    else:
        llm_content = ""

    llm_content = str(llm_content).strip()

    # --- Count tokens ---
    if "usage" in response and "total_tokens" in response["usage"]:
        token_count = response["usage"]["total_tokens"]
    else:
        token_count = len(llm_content.split())  # fallback for local models

    # --- Compute hybrid accuracy ---
    accuracy = hybrid_accuracy(reference_answer, llm_content)

    # --- Extract JSON safely ---
    try:
        # Extract ```json blocks first
        json_match = re.search(r'```json\s*(.*?)\s*```', llm_content, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = llm_content

        # Extract first {...} block to remove extra text
        json_match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
        else:
            raise json.JSONDecodeError("No JSON object found", json_string, 0)

        json_data = json.loads(json_string)

    except json.JSONDecodeError:
        print("Warning: Could not parse JSON from LLM response.")
        print(f"Raw LLM response (first 200 chars): {llm_content[:200]}")
        json_data = {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": "LLM returned invalid JSON.",
        }

    # --- Append extra info ---
    json_data["accuracy"] = accuracy
    json_data["token_count"] = token_count
    json_data["duration"] = duration

    return json_data


# --- Evaluate student submissions ---
def evaluate_student_submission(merged_data, model_settings):
    results = []

    # question_id, question_text, model_ans, student_ans_raw are now directly available from merged_data
    for question_id, question_text, model_ans, student_ans_raw in merged_data:
        # Pass the specific question_text to extract_user_answer for removal
        clean_answer = extract_user_answer(student_ans_raw, question_text)

        # Always use LLM evaluation
        if len(clean_answer) >= 1:
            result = llm_eval(question_text, clean_answer, model_ans, model_settings)
        else:
            result = llm_eval(question_text, student_ans_raw, model_ans, model_settings)

        # Format results properly
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
    """
    Write data from 'student_evaluation_list' in a csv file under the directory 'base_output_dir\student_name'
    csv file name is model + "_evaluation.csv
    """
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
    
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # write to file
    with open(filepath,"w", newline = "", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=student_evaluation_list[0].keys())
        writer.writeheader()
        writer.writerows(student_evaluation_list)


def batch_process_student_submissions(student_submissions_data : list[str], questions_answers : list[dict], llm_choice = "gpt-4o"):
    """
    Batch-evaluates student submissions using an LLM and writes per-student evaluation files.

    Args:
        student_submissions_data (list[str]): List of dicts, each containing 'student_name' and 'submission_path'.
        questions_answers (list[dict]): List of dicts with 'question_id', 'question', and 'answer' keys.
        llm_choice (str): Model choice (default: 'gpt-4o').

    Returns:
        tuple: (model_name, metadata_eval_overall)
            model_name (str): The evaluated model name.
            metadata_eval_overall (list[dict]): Per-question metadata such as accuracy, tokens, and duration.
    """
    # Load model configuration and runtime settings for the selected LLM
    model_settings = get_llm_setting(llm_choice)

    # Will collect global metadata (accuracy, token usage, timing) for analysis
    metadata_eval_overall = []

    # Process each student's submission file
    for student in student_submissions_data:
        # List of tuples: (question_id, question_text, model_answer, student_answer)
        merged = []
        # Results for this specific student
        student_answer_eval = []
        # Parse student’s submitted answers (assumes returns {'answers': {'1.': '...', '2.': '...'}})
        student_data = studentsolutionformatter.parse_student_submission(student['submission_path'])

        # Set up questions to be evaluated by LLM
        for q in questions_answers:
            q_num = str(q['question_id']) + '.'  # Ensure consistent key format (e.g., "1.")
            question_text = q['question']
            model_ans = q['answer']
            student_ans = student_data['answers'].get(q_num, "")  # fallback empty string if missing

            merged.append((q_num, question_text, model_ans, student_ans)) # Include question_id

        # --- Run evaluation ---
        # 'evaluate_student_submission' should return structured data with scoring and metadata
        results = evaluate_student_submission(merged, model_settings)

        # Process evaluation results and collect both answer-level and model-level metadata
        for r in results:
            # Skip entries with missing question_id (defensive)
            if (r['question_id'] == "None."):
                continue

            # Store per-question evaluation results
            student_answer_results = {
                "question_id": r['question_id'],
                "answer": r['raw_user_answer'],
                "correctness": r['correctness'],
                "score": r['score'],
                "explanation": r['explanation']
            }

            # Collect LLM metadata for performance/tracking
            llm_metadata = {
                'accuracy': r['accuracy'],
                'token_count': r['token_count'],
                'duration': r['duration']
            }

            student_answer_eval.append(student_answer_results)
            metadata_eval_overall.append(llm_metadata)

        # --- Write evaluation results for this student ---
        # Ensure output directory exists before writing
        try:
            Write_Student_Evaluation(student_answer_eval, 
                                     student['student_name'], 
                                     model=model_settings['model'])
        except FileNotFoundError:
            print(f"⚠️ Skipped writing for {student['student_name']}: output path not found.")
    
    # Return the evaluated model name and collected metadata
    return model_settings['model'], metadata_eval_overall

def get_llm_setting (llm_choice: str):
    """
    Get predefined LLM settings to be used for the evaluator
    """
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
        # "mpt-7b": {
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
    # Special command for getting a list of all llm model names
    if llm_choice.lower() == "all":
        return LLM_SETTINGS.keys()
    
    # Otherwise return 
    return LLM_SETTINGS[llm_choice.lower()]

def llm_eval_student_batch_process (student_submissions_data : list[str], questions_answers : list[dict]):
    """
    Run through all predefined LLMs to evaluate accuracy, token count and llm duration for a given list of 
    student answers and answer key, allowing evaluation of predefined models
    """
    # Get list of available llms
    llm = get_llm_setting("All")
    # print(f"LLM list: {llm}")

    llm_results = []
    # processing LLM evaluation csv
    for settings in llm:
        torch.cuda.empty_cache()
        # print (LLM_SETTINGS[settings])
        llm_results.append(batch_process_student_submissions(student_submissions_data, questions_answers, settings))

    # Store results
    summary = {}

    # Summarizing results
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


