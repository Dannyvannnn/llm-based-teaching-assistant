import json
import os
import re
import csv
import time

import studentsolutionformatter
from openai_client import client

is_scoring = True


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
        model="gpt-4o", 
        temperature=None,
        top_p=None,
        max_tokens=None,
        ):
    
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
    if is_scoring:
        start_time = time.time()
    
    response = get_response_with_model(prompt, **model_settings)

    if is_scoring:
        end_time = time.time()
        
        # quantity scoring
        output = response.choices[0].message.content.strip()
        token_count = response.usage.total_tokens if hasattr(response, "usage") else len(output.split())
        duration = end_time - start_time

        # Accuracy
        accuracy = int(reference_answer.lower() in output.lower())

    # Add checks for empty response or choices
    if not response or not response.choices:
        print("Warning: LLM response or choices list is empty.")
        return {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": "LLM returned an empty response or no choices."
        }

    llm_content = response.choices[0].message.content
    if not llm_content:
        print("Warning: LLM message content is empty.")
        return {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": "LLM returned empty message content."
        }

    # Extract JSON from markdown code block if present
    json_match = re.search(r'```json\s*(.*?)\s*```', llm_content, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        json_string = llm_content # Assume it's pure JSON if no markdown block

    try:
        json_data = json.loads(json_string)

        if is_scoring:
            # json_data["model"] = model_settings['model']
            json_data["accuracy"] = accuracy
            json_data["token_count"] = token_count
            json_data["duration"] = duration

        return json_data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Raw LLM response (attempted parse): {json_string}")
        print(f"Original LLM content: {llm_content}")
        # Return a default 'incorrect' evaluation to allow processing to continue
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


def batch_process_student_submissions(student_submissions_data : list[str], questions_answers : list[dict], model_settings):
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

        # print("-----", student['student_name'], "-----")
        for r in results:
            if (r['question_id'] == "None."):
                continue

            student_answer_results = {
                # "student_name": student['student_name'],
                "question_id": r['question_id'],
                "answer": r['raw_user_answer'],
                "correctness": r['correctness'],
                "score": r['score'],
                "explanation": r['explanation']
            }

            if is_scoring:
                # student_answer_results["model"] = r['model']
                student_answer_results["accuracy"] = r['accuracy']
                student_answer_results["token_count"] = r['token_count']
                student_answer_results["duration"] = r['duration']

            student_answer_eval.append(student_answer_results)

        Write_Student_Evaluation(student_answer_eval, student['student_name'], model = model_settings['model'])

def llm_eval_student_batch_process (student_submissions_data : list[str], questions_answers : list[dict]):
    # --- LLM Settings to be tested --- 
    LLM_SETTINGS = {
        "gpt-4o": {
            "model": "gpt-4o",
            "temperature": 0,
            "top_p": 1,
        },
        "gpt-4o-mini": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "top_p": 1,
        },
        "gpt-3.5-turbo": {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "top_p": 1,
        },
    }

    # processing LLM evaluation csv
    for settings in LLM_SETTINGS:
        # print (LLM_SETTINGS[settings])
        batch_process_student_submissions(student_submissions_data, questions_answers, LLM_SETTINGS[settings])

    # Go through each csv to sum up time taken, accuracy and token count

