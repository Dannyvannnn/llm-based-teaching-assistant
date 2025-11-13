import os
import pandas as pd

def combine_student_evaluations(submissions_dir, model_name = "gpt-4o", output_csv=None):
    """
    Combines LLM evaluation CSVs for a specific model across student submissions
    and generates a table showing the correctness score of each student for each question.

    Args:
        submissions_dir (str): Path to the main submissions directory.
        model_name (str): The model name to look for in CSV files (e.g., "gpt-4o").
        output_csv (str, optional): Path to save the combined CSV. If None, CSV is not saved.

    Returns:
        pd.DataFrame: Combined correctness scores with students as rows and questions as columns.
    """
    combined_data = []

    # Loop through each student directory
    for student_name in os.listdir(submissions_dir):
        student_path = os.path.join(submissions_dir, student_name)
        # print (f"Student Path: {student_path}")
        if not os.path.isdir(student_path):
            continue

        # Look for CSV file for this model
        csv_filename = f"{model_name}_evaluation.csv"
        csv_path = os.path.join(student_path, csv_filename)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Extract correctness scores
            correctness_scores = df.set_index('question_id')['score'].to_dict()
            correctness_scores['student'] = student_name
            combined_data.append(correctness_scores)

    # Combine all students into a single DataFrame
    combined_df = pd.DataFrame(combined_data)
    combined_df.set_index('student', inplace=True)

    # Optionally save to CSV
    if output_csv == None:
        combined_df.to_csv(model_name + "_student_score_matrix.csv")
    else:
        combined_df.to_csv(output_csv)

    return combined_df
