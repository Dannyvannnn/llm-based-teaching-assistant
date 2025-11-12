import exerciseformatter
import studentsolutionformatter
import solutionevaluator

soln_file_path = "rugby_football_club_solution.txt"
student_submission_directory = "submissions"

def main():
    ef = exerciseformatter.question_answer_extractor(soln_file_path)
    print(f"Extracted Question-Answer Pairs: {ef}")
    sf = studentsolutionformatter.scan_for_student_submissions(student_submission_directory)
    print(f"Student Submissions: {sf}\n")
    solutionevaluator.llm_eval_student_batch_process(sf, ef)


if __name__ == "__main__":
    main()