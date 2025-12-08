import json
import re
from typing import List, Dict, Any

# ==========================================
# 1. 模型載入區 (Pseudo Code / 偽代碼)
# ==========================================

class TargetModel:
    """這是要測試的 Model 看你要使用什麼, 麻煩你了QQ"""
    def generate(self, prompt: str) -> str:
        # TODO: 在這裡填入實際模型的推論代碼
        # return model.generate(prompt)
        return "Final Answer: B" # 假裝輸出的範例

class Llama3Evaluator:
    """這是用來評分的 Llama 3 模型 (LLM-as-a-Judge)，你可以選用其他模型"""
    def __init__(self):
        # TODO: 載入 Llama 3 模型或 API，或是其他模型
        pass

    def grade_answer(self, reference: str, prediction: str, question: str) -> int:
        """
        將參考答案、模型回答與問題組合成 Prompt, 讓 Llama 3 打分, 分數參照表我已經寫好了, prompt也設計好了
        """
        # 根據圖片中的表格設計的 Prompt
        rubric = """
        Score 0 (Not Valid): Unnatural, incoherent or unreadable.
        Score 1 (Terrible): Irrelevant to the question asked.
        Score 2 (Wrong): Different from the reference answer, but still relevant to the question.
        Score 3 (Right): Has the same meaning as the reference, but may be phrased differently.
        Score 4 (Excellent): Same as the reference or more naturally.
        """

        prompt = f"""
        You are an impartial judge evaluating the quality of an AI assistant's response to a question.
        
        Question: {question}
        Reference Answer: {reference}
        Candidate Response: {prediction}

        Please rate the Candidate Response based on the following rubric:
        {rubric}

        Output ONLY the integer score (0, 1, 2, 3, or 4). Do not output any explanation.
        """
        
        # TODO: 呼叫 Llama 3 生成回應
        # raw_score = self.model.generate(prompt)
        raw_score = "3" # 假裝 Llama 3 覺得回答得不錯
        
        # 解析分數 (防止模型多話)
        try:
            score = int(re.search(r'\d', raw_score).group())
            return score
        except:
            return 0

# ==========================================
# 只有上方需要請你寫code，下面的函數是我寫好可以直接用的
# ==========================================

def parse_final_answer_mc(full_text):
    """從選擇題的回答中提取 A/B/C/D/E"""
    # 優先尋找明確的標籤
    patterns = [
        r"Final Answer:\s*([A-E])",
        r"Answer:\s*([A-E])",
        r"Option:\s*([A-E])"
    ]
    for p in patterns:
        match = re.search(p, full_text, re.IGNORECASE)
        if match: return match.group(1).upper()
    
    # 如果沒有明確標籤，尋找最後出現的單獨字母
    matches = re.findall(r"\b([A-E])\b", full_text)
    if matches: return matches[-1].upper()
    return ""

def score_mc_acc(pred, correct_answer):
    """MCQ 評分"""
    if not pred: return 0.0
    return 1.0 if pred.strip().upper() == correct_answer.strip().upper() else 0.0

def format_mc_prompt(question, options):
    """格式化選擇題 Prompt"""
    prompt = f"Question: {question}\nOptions:\n"
    for idx, opt in enumerate(options):
        label = chr(65 + idx) # A, B, C...
        prompt += f"{label}. {opt}\n"
    prompt += "\nAnswer the question and strictly follow the format: 'Final Answer: [Option Letter]'."
    return prompt

def format_open_prompt(question):
    """格式化開放式問題 Prompt"""
    return f"Question: {question}\nAnswer directly. Finally, provide the conclusion."

# ==========================================
# 主流程邏輯
# ==========================================

def main_evaluation_pipeline(json_file_path):
    # 初始化模型
    target_model = TargetModel()
    judge_model = Llama3Evaluator()

    # 讀取 JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}

    # 遍歷每篇論文
    for paper_title, qa_list in data.items():
        print(f"Processing Paper: {paper_title}")
        paper_scores = {
            "mc_correct": 0,
            "mc_total": 0,
            "open_scores": [], # 存放 0-4 的分數
            "details": []
        }

        for item in qa_list:
            question = item['question']
            reference = item['answer']
            
            # --- 判斷題型 ---
            # 如果有 option 且 option 列表不為空，則是單選題
            if 'option' in item and item['option'] and len(item['option']) > 0:
                # === 單選題處理 (Multiple Choice) ===
                prompt = format_mc_prompt(question, item['option'])
                model_output = target_model.generate(prompt)
                
                # 解析選項
                parsed_pred = parse_final_answer_mc(model_output)
                
                # 評分 (0 或 1)
                is_correct = score_mc_acc(parsed_pred, reference)
                
                paper_scores["mc_total"] += 1
                paper_scores["mc_correct"] += is_correct
                
                paper_scores["details"].append({
                    "type": "MCQ",
                    "q": question,
                    "pred_raw": model_output,
                    "pred_parsed": parsed_pred,
                    "ref": reference,
                    "score": is_correct
                })

            else:
                # === 開放式問題處理 (Open-ended) ===
                prompt = format_open_prompt(question)
                model_output = target_model.generate(prompt)
                
                # 使用 Llama 3 或你選的其他模型進行語意評分 (0-4 分)
                similarity_score = judge_model.grade_answer(
                    reference=reference,
                    prediction=model_output,
                    question=question
                )
                
                paper_scores["open_scores"].append(similarity_score)
                
                paper_scores["details"].append({
                    "type": "Open",
                    "q": question,
                    "pred": model_output,
                    "ref": reference,
                    "score": similarity_score  # 這是 Llama 3 給出的 0-4 分
                })

        results[paper_title] = paper_scores

    # --- 輸出統計結果 ---
    print("\n=== Evaluation Summary ===")
    for title, stats in results.items():
        mc_acc = (stats['mc_correct'] / stats['mc_total']) * 100 if stats['mc_total'] > 0 else 0
        avg_open_score = (sum(stats['open_scores']) / len(stats['open_scores'])) if stats['open_scores'] else 0
        
        print(f"Paper: {title}")
        print(f"  - MCQ Accuracy: {mc_acc:.2f}% ({stats['mc_correct']}/{stats['mc_total']})")
        print(f"  - Open-ended Avg Score (0-4): {avg_open_score:.2f}")

    # 可以選擇將詳細結果存回 json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # json 檔案名為 papers.json
    main_evaluation_pipeline("papers.json")
    # 可以快樂結束了, 喔耶!!