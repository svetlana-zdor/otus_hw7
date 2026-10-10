import os
import ollama

PROMPTS_FILE = "prompts.txt"
RESULTS_FILE = "compare_answers_resuls.txt"
MODEL_NAME = "qwen2.5" 


def load_prompts(filename):
  if not os.path.exists(filename):
    raise FileNotFoundError(f"Файл {filename} не найден!")

  with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

  parts = content.split("[PROMPT ")
  prompts = []
  for part in parts[1:]:
    lines = part.split("\n", 1)
    prompt_num = lines[0].strip("]")
    prompt_text = lines[1].strip() if len(lines) > 1 else ""
    prompts.append((f"Промпт {prompt_num}", prompt_text))
  return prompts


def main():
  try:
    prompts = load_prompts(PROMPTS_FILE)
  except Exception as e:
    print(f"Ошибка: {e}")
    return

  results = []

  for name, prompt in prompts:
    print(f"Обработка: {name} (модель: {MODEL_NAME})...")

    response = ollama.chat(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
    )
    generated_text = response["message"]["content"]

    eval_prompt = f"""
        Оцени следующий ответ LLM по трем критериям по шкале от 1 до 5:
        1. Полнота (насколько полно покрыты граничные случаи функции calculate_discount).
        2. Формат (насколько структура удобна для чтения, структурирована или табличная).
        3. Точность (насколько корректны ожидаемые результаты тестов и описания).

        Ответ для оценки:
        {generated_text}

        Выдай оценку строго в формате:
        Полнота: [оценка]/5
        Формат: [оценка]/5
        Точность: [оценка]/5
        Комментарий: [краткое обоснование оценки]
        """

    print(f"Оценка качества для {name}...")
    eval_response = ollama.chat(
        model=MODEL_NAME, messages=[{"role": "user", "content": eval_prompt}]
    )
    evaluation = eval_response["message"]["content"]

    result_entry = f"""=== {name} ===
ПРОМПТ:
{prompt}

СГЕНЕРИРОВАННЫЕ ТЕСТ-КЕЙСЫ:
{generated_text}

ОЦЕНКА КАЧЕСТВА (LLM-AS-A-JUDGE):
{evaluation}

{'=' * 50}

"""
    results.append(result_entry)

  # Сохранение результатов в файл compare_answers_resuls.txt
  with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.writelines(results)

  print(f"Готово! Результаты успешно записаны в файл {RESULTS_FILE}")


if __name__ == "__main__":
  main()