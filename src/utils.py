import csv

from langchain_core.messages import AIMessage

from src.models import Dish

def parse_deepseek_output(output: AIMessage) -> str:
    response = str(output.content)

    print('response', response)
    if '</think>' in response:
        response_without_think = response.split('</think>\n\n')[1]
        return response_without_think.strip()
    return response.strip()

def load_dishes(filename: str) -> list[Dish]:
    results: list[Dish] = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            results.append(Dish(**row)) # type: ignore
    return results

def save_dishes(dishes: list[Dish], filename: str, fieldnames: list[str]):
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in dishes:
            writer.writerow(row.dict())
