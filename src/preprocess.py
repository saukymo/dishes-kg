import re
import csv

from src.models import Dish
from src.constants import photo_captions, recommend_dishes, all_dishes
from src.utils import save_dishes

def load_origin_text(filename: str) -> set[str]:
    results = set()
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            normalized_text = normalize(row['text'])
            results.add(normalized_text)

    return results

pattern = re.compile(r'\(.*?\)|【.*?】|（.*?）')
def remove_parethesis(dish: str) -> str:
    return pattern.sub('', dish)

def normalize(dish: str) -> str:
    return remove_parethesis(dish.strip().lower())

def main():
    dishes = load_origin_text(photo_captions) | load_origin_text(recommend_dishes)
    print(len(dishes))
    save_dishes(
        [Dish(idx=idx, text=dish) for idx, dish in enumerate(dishes)],
        all_dishes,
        ['idx', 'text']
    )

if __name__ == '__main__':
    main()
