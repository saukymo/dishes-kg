import asyncio

from langchain_ollama import ChatOllama
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate

from src.models import Dish
from src.constants import LLM_MODEL, BATCH_SIZE, all_dishes, tokenized_dishes
from src.utils import parse_deepseek_output, load_dishes, save_dishes

def get_few_shot_examples() -> list[dict[str, str]]:
    return [
        {
            'input': '酸辣土豆丝',
            'output': '酸辣|土豆丝'
        },
        {
            'input': '水煮肉片',
            'output': '水煮|肉片'
        },
        {
            'input': '腊味饭',
            'output': '腊味|饭'
        },
        {
            'input': '正新鸡排',
            'output': '正新|鸡排'
        },
        {
            'input': '小炒黄牛肉',
            'output': '小炒|黄牛肉'
        },
        {
            'input': '牛气冲天堡',
            'output': '牛气|冲天|堡'
        },
        {
            'input': '孜然羊肉盖烧饭',
            'output': '孜然|羊肉|盖烧饭'
        },
        {
            'input': '富士苹果',
            'output': '富士|苹果'
        },
        {
            'input': '椒麻小酥肉',
            'output': '椒麻|小酥肉'
        },
        {
            'input': '正山小种',
            'output': '正山|小种'
        }
    ]

def get_llm_with_prompt():
    system = """你是一个精通中文热爱美食的语言学家, 你需要帮忙给用户输入的美食分词。""" \
    """反思一下，你给出的结果中，词语的数量是否和输入的词数一致。不要多加也不要遗漏。""" \
    """回答之前，请反复确认输出是每个词语的分词结果，每个分词后的部分之间用|隔开。不要有其他任何额外的内容或者空行。"""

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples = get_few_shot_examples(),
        example_prompt = (
            HumanMessagePromptTemplate.from_template("{input}")
            + AIMessagePromptTemplate.from_template("{output}")
        )
    )

    chat_prompt_with_system = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system),
            few_shot_prompt,
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    llm: ChatOllama = ChatOllama(model=LLM_MODEL, temperature=0)

    llm_with_prompt = chat_prompt_with_system | llm | parse_deepseek_output

    return llm_with_prompt


async def get_batch_dishes_result(llm, dishes: list[str]) -> list[str]:

    response = await llm.abatch([{'input': dish} for dish in dishes])
    return response


async def process_dishes(dishes: list[Dish], candidates: list[tuple[int, str]], llm):
    print(len(candidates), candidates)
    response = await get_batch_dishes_result(llm, [candidate[1] for candidate in candidates])

    assert len(response) == len(candidates), f"results: {response}m candidates: {candidates}"

    for result in zip(candidates, response):
        _id, dish = result[0]
        tokens = result[1]
        print(_id, dish, tokens)
        dishes[_id].tokens = tokens

    save_dishes(dishes, tokenized_dishes, ['idx', 'text', 'tokens'])

async def tokenizer(dishes: list[Dish]):
    llm = get_llm_with_prompt()
    candidates = [(i, dish.text) for i, dish in enumerate(dishes)]
    for i in range(0, len(candidates), BATCH_SIZE):
        await process_dishes(dishes, candidates[i:i+BATCH_SIZE], llm)

async def main():
    dishes = load_dishes(all_dishes)

    await tokenizer(dishes)

if __name__ == '__main__':
    asyncio.run(main())
