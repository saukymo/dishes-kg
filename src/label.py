import asyncio

from langchain_ollama import ChatOllama
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate

from src.models import Dish
from src.constants import LLM_MODEL, BATCH_SIZE, labeled_dishes, tokenized_dishes
from src.utils import parse_deepseek_output, load_dishes, save_dishes

def get_few_shot_examples() -> list[dict[str, str]]:
    return [
        {
            'input': '酸辣|土豆丝',
            'output': '风味|材料'
        },
        {
            'input': '水煮|肉片',
            'output': '工艺|材料'
        },
        {
            'input': '腊味|饭',
            'output': '风味|形式'
        },
        {
            'input': '正新|鸡排',
            'output': '品牌|材料'
        },
        {
            'input': '小炒|黄牛肉',
            'output': '工艺|材料'
        },
        {
            'input': '牛气|冲天|堡',
            'output': '其他|其他|形式'
        },
        {
            'input': '孜然|羊肉|盖烧饭',
            'output': '原材料|材料|形式'
        },
        {
            'input': '富士|苹果',
            'output': '地名|材料'
        },
        {
            'input': '椒麻|小酥肉',
            'output': '风味|材料'
        },
        {
            'input': '正山|小种',
            'output': '地名|材料'
        }
    ]

def get_llm_with_prompt():
    system = """你是一个精通中文热爱美食的语言学家, 你需要帮忙给用户输入分词后的各个部分分类。""" \
    """可能的类别有以下几种：""" \
    """1. 材料: 表示是这个菜的主要原材料之一。例如`西红柿|炒|蛋`中的`西红柿`和`蛋`""" \
    """2. 形式: 表示这个菜的大类别。例如`牛肉|面`中的`面`""" \
    """3. 工艺: 表示这个菜制作工艺。例如`手撕|羊肉`中的`手撕`""" \
    """4. 风味: 表示这个菜的口味。例如`酸辣|土豆丝`中的`酸辣`""" \
    """5. 地名：表示这个菜的发源或者流行的地方。例如`重庆|小面`中的`重庆`""" \
    """6. 品牌：表示这个菜的发源或者流行的品牌。例如`正新|鸡排`中的`正新`""" \
    """7. 其他：表示这个词语不属于上述任何一类。""" \
    """回答之前，请确认如下几点""" \
    """1. 你给出的结果中，词语的数量是否和输入的词数一致。不要多加也不要遗漏。""" \
    """2. 输出的分类和输入分词结果一一对应，每个分类之间用|隔开。不要有空格"""\
    """3. 分类名要准确，不能出现以上7种分类之外的分类。""" \
    """4. 输出只有一行。""" \
    """5. 不要输出用户输入的内容""" \
    """6. 反复确认，不要有分类之外的任何内容！！！"""

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
        labels = result[1]
        print(_id, dish, labels)
        dishes[_id].labels = labels

    save_dishes(dishes, labeled_dishes, ['idx', 'text', 'tokens', 'labels'])

async def label_dishes(dishes: list[Dish]):
    llm = get_llm_with_prompt()
    candidates = [(i, dish.tokens) for i, dish in enumerate(dishes)]
    for i in range(0, len(candidates), BATCH_SIZE):
        await process_dishes(dishes, candidates[i:i+BATCH_SIZE], llm)

async def test():
    test_dishes = [
        "韭菜|猪肉|水饺",
        "阳光|柠柚|蛋糕",
        "酸菜|粉丝|汤",
        "莓果|提拉米苏",
        "奥尔良|鸡肉|披萨",
        "元气森林|氦苏打水"
    ]
    await label_dishes([Dish(idx=idx, text=dish, tokens=dish) for idx, dish in enumerate(test_dishes)])

async def main():
    dishes = load_dishes(tokenized_dishes)
    await label_dishes(dishes)

if __name__ == '__main__':
    asyncio.run(main())
