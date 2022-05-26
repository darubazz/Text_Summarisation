import logging
import typer

from pathlib import Path
from typing import Optional

from luhn_summarizer import LuhnSummarizer

logging.basicConfig(filename='./logs/app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filemode='a')

app = typer.Typer()
summarizer = LuhnSummarizer()


@app.command()
def summarize(text_path: Optional[Path]):
    with open(text_path, "r", encoding='utf-8') as f:
        text = "".join(f.readlines())

    sum = summarizer.summarize(text)

    # saving results
    save_path = Path("./data/results/result_text_ru.txt")
    with open(save_path, "w") as f:
        f.writelines(sum)

    logging.info(f"Summarization successes. Text from {text_path} saved to {save_path}")


if __name__ == '__main__':
    app()
