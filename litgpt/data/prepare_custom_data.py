# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
import time
from pathlib import Path

from litgpt.tokenizer import Tokenizer
from litgpt.data.prepare_starcoder import DataChunkRecipe
from litgpt.utils import CLI, extend_checkpoint_dir
from transformers import AutoTokenizer

class CustomDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer

    def prepare_structure(self, input_dir):
        return [str(Path(input_dir) / "train.txt"), str(Path(input_dir) / "val.txt")]

    def prepare_item(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        text_ids = self.tokenizer.encode(text, bos=False, eos=True)
        for i in range(0, len(text_ids), self.chunk_size):
            yield text_ids[i:i+self.chunk_size]


def prepare(
    input_dir: Path = Path("data/custom_data/"),
    output_dir: Path = Path("data/processed_custom_data/"),
    tokenizer_path: Path = None,
    chunk_size: int = 2048,
    fast_dev_run: bool = False,
) -> None:
    from litdata.processing.data_processor import DataProcessor


    tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
    data_recipe = CustomDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":

    CLI(prepare)