class VLMDataCollator:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):

        texts = []
        images = []

        for item in examples:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {
                            "type": "text",
                            "text": item["question"],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": item["answer"],
                },
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            texts.append(text)
            images.append(item["image"])

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch[
            "input_ids"
        ].clone()

        pad_token_id = (
            self.processor
            .tokenizer
            .pad_token_id
        )

        labels[
            labels == pad_token_id
        ] = -100

        batch["labels"] = labels

        return batch