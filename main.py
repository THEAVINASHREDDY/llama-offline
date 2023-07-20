import transformers, torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=False,
        offload_folder="offload",
        offload_state_dict = True,
        torch_dtype=torch.float16,
        device_map="auto"
    )



instruction = "How old is the universe?"
inputs = tokenizer(
    f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: {instruction}
### Response:""",
    return_tensors="pt",
)
input_ids = inputs["input_ids"].to(model.device)

generation_config = transformers.GenerationConfig(
    do_sample=True,
    temperature=0.1,
    top_p=0.75,
    top_k=80,
    repetition_penalty=1.5,
    max_new_tokens=128,
)

with torch.no_grad():
    generation_output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=generation_config,
    )
output_text = tokenizer.decode(
    generation_output[0].cpu(), skip_special_tokens=True
).strip()
print(output_text)