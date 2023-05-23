import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def correct_text(model, tokenizer, input_text, max_length=512):
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True).input_ids
    input_ids = input_ids.to(device)
    output_ids = model.generate(input_ids, num_beams=50, temperature=10.0, output_scores=True)
    print(output_ids)
    corrected_text = tokenizer.decode(output_ids[0][3:], skip_special_tokens=True)
    return corrected_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("./trained_model").to(device)
tokenizer = T5Tokenizer.from_pretrained("./trained_model", model_max_length=512)

# Asegurarse de que el modelo esté en modo de evaluación
model.eval()

# Si está utilizando una GPU, mover el modelo a la GPU
if torch.cuda.is_available():
    model.cuda()

# Ejemplo de uso de la función correct_text
input_text = ""

corrected_text = correct_text(model, tokenizer, input_text)
print(f"Texto original: {input_text}")
print(f"Texto corregido: {corrected_text}")

