from kittentts import KittenTTS
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
import numpy as np
import re

model_name = "Qwen/Qwen3-0.6B"
m = KittenTTS("KittenML/kitten-tts-nano-0.2")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

# Prepare the model input
prompt = "Can you suggest me a typical Indian recipe?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# Parsing thinking content
try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking_content:", thinking_content)
print("content:", content)

# Text saving on file (fix path as needed)
file_path_txt = r'Text_answer.txt'
with open(file_path_txt, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Text_saved_in:{file_path_txt}")

# We split up the content using punctuation
phrases = re.split(r'[.!?\n]+', content)

audio_pieces = []

print("Generating_audio")
for phrase in phrases:
    filtered_phrase = phrase.replace("*", "")
    filtered_phrase = filtered_phrase.replace("#", "")

    # Clean up the phrase and check that it's not empty
    clean_phrase = filtered_phrase.strip()
    if len(clean_phrase) > 1:
        try:
            # Adds a pause to let the audio model generate the full sentence
            text_for_tts = clean_phrase + " ..."

            # Generate audio for a single piece
            audio_chunk = m.generate(text_for_tts, voice='expr-voice-2-f')
            audio_pieces.append(audio_chunk)

            # Adds a little silence between sentences
            silence = np.zeros(int(24000 * 0.2))
            audio_pieces.append(silence)

        except Exception as e:
            print(f"Error_on_phrase:{e}")

# Put together all the pieces in a single array
if audio_pieces:
    full_audio = np.concatenate(audio_pieces)

    # Save the audio (fix path as needed)
    output_path = r'Audio_answer.wav'
    sf.write(output_path, full_audio, 24000)
    print(f"Audio_correctly_saved_in:{output_path}")
else:
    print("No_audio_generated")
