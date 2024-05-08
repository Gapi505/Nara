from llama_cpp import Llama
import gc
import torch
import yaml
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import sys
import asyncio


prompt_template = None
system_prompt_unconstructed = None
with open("templates/prompt_template.yaml", "r") as f:
    prompt_template = yaml.safe_load(f)
with open("templates/system_prompt.yaml", "r") as f:
    system_prompt_unconstructed = yaml.safe_load(f)

uncertan_conversation = ""
certan_conversation = ""

uncertan_message = ""
certan_message = ""


llm = None
transcriber = None


def free_memory():
    global llm, transcriber
    
    llm = None
    transcriber = None
    gc.collect()
    torch.cuda.empty_cache()

# Loads the model
def load_llm():
    global llm
    
    if llm is None:
        llm = Llama.from_pretrained(
            "PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed",
            filename="*Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=True,
        )

def load_transcriber():
    global transcriber
    transcriber = pipeline(
        "automatic-speech-recognition", model="openai/whisper-base.en", device="cuda"
    )

def construct_prompt(conversation):
    with open("templates/system_prompt.yaml", "r") as f:
        system_prompt_unconstructed = yaml.safe_load(f)
    config = system_prompt_unconstructed["config"].split(" ")

    system_prompt = ""
    for conf in config:
        system_prompt += system_prompt_unconstructed[conf] + "\n"
    
    prompt = prompt_template["prompt"].format(system_prompt=system_prompt,conversation=conversation)

    return prompt

def construct_message(user, message):
    fmessage = prompt_template["message"].format(user=user, message=message)
    return fmessage


async def transcribe(chunk_length_s=100.0, stream_chunk_s=0.5):
    global uncertan_message
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    previousTranscription = None
    repetitionCount = 0
    for transcription in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        sys.stdout.write("\033[K")
        print(transcription["text"], end="\r")
        uncertan_message = construct_message("user", transcription["text"])
        if previousTranscription and transcription["text"] == previousTranscription["text"]:
            repetitionCount += 1
        else:
            repetitionCount = 0
        
        if repetitionCount > 3:
            break
        if not transcription["partial"][0]:
            break
        previousTranscription = transcription

    return transcription["text"]

load_transcriber()
load_llm()

async def main_loop():
    global uncertan_conversation, certan_conversation, uncertan_message, certan_message
    transcriptionJob = asyncio.create_task(transcribe())
    print("Start conversation")
    while True:
        if transcriptionJob.done():
            transcription = transcriptionJob.result()
            print(transcription)
            break
        #print(uncertan_message)
        uncertan_conversation = certan_conversation + "\n" + uncertan_message
        uncertan_prompt = construct_prompt(uncertan_conversation)
        await asyncio.sleep(1)

asyncio.run(main_loop())