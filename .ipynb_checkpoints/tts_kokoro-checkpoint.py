from kokoro import KPipeline
import soundfile as sf
from playsound import playsound
from train_tacotron import spectrogram
# Initialize the Kokoro pipeline for American English
pipeline = KPipeline(lang_code='a')  

# Input text for speech synthesis
text = spectrogram
generator = pipeline(text, voice='af_heart', speed=1)

# Process and save the audio, then play it
for i, (gs, ps, audio) in enumerate(generator):
    #print(f"Segment {i}: {gs}")
    file_name = f'segment_{i}.wav'
    sf.write(file_name, audio, 24000)
    playsound(file_name)
