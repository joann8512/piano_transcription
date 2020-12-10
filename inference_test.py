from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import os
import time


def main():
    st = time.time()
    song = "./audio/seg/Q4_aQ4PIaRMLwE_0.mp3"
    
    # Load audio
    print("### Loading ###")
    (audio, _) = load_audio(song, sr=sample_rate, mono=True)

    # Transcriptor
    print("### Transcriptor ###")
    transcriptor = PianoTranscription(device='cuda')    # 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    print("### Transcribing/Writing ###")
    name = song.split(".")[1].split("/")[-1]
    transcribed_dict = transcriptor.transcribe(audio, os.path.join("./output", name+".mid"))
    ed = time.time()
    
    print("### DONE ###")
    print("Took "+str(ed-st)+" seconds to process.")
    
    
if __name__ == "__main__":
    main()