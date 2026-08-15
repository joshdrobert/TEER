import os
from gtts import gTTS
from moviepy import ImageClip, VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, CompositeAudioClip

# Define the scenes with their corresponding text and visual
scenes = [
    {
        "visual": "final_presentation_extracted/ppt/media/image11.png", 
        "text": "Transcatheter Edge-to-Edge Repair, or TEER, has revolutionized the treatment of mitral regurgitation. However, clinical decision-making remains complex and highly variable, leaving many patients with suboptimal outcomes. Today, we present our platform designed to eliminate this guesswork.",
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image12.jpg", 
        "text": "We are targeting a critical unmet clinical need: giving interventional cardiologists precise, patient-specific hemodynamic data before they even enter the operating room. This reduces procedure times and improves diagnostic accuracy for thousands of patients.",
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image33.gif", 
        "text": "Let me show you how it works. Our AI-driven Clinical Decision Support System takes standard imaging and generates a patient-specific 3D simulation. Here, you see the real-time hemodynamic analysis of the mitral valve.",
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image34.gif", 
        "text": "Clinicians can interact with this model to test different MitraClip placement strategies. Our platform instantly predicts the resulting reduction in regurgitation and impact on pressure gradients, ensuring the best possible clip position.",
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image37.gif", 
        "text": "This isn't just a concept. Our prototype is fully functional, capable of processing patient data to deliver actionable insights. It streamlines clinical workflow by integrating seamlessly into the pre-operative planning phase.",
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image22.png", 
        "text": "Looking ahead, our go-to-market strategy focuses on partnering directly with structural heart programs and device manufacturers. By operating on a SaaS model, we ensure scalable deployment across hospital networks, making precision care accessible everywhere.",
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image1.png", 
        "text": "In summary, our platform solves a critical gap in TEER procedures by using advanced simulation to optimize clinical decision-making. We are seeking pilot partners and funding to accelerate our clinical validation. Thank you.",
    }
]

def generate_audio_and_durations():
    audio_clips = []
    durations = []
    
    for i, scene in enumerate(scenes):
        text = scene["text"]
        audio_path = f"scene_{i}.mp3"
        
        # Generate TTS
        tts = gTTS(text, lang='en', tld='com')
        tts.save(audio_path)
        
        # Load audio to get duration
        audio = AudioFileClip(audio_path)
        audio_clips.append(audio)
        
        # Add a small buffer of 0.5s after each scene
        durations.append(audio.duration + 0.5)
        
    return audio_clips, durations

def create_video():
    print("Generating voiceovers...")
    audio_clips, durations = generate_audio_and_durations()
    
    print("Processing visuals...")
    video_clips = []
    
    for i, scene in enumerate(scenes):
        visual_path = scene["visual"]
        duration = durations[i]
        
        if not os.path.exists(visual_path):
            print(f"Warning: {visual_path} not found. Using placeholder.")
            # fallback to whatever is there
            continue
            
        if visual_path.endswith('.gif') or visual_path.endswith('.mp4'):
            # Load gif, loop if needed
            clip = VideoFileClip(visual_path)
            if clip.duration < duration:
                # Need to loop it
                import moviepy.video.fx as vfx
                clip = clip.with_effects([vfx.Loop(duration=duration)])
            else:
                clip = clip.subclipped(0, duration)
        else:
            # It's an image
            clip = ImageClip(visual_path).with_duration(duration)
            
        # Resize to standard 1080p width/height to avoid concatenation issues
        clip = clip.resized(width=1920, height=1080)
        
        # Set audio
        clip = clip.with_audio(audio_clips[i])
        video_clips.append(clip)
        
    print("Concatenating clips...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    total_duration = final_video.duration
    print(f"Total video duration: {total_duration} seconds ({total_duration/60:.2f} minutes)")
    
    if total_duration > 240:
        print("WARNING: Video is over 4 minutes long!")
        
    print("Rendering video...")
    final_video.write_videofile("Pitch_Demo_Video.mp4", fps=24, preset="ultrafast")
    
    # Clean up audio files
    for i in range(len(scenes)):
        try:
            os.remove(f"scene_{i}.mp3")
        except:
            pass
            
    print("Done!")

if __name__ == "__main__":
    create_video()
