import os
import asyncio
from moviepy import ImageClip, VideoFileClip, concatenate_videoclips, AudioFileClip

scenes = [
    {
        "visual": "teer_hook.mp4",
        "text": "Mitral Regurgitation affects 1 in 10 adults over 75, making it one of the most common valve diseases. While Transcatheter Edge-to-Edge Repair, or TEER, has revolutionized treatment with devices like the MitraClip, the procedure remains highly complex. Currently, clinical decision-making relies on intraoperative estimation, leaving too many patients with suboptimal outcomes."
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image23.png", 
        "text": "There is a critical unmet need for precision guidance. That's where MC-AURA comes in. We bypass manual modeling by extracting data directly from clinical TEE scans to generate high-fidelity PyVista meshes."
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image34.gif", 
        "text": "This is our AI-driven Clinical Decision Support System. Here, you see the real-time hemodynamic analysis of the mitral valve in our web application."
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image37.gif", 
        "text": "Clinicians can interact with this patient-specific 3D model to test different MitraClip placement strategies in silico. Our fluid-structure interaction pipeline instantly predicts the resulting reduction in regurgitation and mechanical stress."
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image22.png", 
        "text": "Looking ahead, our go-to-market strategy focuses on partnering directly with structural heart programs and device manufacturers. By operating on a SaaS model, we ensure scalable deployment across hospital networks, making precision care accessible everywhere."
    },
    {
        "visual": "final_presentation_extracted/ppt/media/image15.png", 
        "text": "In summary, MC-AURA solves the critical gap in TEER procedures by using advanced simulation to optimize clinical decision-making. We are seeking pilot partners and funding to accelerate our physical phantom benchtop testing and clinical validation. Thank you."
    }
]

async def generate_tts(text, output_file):
    import edge_tts
    # High-quality professional voice
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_file)

def generate_video():
    print("Generating AI voiceovers via edge-tts...")
    audio_clips = []
    durations = []
    
    for i, scene in enumerate(scenes):
        audio_path = f"scene_{i}.mp3"
        asyncio.run(generate_tts(scene["text"], audio_path))
        
        audio = AudioFileClip(audio_path)
        audio_clips.append(audio)
        durations.append(audio.duration + 0.5)

    print("Processing visuals...")
    video_clips = []
    
    for i, scene in enumerate(scenes):
        visual_path = scene["visual"]
        duration = durations[i]
        
        if not os.path.exists(visual_path):
            print(f"Warning: {visual_path} not found. Skipping...")
            continue
            
        if visual_path.endswith('.gif') or visual_path.endswith('.mp4'):
            clip = VideoFileClip(visual_path)
            # Remove existing audio from video if any
            if getattr(clip, "audio", None) is not None:
                clip = clip.without_audio()
                
            if clip.duration < duration:
                import moviepy.video.fx as vfx
                clip = clip.with_effects([vfx.Loop(duration=duration)])
            else:
                clip = clip.subclipped(0, duration)
        else:
            clip = ImageClip(visual_path).with_duration(duration)
            
        clip = clip.resized(width=1920, height=1080)
        clip = clip.with_audio(audio_clips[i])
        video_clips.append(clip)
        
    print("Concatenating clips...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    total_duration = final_video.duration
    print(f"Total video duration: {total_duration} seconds ({total_duration/60:.2f} minutes)")
    
    print("Rendering video...")
    final_video.write_videofile("Pitch_Demo_Video_v2.mp4", fps=24, preset="ultrafast")
    
    # Cleanup audio
    for i in range(len(scenes)):
        try:
            os.remove(f"scene_{i}.mp3")
        except:
            pass

if __name__ == "__main__":
    generate_video()
