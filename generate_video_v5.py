import os
import asyncio
from moviepy import ImageClip, VideoFileClip, concatenate_videoclips, AudioFileClip

scenes = [
    {
        "visual": "teer_hook.mp4",
        "text": "Mitral Regurgitation, or MR, is a debilitating condition that affects 1 in 10 adults over the age of 75. While traditional open-heart surgery is highly invasive, Transcatheter Edge-to-Edge Repair, or TEER, has revolutionized treatment with devices like the MitraClip. However, the procedure remains incredibly complex. Clinicians are forced to rely on intraoperative estimation, effectively guessing the optimal clip positioning. This leads to a high variability in results, leaving too many patients with residual regurgitation and suboptimal long-term outcomes."
    },
    {
        "visual": "slides_cropped/slide_04.png", 
        "text": "There is a critical, unmet clinical need to help clinicians determine the optimal MitraClip positioning before the procedure begins. That is exactly what MC-AURA does."
    },
    {
        "visual": "slides_cropped/slide_07.png", 
        "text": "Our Anatomy-Informed Ultrasound Repair Assistance platform bypasses manual modeling entirely. By extracting non-zero volumetric data directly from clinical Transesophageal Echocardiography—or TEE—scans, our system automatically generates high-fidelity, repaired PyVista meshes. This validates the technical feasibility of real-time clinical deployment."
    },
    {
        "visual": "slides_cropped/slide_09.png", 
        "text": "But we don't just map the anatomy; we simulate the physiology. We successfully completed preliminary Navier-Stokes flow simulations. And, after testing multiple Fluid-Structure Interaction frameworks like SimVascular and IBAMR, we are actively developing our V2 model utilizing FEniCS and DolphinX to ensure stable convergence at peak systolic pressures."
    },
    {
        "visual": "landing_demo_fixed.mp4", 
        "text": "All of this complex computation is abstracted away behind our intuitive web interface. Our AI-driven Clinical Decision Support System provides interventional cardiologists with actionable data."
    },
    {
        "visual": "demo_fixed.mp4", 
        "text": "Let's take a look at the interactive demonstration. Clinicians can interact with the patient-specific 3D model, applying virtual clips across distinct spatial zones in silico. As you can see, our pipeline instantly predicts the resulting reduction in regurgitant volume and tracks hemodynamic changes over time, including left atrial and ventricular pressures."
    },
    {
        "visual": "workspace_demo_fixed.mp4", 
        "text": "This isn't just theoretical. The functional demonstration provides the verified training dataset necessary for our downstream machine learning optimization algorithms. We are building a system that predicts the best outcome instantly, standardizing care across the board."
    },
    {
        "visual": "slides_cropped/slide_20.png", 
        "text": "We are the MC-AURA team—Rucha, Cyril, and Joshua. Under the guidance of our mentors at Houston Methodist and Texas A&M EnMed, we are working to integrate this into the clinical workflow through a SaaS model and virtual reality integrations with surgical hardware."
    },
    {
        "visual": "slides_cropped/slide_21.png", 
        "text": "In summary, MC-AURA transforms subjective guesswork into objective, data-driven precision. We are actively seeking pilot partners and funding to accelerate our physical phantom benchtop testing and clinical validation. Thank you for your time."
    }
]

def generate_tts(text, output_file, kokoro):
    import soundfile as sf
    print(f"Generating TTS for: {text[:30]}...")
    # Switch to af_bella (one of the most realistic female voices) with slightly slower speed for realism
    samples, sample_rate = kokoro.create(text, voice="af_bella", speed=0.9, lang="en-us")
    sf.write(output_file, samples, sample_rate)

def generate_video():
    print("Initializing Kokoro TTS...")
    from kokoro_onnx import Kokoro
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
    
    print("Generating AI voiceovers via Kokoro...")
    audio_clips = []
    durations = []
    
    for i, scene in enumerate(scenes):
        audio_path = f"scene_v5_{i}.wav"
        generate_tts(scene["text"], audio_path, kokoro)
        
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
            
        if visual_path.endswith(('.webm', '.gif', '.mp4', '.mov')):
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
    final_video.write_videofile("Pitch_Demo_Video_v5.mp4", fps=24, preset="ultrafast")
    
    # Cleanup audio
    for i in range(len(scenes)):
        try:
            os.remove(f"scene_v5_{i}.wav")
        except:
            pass
    print("V5 Video generation complete.")

if __name__ == "__main__":
    generate_video()
