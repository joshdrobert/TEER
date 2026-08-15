import os
import subprocess
from moviepy import ImageClip, VideoFileClip, concatenate_videoclips, AudioFileClip
import shutil

scenes = [
    {
        "visual": "teer_hook.mp4",
        "text": "Mitral Regurgitation, or MR, is a debilitating condition that affects 1 in 10 adults over the age of 75. While traditional open-heart surgery is highly invasive, Transcatheter Edge-to-Edge Repair, or Tear, has revolutionized treatment with devices like the Mytra Clip. However, the procedure remains incredibly complex. Clinicians are forced to rely on intraoperative estimation, effectively guessing the optimal clip positioning. This leads to a high variability in results, leaving too many patients with residual regurgitation and suboptimal long-term outcomes."
    },
    {
        "visual": "slides_cropped/slide_04.png", 
        "text": "There is a critical, unmet clinical need to help clinicians determine the optimal Mytra Clip positioning before the procedure begins. That is exactly what MC-AURA does."
    },
    {
        "visual": "slides_cropped/slide_07.png", 
        "text": "Our Anatomy-Informed Ultrasound Repair Assistance platform bypasses manual modeling entirely. By extracting non-zero volumetric data directly from clinical Transesophageal Echocardiography—or T E E—scans, our system automatically generates high-fidelity, repaired PyVista meshes. This validates the technical feasibility of real-time clinical deployment."
    },
    {
        "visual": "slides_cropped/slide_09.png", 
        "text": "But we don't just map the anatomy; we simulate the physiology. We successfully completed preliminary Navier-Stokes flow simulations. And, after testing multiple Fluid-Structure Interaction frameworks like SimVascular and I B A M R, we are actively developing our V2 model utilizing F E ni C S and Dolphin X to ensure stable convergence at peak systolic pressures."
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
        "visual": "slides_cropped/slide_21.png", 
        "text": "In summary, MC-AURA transforms subjective guesswork into objective, data-driven precision. We are actively seeking pilot partners and funding to accelerate our physical phantom benchtop testing and clinical validation. Thank you for your time."
    }
]

def generate_tts(text, output_file):
    print(f"Generating F5-TTS for: {text[:30]}...")
    # Passing --device cpu to prevent Apple Silicon MPS segmentation faults
    # F5-TTS uses -o for output DIR and -w for output FILE name
    if os.path.exists(output_file) and os.path.isdir(output_file):
        shutil.rmtree(output_file)
        
    cmd = f'f5-tts_infer-cli --device cpu --model F5TTS_Base --gen_text "{text}" -o . -w {output_file}'
    subprocess.run(cmd, shell=True, check=True)

def generate_video():
    print("Generating AI voiceovers via F5-TTS (OmniVoice backend)...")
    audio_clips = []
    durations = []
    
    for i, scene in enumerate(scenes):
        audio_path = f"scene_v7_{i}.wav"
        if not os.path.exists(audio_path) or os.path.isdir(audio_path):
            generate_tts(scene["text"], audio_path)
            
        audio = AudioFileClip(audio_path)
        audio_clips.append(audio)
        durations.append(audio.duration + 0.5)

    print("Processing visuals...")
    video_clips = []
    
    for i, scene in enumerate(scenes):
        visual_path = scene["visual"]
        duration = durations[i]
        
        if visual_path.endswith(('.webm', '.gif', '.mp4', '.mov')):
            clip = VideoFileClip(visual_path)
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
    
    print("Rendering video...")
    final_video.write_videofile("Pitch_Demo_Video_v7.mp4", fps=24, preset="ultrafast")
    print("V7 Video generation complete.")

if __name__ == "__main__":
    generate_video()
