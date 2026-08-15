from playwright.sync_api import sync_playwright
import os
import shutil

def record_workspace():
    video_dir = "videos"
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
        
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # We set record_video_dir to capture the video
        context = browser.new_context(
            record_video_dir=video_dir, 
            record_video_size={"width": 1920, "height": 1080},
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        file_url = f"file://{os.path.abspath('workspace.html')}"
        page.goto(file_url)
        
        # Wait for the model to load
        page.wait_for_timeout(3000)
        
        # Simulate interacting with the page by moving the mouse around
        page.mouse.move(960, 540)
        page.mouse.down()
        page.mouse.move(1200, 540, steps=20)
        page.mouse.up()
        
        page.wait_for_timeout(1000)
        
        # Rotate back
        page.mouse.move(1200, 540)
        page.mouse.down()
        page.mouse.move(960, 540, steps=20)
        page.mouse.up()
        
        # Wait to capture the interactions
        page.wait_for_timeout(8000)
        
        context.close()
        browser.close()

if __name__ == "__main__":
    record_workspace()
