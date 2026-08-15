from playwright.sync_api import sync_playwright
import os
import shutil

def record_sites():
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
        
        # Record the main landing page
        page1 = context.new_page()
        page1.goto("https://joshdrobert.github.io/TEER/")
        page1.wait_for_timeout(2000)
        # Scroll down slightly to show stats
        page1.mouse.wheel(0, 500)
        page1.wait_for_timeout(3000)
        page1.mouse.wheel(0, 500)
        page1.wait_for_timeout(3000)
        page1.close()

        # Record the workspace demo
        page2 = context.new_page()
        page2.goto("https://joshdrobert.github.io/TEER/workspace.html")
        page2.wait_for_timeout(3000)
        # Simulate interacting with the page
        page2.mouse.move(960, 540)
        page2.mouse.down()
        page2.mouse.move(1200, 540, steps=20)
        page2.mouse.up()
        page2.wait_for_timeout(1000)
        
        # Click on something maybe? (optional)
        page2.wait_for_timeout(5000)
        page2.close()
        
        context.close()
        browser.close()

if __name__ == "__main__":
    record_sites()
