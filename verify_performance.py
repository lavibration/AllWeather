import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("http://localhost:8501")
        await page.wait_for_timeout(5000)

        # Click on Performance tab
        # Tabs are often buttons or div with role="tab"
        # We look for text "Performance"
        await page.click("text=Performance")
        await page.wait_for_timeout(3000)

        await page.screenshot(path="/home/jules/verification/final_performance.png", full_page=True)
        await browser.close()

asyncio.run(run())
