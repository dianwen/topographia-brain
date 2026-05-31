// Open a headed browser on an all-bot game (seeds the seat-0 viewer token, then navigates).
// Usage: node brain/scripts/watch_bot_game.mjs <gameId> <token> [port]
import { chromium } from 'playwright';

const [, , gameId, token, port = '3100'] = process.argv;
const base = `http://localhost:${port}`;

const browser = await chromium.launch({ headless: false, args: ['--window-size=1400,1000'] });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 1000 } });
const page = await ctx.newPage();
await page.goto(base);
await page.evaluate(([id, tok]) => sessionStorage.setItem(`game:${id}`, tok), [gameId, token]);
await page.goto(`${base}/?game=${gameId}`);
console.log('OPENED', `${base}/?game=${gameId}`);
await page.waitForTimeout(600000); // keep the window open ~10 min for viewing
await browser.close();
